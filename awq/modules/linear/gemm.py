"""
AWQ GEMM量化线性层实现

该模块实现了GEMM（General Matrix-Matrix Multiplication）版本的AWQ量化线性层,
专门针对大批量矩阵-矩阵乘法进行优化。GEMM适用于批量推理和训练场景,
支持梯度计算,是AWQ中功能最全面的量化线性层实现。

核心特性：
- 4位权重量化,节省75%内存
- 支持大批量处理,优化吞吐量
- 完整的梯度计算支持,可用于训练
- 多后端支持：CUDA扩展、Triton内核、朴素实现
- 智能后端选择机制

适用场景：
- 大批量推理服务
- 模型微调和训练
- 批处理任务
- 需要梯度计算的场景

后端优先级：
1. awq_ext CUDA扩展（最高性能）
2. Triton内核（跨平台支持）
3. 朴素实现（兼容性最佳）
"""

# 导入必要的库
import torch  # PyTorch主库,张量操作和自动微分
import warnings  # 警告信息处理
import torch.nn as nn  # PyTorch神经网络模块
from torch.autograd import Function  # 自定义自动微分函数，用于实现前向和反向传播
from awq.utils.module import try_import  # 动态导入扩展模块，用于安全导入可选依赖
from awq.utils.utils import get_best_device  # 获取最佳计算设备，支持CPU、CUDA、MPS等
from awq.utils.packing_utils import dequantize_gemm  # 朴素反量化实现，用于无加速后端时的兼容性

# ==================== 扩展模块导入 ====================
# 注意：我们检查awq_ext或triton是否可用。如果两者都安装,优先使用awq_ext。

# 尝试导入AWQ CUDA扩展模块，提供最高性能的量化计算
awq_ext, msg = try_import("awq_ext")
user_has_been_warned = False  # 用于防止重复警告

# 尝试导入Triton内核，提供跨平台的高性能实现
try:
    from awq.modules.triton.gemm import awq_gemm_triton, awq_dequantize_triton
    # Triton支持CUDA、ROCm和XPU。如果能导入triton,就可以使用它。
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# 改编自 https://github.com/compressa-ai/AutoAWQ/tree/dev
class WQLinearMMFunction(Function):
    """
    GEMM量化线性层的自动微分函数

    该类实现了支持梯度的量化矩阵乘法,是GEMM版本的核心。
    它继承了PyTorch的Function类,自定义了前向和反向传播逻辑,
    使得量化层能够参与训练过程。

    特性：
        - 支持前向传播的多种后端
        - 完整的梯度计算实现
        - 智能后端选择策略
        - 内存优化的计算流程

    后端选择逻辑：
        1. 优先使用awq_ext CUDA扩展（最高性能）
        2. 其次使用Triton内核（跨平台支持）
        3. 最后使用朴素实现（兼容性）

    性能启发式：
        - 大矩阵使用反量化+matmul策略
        - 小矩阵使用专用量化内核
    """

    @staticmethod
    def forward(
        ctx,
        x:torch.Tensor,      # x.shape: [*, in_features]
        qweight,             # qweight.shape: [in_features, out_features//8]
        qzeros,              # qzeros.shape: [in_features//group_size, out_features//8]
        scales,              # scales.shape: [in_features//group_size, out_features]
        w_bit=4,
        group_size=128,
        bias=None,           # bias.shape: [out_features] or None
        out_features=0,
    ):
        """
        前向传播实现

        执行量化矩阵-矩阵乘法,支持多种计算后端和性能优化策略。
        根据输入大小智能选择最优的计算方式。

        Args:
            ctx: 自动微分上下文,用于保存反向传播所需信息
            x (torch.Tensor): 输入张量
            qweight (torch.Tensor): 量化权重
            qzeros (torch.Tensor): 量化零点
            scales (torch.Tensor): 缩放因子
            w_bit (int): 量化位数,默认4
            group_size (int): 分组大小,默认128
            bias (torch.Tensor): 可选偏置
            out_features (int): 输出特征维度

        Returns:
            torch.Tensor: 量化线性变换的结果

        计算策略：
            - 大矩阵 (>1024元素): 反量化 + torch.matmul
            - 小矩阵 (≤1024元素): 专用量化内核
        """
        # 前向传播可以使用ctx保存反向传播需要的信息
        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features

        # 计算输出形状
        out_shape = x.shape[:-1] + (out_features,)
        # 确保输入为FP16格式
        x = x.to(torch.float16)

        # 处理空输入的特殊情况
        if x.shape[0] == 0:
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # ==================== 多后端计算策略 ====================

        # 策略1：使用AWQ CUDA扩展（最高性能）
        if awq_ext is not None:
            # FP16矩阵乘法的启发式条件：矩阵元素数 >= 1024
            FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024

            if FP16_MATMUL_HEURISTIC_CONDITION:
                # 大矩阵：反量化 + 标准矩阵乘法
                # 反量化权重回FP16格式, out.shape: (in_features, out_features)
                out = awq_ext.dequantize_weights_cuda(
                    qweight, scales, qzeros, 0, 0, 0, False
                )
                # 使用cuBLAS优化的矩阵乘法, out: (..., out_features)
                out = torch.matmul(x, out)
            else:
                # 小矩阵：使用专用的量化GEMM内核
                # x.reshape(-1, x.shape[-1]): (x.shape[0]*...*1, in_features), out: (reshaped.shape[0], out_features)
                out = awq_ext.gemm_forward_cuda(
                    x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, 8
                )

        # 策略2：使用Triton内核（跨平台支持）
        elif TRITON_AVAILABLE:
            FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024

            if FP16_MATMUL_HEURISTIC_CONDITION:
                # 大矩阵：Triton反量化 + PyTorch矩阵乘法
                out = awq_dequantize_triton(qweight, scales, qzeros)
                out = torch.matmul(x, out.to(x.dtype))
            else:
                # 小矩阵：Triton专用量化内核
                out = awq_gemm_triton(
                    x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, split_k_iters=8,
                )

        # 策略3：朴素实现（兼容性最佳,但性能较慢）
        else:
            global user_has_been_warned
            if not user_has_been_warned:
                warnings.warn("使用朴素（慢速）实现。" + msg)
                user_has_been_warned = True
            # 完全在Python中执行反量化和矩阵乘法
            out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
            out = torch.matmul(x, out)

        # ==================== 后处理 ====================
        # 添加偏置
        out = out + bias if bias is not None else out
        # 恢复输出形状
        out = out.reshape(out_shape)

        # 确保输出始终是3D张量（如果输入是2D则添加batch维度）
        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播实现

        计算量化线性层的梯度,支持权重共享机制。
        由于权重是量化的,梯度只传播到输入,权重参数通过其他方式更新。

        Args:
            ctx: 前向传播保存的上下文信息
            grad_output (torch.Tensor): 输出梯度, shape: [*, out_features]

        Returns:
            tuple: (输入梯度, 权重梯度, 零点梯度, 缩放因子梯度, ...)
                  只有输入需要梯度,其他参数返回None

        梯度计算：
            - 反量化权重到原始精度
            - 计算输入梯度：grad_input = grad_output × weight^T
            - 使用批处理矩阵乘法优化性能

        注意：
            量化参数的更新通常通过量化感知训练(QAT)或
            其他专门的量化优化方法处理,不通过标准反向传播。
        """
        # 从上下文中恢复前向传播保存的张量
        input, qweight, qzeros, scales, bias = ctx.saved_tensors
        # input.shape: [*, in_features]
        # qweight.shape: [in_features, out_features//8]
        # qzeros.shape: [in_features//group_size, out_features//8]
        # scales.shape: [in_features//group_size, out_features]
        # bias.shape: [out_features] or None

        # 检查是否有可用的加速后端，如果没有则抛出异常
        if awq_ext is None and not TRITON_AVAILABLE:
            raise ValueError(
                "需要安装triton或autoawq-kernels才能使用`.backward()`。请按照安装指南安装："
                "https://github.com/casper-hansen/AutoAWQ_kernels"
            )

        # ==================== 梯度计算 ====================
        # 将权重反量化回原始精度用于梯度计算
        # Cast to correct dtype for mixed precision training
        if awq_ext is not None:
            # 使用CUDA扩展进行反量化
            weights = awq_ext.dequantize_weights_cuda(
                qweight, scales, qzeros, 1, 0, 0, False  # 参数1表示反向传播模式
            ).to(grad_output.dtype)  # weights.shape: [in_features, out_features]
        else:
            # 使用Triton内核进行反量化
            weights = awq_dequantize_triton(
                qweight, scales, qzeros
            ).to(grad_output.dtype)  # weights.shape: [in_features, out_features]

        # 计算输入梯度（如果需要）
        if ctx.needs_input_grad[0]:
            # 3D矩阵乘法使用torch.bmm：https://pytorch.org/docs/stable/generated/torch.bmm.html
            # 在所有批量大小上传播梯度
            batch_size = grad_output.shape[0]
            # grad_input = grad_output × weights^T
            # 使用批处理矩阵乘法计算输入梯度
            grad_input = grad_output.bmm(weights.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1))
            # grad_input.shape: [batch_size, *, in_features]

        # 返回梯度：只有输入需要梯度,其他参数返回None
        # 返回的元组必须与forward方法的参数顺序对应
        return grad_input, None, None, None, None, None, None, None

class WQLinear_GEMM(nn.Module):
    """
    AWQ GEMM量化线性层类

    该类实现了基于GEMM的AWQ量化线性变换,专门为大批量矩阵-矩阵乘法优化。
    与GEMV版本不同,GEMM版本支持梯度计算,可以用于训练场景。

    Attributes:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        w_bit (int): 量化位数,当前仅支持4位
        group_size (int): 量化分组大小
        training (bool): 是否处于训练模式
        qweight (torch.Tensor): 量化后的权重张量
        qzeros (torch.Tensor): 量化后的零点张量
        scales (torch.Tensor): 缩放因子张量
        bias (torch.Tensor): 可选的偏置张量

    与GEMV的区别：
        - 支持梯度计算和训练
        - 权重布局不同：转置形式以优化批处理
        - 多后端支持：CUDA扩展、Triton、朴素实现
        - 吞吐量优化而非延迟优化

    适用场景：
        - 大批量推理
        - 模型微调训练
        - 批处理服务
        - 量化感知训练
    """

    def __init__(
        self, w_bit, group_size, in_features, out_features, bias, dev, training=False
    ):
        """
        初始化GEMM量化线性层

        Args:
            w_bit (int): 量化位数,当前仅支持4位
            group_size (int): 量化分组大小,-1表示按通道量化
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            bias (bool): 是否使用偏置
            dev (str): 设备类型（'cpu'或'cuda'）
            training (bool): 是否启用训练模式,支持梯度计算,QAT模式

        Raises:
            NotImplementedError: 当w_bit不是4时抛出异常

        设计考虑：
            - 训练模式支持梯度计算
            - 权重布局针对批处理优化
            - 多设备兼容性
        """
        super().__init__()

        # 目前仅支持4位量化
        if w_bit not in [4]:
            raise NotImplementedError("目前仅支持4位量化。")

        # 基本属性设置
        self.in_features = in_features      # 输入维度
        self.out_features = out_features    # 输出维度
        self.w_bit = w_bit                  # 量化位数
        # 如果group_size为-1，则按通道量化（整个输入特征作为一个组）
        self.group_size = group_size if group_size != -1 else in_features
        self.training = training            # 训练模式标志

        # ==================== 内存对齐检查 ====================
        # 确保输入特征数能被分组大小整除，保证量化分组的完整性
        assert self.in_features % self.group_size == 0, "输入特征必须能被分组大小整除"
        # 确保输出特征数能被权重打包单位整除，保证32位整数能完整存储权重
        assert out_features % (32 // self.w_bit) == 0, "输出特征必须能被打包单位整除"

        # ==================== 初始化量化参数缓冲区 ====================
        # GEMM的权重布局：转置形式 [in_features, out_features//8] 优化批处理
        # 使用register_buffer注册不参与训练更新的参数
        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),  # 转置布局，优化批处理访问模式
                dtype=torch.int32,  # 32位整数存储打包的4位权重
                device=dev,  # 指定计算设备
            ),
        )  # qweight.shape: [in_features, out_features//8]

        # 零点布局：[group_count, out_features//8]
        # 零点按分组存储，每组对应一个零点值
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,  # 32位整数存储打包的4位零点
                device=dev,  # 指定计算设备
            ),
        )  # qzeros.shape: [in_features//group_size, out_features//8]

        # 缩放因子布局：[group_count, out_features]
        # 缩放因子按分组存储，每组对应多个输出特征的缩放因子
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,  # 使用FP16存储缩放因子以节省内存
                device=dev,  # 指定计算设备
            ),
        )  # scales.shape: [in_features//group_size, out_features]

        # 可选偏置：如果需要偏置则注册为缓冲区，否则设为None
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((out_features), dtype=torch.float16, device=dev),
            )  # bias.shape: [out_features]
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear:nn.Linear, w_bit, group_size, init_only=False, scales:torch.Tensor=None, zeros:torch.Tensor=None
    ):
        """
        从标准线性层创建GEMM量化线性层

        该方法将标准的浮点线性层转换为量化线性层，通过量化权重并存储量化参数。
        支持仅初始化模式（用于加载预训练权重）和完整量化模式。

        Args:
            linear (nn.Linear): 标准线性层
            w_bit (int): 量化位数，当前仅支持4位
            group_size (int): 量化分组大小
            init_only (bool): 是否仅初始化，不进行实际量化
            scales (torch.Tensor): 预计算的缩放因子, shape: [in_features//group_size, out_features]
            zeros (torch.Tensor): 预计算的零点值, shape: [in_features//group_size, out_features]

        Returns:
            WQLinear_GEMM: 量化后的线性层实例
        """
        # 创建量化线性层实例，此时才初始化这个类
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear
        # 实际量化需要缩放因子和零点信息
        assert scales is not None and zeros is not None
        # 计算缩放零点（零点*缩放因子）
        scale_zeros = zeros * scales

        # 如果仅初始化模式，直接返回实例（用于加载权重）

        # 复制缩放因子和偏置到量化层
        awq_linear.scales = scales.clone().half()  # scales.shape: [in_features//group_size, out_features]
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()  # bias.shape: [out_features]

        # 计算每个打包单元的权重数量（32位整数中包含的权重数）
        pack_num = 32 // awq_linear.w_bit

        # 量化权重计算
        intweight = []  # intweight: List of tensors, each with shape [out_features, 1]
        # 遍历每个输入特征维度
        for idx in range(awq_linear.in_features):
            # 对权重进行量化：(权重 + 缩放零点) / 缩放因子，然后四舍五入为整数
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[idx // group_size])
                    / awq_linear.scales[idx // group_size]
                ).to(torch.int)[:, None]
            )
        # 拼接所有量化权重
        intweight = torch.cat(intweight, dim=1)  # intweight.shape: [out_features, in_features]
        # 转置并确保内存连续
        intweight = intweight.t().contiguous()
        # 转换为32位整数类型
        intweight = intweight.to(dtype=torch.int32)

        # 获取最佳计算设备
        best_device = get_best_device()

        # 避免MPS设备上不支持的操作
        # Avoid: The operator 'aten::__lshift__.Scalar' is not currently implemented for the MPS device
        if "mps" in best_device:
            intweight = intweight.to("cpu")

        # 初始化量化权重张量
        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=intweight.device,
        )  # qweight.shape: [out_features, in_features//8]

        # 按列打包量化权重
        for col in range(intweight.shape[1] // pack_num):
            # 4位量化使用特定的重排顺序
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            # 将多个权重打包到一个32位整数中
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        # 保存量化权重
        awq_linear.qweight = qweight #__init__函数中的qweight

        # 处理零点数据类型和设备
        zeros = zeros.to(dtype=torch.int32, device=best_device)

        # 避免MPS设备上不支持的操作
        if "mps" in best_device:
            zeros = zeros.to("cpu")

        # 初始化量化零点张量
        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=zeros.device,
        )  # qzeros.shape: [in_features//group_size, out_features//8]

        # 按列打包量化零点
        for col in range(zeros.shape[1] // pack_num):
            # 4位量化使用特定的重排顺序
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            # 将多个零点打包到一个32位整数中
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        # 保存量化零点
        awq_linear.qzeros = qzeros

        return awq_linear

    def forward(self, x):
        """
        GEMM量化线性层的前向传播

        支持训练和推理两种模式,通过自定义Function实现。
        训练模式支持梯度计算,推理模式禁用梯度以提高性能。

        Args:
            x (torch.Tensor): 输入张量,形状为[*, in_features]

        Returns:
            torch.Tensor: 输出张量,形状为[*, out_features]
        """
        # 计算输出形状
        out_shape = x.shape[:-1] + (self.out_features,)  # out_shape: [*, out_features]

        # 数据类型处理：保存原始数据类型并转换为FP16
        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        # 根据训练模式选择梯度计算方式
        if self.training:
            # 训练模式：启用梯度计算，使用自定义的WQLinearMMFunction
            out = WQLinearMMFunction.apply(
                x, self.qweight, self.qzeros, self.scales,
                self.w_bit, self.group_size, self.bias, self.out_features,
            )
        else:
            # 推理模式：禁用梯度以提高性能，使用torch.no_grad()上下文管理器
            with torch.no_grad():
                out = WQLinearMMFunction.apply(
                    x, self.qweight, self.qzeros, self.scales,
                    self.w_bit, self.group_size, self.bias, self.out_features,
                )

        # 恢复原始数据类型（如果不是FP16）
        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        # 重塑输出张量形状并返回
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        """
        返回模块的字符串表示

        该方法用于在打印模块时显示关键参数信息，包括输入输出特征数、
        是否使用偏置、量化位数和分组大小。

        Returns:
            str: 模块参数的字符串表示
        """
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features, self.out_features, self.bias is not None,
                self.w_bit, self.group_size,
            )
        )
