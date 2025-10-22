"""
AWQ GEMV量化线性层实现

该模块实现了GEMV（General Matrix-Vector Multiplication）版本的AWQ量化线性层，
专门针对单向量或小批量矩阵-向量乘法进行优化。GEMV适用于文本生成中的逐token推理场景，
通过4位权重量化显著减少内存占用和计算延迟。

核心特性：
- 4位权重量化，节省75%内存
- 专门的CUDA内核优化
- 支持分组量化（group_size）
- 智能内核选择（大/小批量）
- 高效的反量化计算

适用场景：
- 文本生成的逐token解码
- 小批量推理
- 内存受限的边缘设备
- 延迟敏感的实时应用
"""

# 导入必要的库
import torch  # PyTorch主库，张量操作和自动微分
import warnings  # 警告信息处理
import torch.nn as nn  # PyTorch神经网络模块
from awq.utils.module import try_import  # 动态导入扩展模块的工具函数

# 尝试导入AWQ扩展模块，该模块包含了优化的CUDA内核
awq_ext, msg = try_import("awq_ext")

def make_divisible(c, divisor):
    """
    确保数值能被除数整除的辅助函数

    该函数计算使得c能够被divisor整除的最小整数，
    用于内存对齐和批量处理优化。

    Args:
        c (int): 原始数值
        divisor (int): 目标除数

    Returns:
        int: 能被divisor整除的最小整数

    Example:
        make_divisible(10, 8) = 2  # 10需要增加2才能被8整除(12)
    """
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    """
    计算零点存储所需的宽度

    该函数根据输入特征数、分组大小和打包数量计算零点张量的宽度，
    确保内存布局的优化和对齐。

    Args:
        in_features (int): 输入特征维度
        group_size (int): 量化分组大小，默认128
        pack_num (int): 每个int32打包的量化值数量，默认8（对应4bit）

    Returns:
        int: 计算得到的零点宽度

    设计原理：
        - 不同group_size需要不同的内存对齐策略
        - pack_num决定了每个int32能存储多少个量化值
        - 确保CUDA内核访问的内存对齐
    """
    # 根据不同的分组大小设置相应的倍数因子
    # 较小的group_size需要更大的size_multiplier来保持内存对齐
    if group_size >= 128:
        size_multiplier = 1  # 标准分组，无需额外倍数
    elif group_size == 64:
        size_multiplier = 2  # 小分组，需要2倍对齐
    elif group_size == 32:
        size_multiplier = 4  # 更小分组，需要4倍对齐
    else:
        raise NotImplementedError("不支持的group_size，仅支持32、64、128")

    # 计算基础宽度：确保能被pack_num整除
    base_width = make_divisible(in_features // group_size, pack_num)
    # 再确保能被size_multiplier整齐，并应用倍数
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


class WQLinear_GEMV(nn.Module):
    """
    AWQ GEMV量化线性层类

    该类实现了基于GEMV的AWQ量化线性变换，专门为单向量或小批量场景优化。
    通过4位权重量化显著减少内存占用，同时使用专门的CUDA内核确保推理效率。

    Attributes:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        w_bit (int): 量化位数，当前仅支持4位
        group_size (int): 量化分组大小
        split_k_iters (int): 分块计算迭代次数，用于大矩阵优化
        qweight (torch.Tensor): 量化后的权重张量
        qzeros (torch.Tensor): 量化后的零点张量
        scales (torch.Tensor): 缩放因子张量
        bias (torch.Tensor): 可选的偏置张量

    内存优化：
        - 4位权重量化：节省75%内存
        - 分组量化：平衡精度和效率
        - 高效打包：充分利用int32存储空间

    性能优化：
        - 专用CUDA内核：针对GEMV场景优化
        - 智能分块：大矩阵的分块计算策略
        - 内存对齐：确保最优的访问模式
    """

    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        """
        初始化GEMV量化线性层

        Args:
            w_bit (int): 量化位数，当前仅支持4位
            group_size (int): 量化分组大小，-1表示按通道量化
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            bias (bool): 是否使用偏置
            dev (str): 设备类型（'cpu'或'cuda'）

        Raises:
            NotImplementedError: 当w_bit不是4时抛出异常

        设计考虑：
            - group_size控制量化精度和内存效率的平衡
            - 内存对齐确保CUDA内核的最优性能
            - 设备管理支持CPU和GPU部署
        """
        super().__init__()

        # 目前仅支持4位量化，这是AWQ的标准配置
        if w_bit not in [4]:
            raise NotImplementedError("目前仅支持4位量化。")

        # 基本属性设置
        self.in_features = in_features      # 输入维度
        self.out_features = out_features    # 输出维度
        self.w_bit = w_bit                  # 量化位数
        # 分组大小：-1表示按通道量化，否则使用指定分组大小
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8              # 分块计算迭代次数，优化大矩阵计算

        # ==================== 内存对齐检查 ====================
        # 确保输入特征能被分组大小整除，保证分组量化的完整性
        assert self.in_features % self.group_size == 0, "输入特征必须能被分组大小整除"
        # 确保输出特征能被打包单位整除，保证4位打包的完整性
        assert out_features % (32 // self.w_bit) == 0, "输出特征必须能被打包单位整除"

        # 计算每个int32能打包的4位量化值数量：32/4 = 8
        pack_num = 32 // self.w_bit

        # ==================== 初始化量化参数缓冲区 ====================

        # 量化权重缓冲区：存储4位打包的权重
        # 形状：[输出特征, 输入特征/8] (每个int32存储8个4位权重)
        self.register_buffer(
            "qweight",
            torch.zeros(
                (out_features, in_features // pack_num), dtype=torch.int32, device=dev
            ),
        )

        # 量化零点缓冲区：存储每组的量化零点
        # 形状：[输出特征, 计算得到的零点宽度]
        # 零点用于对称量化，确保量化范围的中心对齐
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (out_features, calculate_zeros_width(in_features, self.group_size)),
                dtype=torch.int32,
                device=dev,
            ),
        )

        # 缩放因子缓冲区：存储每组的量化缩放因子
        # 形状：[输出特征, 零点宽度*打包数]
        # 缩放因子将4位量化值映射回浮点数范围
        self.register_buffer(
            "scales",
            torch.zeros(
                (
                    out_features,
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )

        # 可选的偏置缓冲区
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear:nn.Linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        """
        从标准线性层创建GEMV量化线性层

        该类方法是将预训练的FP16/FP32线性层转换为4位量化层的主要接口。
        它根据提供的缩放因子和零点信息执行实际的权重量化。

        Args:
            linear (nn.Linear): 原始的FP16/FP32线性层
            w_bit (int): 量化位数（通常是4）
            group_size (int): 量化分组大小
            init_only (bool): 是否仅初始化而不进行实际量化
            scales (torch.Tensor): 缩放因子，形状为[out_features, num_groups]
            zeros (torch.Tensor): 零点值，形状为[out_features, num_groups]

        Returns:
            WQLinear_GEMV: 量化后的线性层实例

        量化流程：
            1. 创建空的量化层结构
            2. 使用缩放因子和零点计算scale_zeros
            3. 逐通道量化权重：int(round((weight + zero_point) / scale))
            4. 打包4位权重到int32
            5. 打包4位零点到int32

        使用示例：
            # 量化现有线性层
            q_linear = WQLinear_GEMV.from_linear(
                linear_layer,
                w_bit=4,
                group_size=128,
                scales=scales,
                zeros=zeros
            )
        """
        # 第一步：创建空的GEMV量化层
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )

        # 如果只是初始化（用于加载预训练量化模型），直接返回空结构
        if init_only:  # 仅准备结构，用于加载预训练权重
            return awq_linear

        # 第二步：验证量化参数
        # 实际量化需要缩放因子和零点信息
        assert scales is not None and zeros is not None, "实际量化需要提供scales和zeros参数"

        # 计算融合的缩放零点：scale_zeros = zeros * scales
        # 这个值用于将原始权重偏移到合适的量化范围
        scale_zeros = zeros * scales

        # 计算打包参数：每个int32存储的4位值数量
        pack_num = 32 // awq_linear.w_bit  # 4位情况下：32/4 = 8

        # ==================== 处理缩放因子 ====================
        # 创建扩展的缩放因子张量，确保内存对齐
        qscales = torch.zeros(
            (
                scales.shape[0],  # 输出特征维度
                calculate_zeros_width(linear.in_features, group_size) * pack_num,  # 对齐后的维度
            ),
            dtype=torch.float16,
            device=scales.device,
        )
        # 将原始缩放因子复制到扩展张量的前部
        qscales[:, : scales.shape[1]] = scales
        awq_linear.scales = qscales

        # ==================== 处理偏置 ====================
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()  # 复制并转换为半精度

        # ==================== 权重量化核心算法 ====================
        # 逐输入特征进行量化：每个输入特征对应一列权重
        intweight = []  # 存储量化后的整数权重
        for idx in range(awq_linear.in_features):
            # 对当前输入特征对应的所有输出权重进行量化
            # 量化公式：q = round((w + zero_point) / scale)
            # 其中：zero_point = scale_zeros, scale = scales
            quantized_col = torch.round(
                (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size])
                / awq_linear.scales[:, idx // group_size]
            ).to(torch.int)[:, None]
            intweight.append(quantized_col)

        # 将所有列连接成完整的量化权重矩阵
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.to(dtype=torch.int32)

        # ==================== 4位权重打包 ====================
        # 创建打包后的权重张量：压缩到int32存储
        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=intweight.device,
        )

        # 逐列进行4位打包
        for col in range(intweight.shape[1] // pack_num):
            # 定义4位量化的位顺序映射
            if awq_linear.w_bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]  # 8个4位值的顺序
            else:
                raise NotImplementedError("目前仅支持4位量化。")

            # 将8个4位值打包到一个int32中
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)  # 位移到正确位置

        awq_linear.qweight = qweight

        # ==================== 零点量化打包 ====================
        # 将零点值转换为int32类型
        zeros = zeros.to(dtype=torch.int32)

        # 创建打包后的零点张量
        qzeros = torch.zeros(
            (zeros.shape[0], calculate_zeros_width(linear.in_features, group_size)),
            dtype=torch.int32,
            device=zeros.device,
        )

        # 逐列进行4位零点打包（与权重打包类似）
        # 注意：零点数量可能与权重不完全匹配，需要边界检查
        for col in range((zeros.shape[1] + pack_num - 1) // pack_num):  # 向上取整
            # 定义4位零点的位顺序映射
            if awq_linear.w_bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("目前仅支持4位量化。")

            # 将8个4位零点值打包到一个int32中
            for i in range(pack_num):
                # 边界检查：防止越界访问
                if col * pack_num + order_map[i] >= zeros.shape[1]:
                    continue
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)

        awq_linear.qzeros = qzeros
        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        """
        GEMV量化线性层的前向传播

        该方法实现了量化线性变换的前向传播，根据输入批量大小智能选择
        最优的CUDA内核。支持任意输入形状的批处理。

        Args:
            x (torch.Tensor): 输入张量，形状为[*, in_features]
                             可以是任意维度，最后一个维度必须是in_features

        Returns:
            torch.Tensor: 输出张量，形状为[*, out_features]

        内核选择策略：
            - 大批量 (>8): 使用gemmv2_forward_cuda，优化吞吐量
            - 小批量 (≤8): 使用gemv_forward_cuda，优化延迟

        处理流程：
            1. 检查AWQ扩展模块可用性
            2. 重塑输入为2D张量
            3. 确保输入为FP16格式
            4. 根据批量大小选择CUDA内核
            5. 执行量化矩阵-向量乘法
            6. 恢复原始数据类型和形状
            7. 添加偏置（如果有）
        """
        # 检查AWQ扩展模块是否正确安装
        if awq_ext is None:
            raise ModuleNotFoundError("AWQ扩展模块未正确安装。" + msg)

        # 保存输出形状，支持任意输入维度
        out_shape = x.shape[:-1] + (self.out_features,)

        # 重塑输入为2D张量：[batch_size, in_features]
        inputs = x.reshape(-1, x.shape[-1])

        # ==================== 数据类型处理 ====================
        input_dtype = inputs.dtype
        if input_dtype != torch.float16:
            inputs = inputs.half()  # 转换为FP16以匹配CUDA内核要求

        # ==================== 智能内核选择 ====================
        if inputs.shape[0] > 8:
            # 大批量场景：使用gemmv2内核，优化吞吐量
            out = awq_ext.gemmv2_forward_cuda(
                inputs,           # 输入张量
                self.qweight,     # 量化权重
                self.scales,      # 缩放因子
                self.qzeros,      # 量化零点
                self.group_size,  # 分组大小
                self.split_k_iters, # 分块迭代次数
            )
        else:
            # 小批量场景：使用gemv内核，优化延迟
            out = awq_ext.gemv_forward_cuda(
                inputs, self.qweight, self.scales, self.qzeros, self.group_size
            )

        # ==================== 输出后处理 ====================
        # 恢复原始数据类型
        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        # 添加偏置（如果有）
        out = out + self.bias if self.bias is not None else out

        # 恢复原始形状
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        """
        返回模块的字符串表示

        该方法定义了模块在打印时显示的关键信息，
        便于调试和查看网络结构。

        Returns:
            str: 包含关键参数的格式化字符串

        显示内容：
            - in_features: 输入特征维度
            - out_features: 输出特征维度
            - bias: 是否包含偏置
            - w_bit: 量化位数
            - group_size: 分组大小
        """
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )
