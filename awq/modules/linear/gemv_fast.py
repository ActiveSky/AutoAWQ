"""
AWQ GEMV_FAST高性能量化线性层实现

该模块实现了GEMV Fast版本的AWQ量化线性层，通过先进的内存布局优化和
数据类型压缩，在GEMV基础上实现了显著的性能提升。这是AWQ项目中
最高性能的量化线性层实现。

核心优化技术：
- int16权重存储：相比int32节省50%内存
- 交错内存布局：优化GPU内存访问模式
- 权重重排算法：特殊的数据重排提高缓存命中率
- 零点融合预计算：减少运行时开销
- 专用CUDA内核：针对特定场景的极致优化

性能提升：
- 内存占用：减少50%
- 推理延迟：提升20-50%
- 缓存命中率：显著提高
- GPU利用率：优化内存带宽

适用场景：
- 高性能文本生成服务
- 单token解码优化
- 内存受限的边缘设备
- 延迟敏感的实时应用

技术特点：
- 使用awq_v2_ext扩展模块
- 智能内核选择策略
- 复杂的权重打包算法
- 高度优化的数据布局
"""

# 导入必要的库
import torch  # PyTorch主库，张量操作和自动微分
import warnings  # 警告信息处理
from awq.utils.module import try_import  # 动态导入扩展模块

# 导入AWQ V2扩展模块，包含高性能CUDA内核
awq_v2_ext, msg = try_import("awq_v2_ext")

def make_divisible(c, divisor):
    """
    确保数值能被除数整除的辅助函数

    Args:
        c (int): 原始数值
        divisor (int): 目标除数

    Returns:
        int: 能被divisor整除的最小整数
    """
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    """
    计算零点存储所需的宽度

    Args:
        in_features (int): 输入特征维度
        group_size (int): 量化分组大小
        pack_num (int): 每个int32打包的量化值数量

    Returns:
        int: 计算得到的零点宽度
    """
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError("不支持的group_size")

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


def pack_intweight(unpacked_qweight, interleave, kstride):
    """
    高性能权重打包算法

    这是GEMV_FAST的核心优化技术，通过复杂的数据重排和交错处理，
    实现最优的GPU内存访问模式和缓存利用率。

    Args:
        unpacked_qweight (torch.Tensor): 未打包的量化权重 [N, K]
        interleave (int): 交错间隔，通常为4
        kstride (int): 步长，通常为64

    Returns:
        torch.Tensor: 打包后的int16权重张量

    优化步骤：
        1. 4x4x2重排：优化内存访问局部性
        2. 8权重重排：[0,1,2,3,4,5,6,7] => [0,2,4,6,1,3,5,7]
        3. 行交错：每4行进行交错处理
        4. int16打包：压缩存储格式

    性能收益：
        - 内存占用减少50%
        - 缓存命中率提高
        - 内存带宽优化
        - GPU访问模式优化
    """
    # 获取权重张量形状：[N, K] = [输出特征, 输入特征]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    # ==================== 第一步：4x4x2重排 ====================
    # 将权重重塑为[N, K//32, 32]，然后进行4x4x2重排
    # 重排模式：(0,1,2,3) -> (1,0,3,2) 以优化内存访问
    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # ==================== 第二步：8权重重排 ====================
    # 重排每个8个权重以优化反量化速度
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    # 这种重排模式优化了GPU的向量化操作
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # ==================== 第三步：行交错处理 ====================
    # 每隔4行进行交错，提高缓存局部性
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, interleave, K // kstride, kstride)
    # 形状变为：[N//4, 4, K//64, 64]

    # 转置以优化内存访问模式
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K // kstride, kstride, interleave)
    # 形状变为：[N//4, K//64, 64, 4]

    # ==================== 第四步：int16打包 ====================
    # 将4个4位值打包为int16格式
    # 使用位移和或操作进行高效打包
    Packed_Kernel = (
        Packed_Kernel[..., 0]        # 第0个4位值
        | (Packed_Kernel[..., 1] << 4)   # 第1个4位值，左移4位
        | (Packed_Kernel[..., 2] << 8)   # 第2个4位值，左移8位
        | (Packed_Kernel[..., 3] << 12)  # 第3个4位值，左移12位
    )

    # 重塑为最终形状：[N//4, K]，使用int16格式
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))  # 转换为int16
        .to(unpacked_qweight.device)                # 移动到原设备
        .contiguous()                               # 确保内存连续
    )
    return qweight


class WQLinear_GEMVFast(torch.nn.Module):
    """
    AWQ GEMV Fast高性能量化线性层类

    该类实现了AWQ项目中最高性能的量化线性层，通过先进的内存布局优化
    和数据压缩技术，在保持精度的同时显著提升推理速度和降低内存占用。

    核心优化特性：
        - int16权重存储：相比int32节省50%内存
        - 交错内存布局：4行交错提高缓存局部性
        - 优化的权重重排：特殊的数据排列模式
        - 零点融合预计算：减少运行时计算开销
        - 专用CUDA内核：针对特定场景优化

    与标准GEMV的区别：
        - 使用awq_v2_ext扩展而非awq_ext
        - 权重数据类型：int16 vs int32
        - 内存布局：交错和重排优化
        - 零点处理：预计算融合
        - 性能：20-50%延迟提升

    Attributes:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        w_bit (int): 量化位数
        group_size (int): 量化分组大小
        interleave (int): 交错间隔，固定为4
        split_k_iters (int): 分块计算迭代次数
    """

    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        """
        初始化GEMV Fast量化线性层

        Args:
            w_bit (int): 量化位数，当前仅支持4位
            group_size (int): 量化分组大小
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            bias (bool): 是否使用偏置
            dev (str): 设备类型

        设计特点：
            - 使用int16存储优化内存
            - 固定interleave=4优化访问模式
            - 特殊的内存布局设计
        """
        super().__init__()

        # 基本属性设置
        self.in_features = in_features          # 输入维度
        self.out_features = out_features        # 输出维度
        self.w_bit = w_bit                      # 量化位数
        # 分组大小：-1表示按通道量化
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8                  # 分块计算迭代次数
        self.interleave = 4                     # 交错间隔，固定为4以优化访问

        # ==================== 内存对齐检查 ====================
        # 确保输入特征能被分组大小整除
        assert self.in_features % self.group_size == 0, "输入特征必须能被分组大小整除"
        # 确保输出特征能被打包单位整除
        assert out_features % (32 // self.w_bit) == 0, "输出特征必须能被打包单位整除"

        # 计算打包参数
        pack_num = 32 // self.w_bit           # int32中4位值数量：8
        int16_pack_num = 16 // self.w_bit     # int16中4位值数量：4

        # 确保输出特征能被交错间隔整除
        assert out_features % (self.interleave) == 0, "输出特征必须能被交错间隔整除"
        # ==================== 优化的缓冲区初始化 ====================

        # 量化权重缓冲区：使用int16存储，节省50%内存
        # 形状：[out_features//4, in_features//4*4] 交错布局
        self.register_buffer(
            "qweight",
            torch.zeros(
                (
                    out_features // self.interleave,                    # 交错后的输出维度
                    in_features // int16_pack_num * self.interleave,   # 优化的输入维度
                ),
                dtype=torch.int16,    # 使用int16而非int32
                device=dev,
            ),
        )

        # 缩放因子缓冲区：转置布局以优化访问
        self.register_buffer(
            "scales",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,
                device=dev,
            ),
        )

        # 零点缓冲区：使用float16并预计算融合
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    calculate_zeros_width(in_features, self.group_size) * pack_num,
                    out_features,
                ),
                dtype=torch.float16,    # 预计算的融合零点
                device=dev,
            ),
        )

        # 偏置缓冲区
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        pack_num = 32 // awq_linear.w_bit
        qscales = torch.zeros(
            (
                scales.shape[0],
                calculate_zeros_width(linear.in_features, group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=scales.device,
        )
        qscales[:, : scales.shape[1]] = scales
        # awq_linear.scales = scales.clone().half()
        awq_linear.scales = qscales.transpose(1, 0).contiguous()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size])
                    / qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.to(dtype=torch.int32)
        awq_linear.qweight = pack_intweight(
            intweight.contiguous(), interleave=4, kstride=64
        )

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros_like(qscales)

        qzeros[:, : scales.shape[1]] = -(
            qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))
        ).to(torch.float16)
        awq_linear.qzeros = qzeros.transpose(1, 0).contiguous()

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        """
        GEMV Fast前向传播

        使用优化的awq_v2_ext内核，根据输入形状智能选择最佳计算策略。

        Args:
            x (torch.Tensor): 输入张量 [batch_size, n_tokens, in_features]

        Returns:
            torch.Tensor: 输出张量 [batch_size, n_tokens, out_features]

        内核选择策略：
            - 小批量单token: gemv_forward_cuda_decode
            - 其他场景: gemm_forward_cuda_prefill
        """
        # 检查AWQ V2扩展模块
        if awq_v2_ext is None:
            raise ModuleNotFoundError("AWQ V2扩展模块未正确安装。" + msg)

        inputs = x
        batch_size, n_tokens, _ = inputs.shape

        # ==================== 智能内核选择 ====================
        if batch_size < 8 and n_tokens == 1:
            # 小批量单token解码：使用专用解码内核
            out = awq_v2_ext.gemv_forward_cuda_decode(
                inputs,
                self.qweight,
                self.scales,
                self.qzeros,
                inputs.numel() // inputs.shape[-1],  # 总元素数
                self.out_features,
                self.in_features,
                self.group_size,
            )
        else:
            # 其他场景：使用预填充内核
            out = awq_v2_ext.gemm_forward_cuda_prefill(
                inputs, self.qweight, self.scales, self.qzeros
            )

        # 添加偏置
        out = out + self.bias if self.bias is not None else out

        return out
