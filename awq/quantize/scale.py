"""
AWQ量化器缩放和裁剪模块

该模块实现了AWQ（Activation-aware Weight Quantization）算法中的核心缩放和
权重裁剪功能。缩放是AWQ的关键技术，通过分析激活值来确定最佳的权重缩放因子，
从而最小化量化误差。

核心功能：
1. 权重裁剪（apply_clip）：限制权重范围以减少量化误差
2. 缩放应用（apply_scale）：根据激活统计信息应用最优缩放因子
3. 多种缩放策略：支持不同类型的层（归一化层、线性层、激活函数层）
4. 特殊架构支持：Gemma、Bloom等模型的特殊归一化处理

算法原理：
- 通过分析激活值分布来确定每通道的缩放因子
- 应用权重缩放来平衡量化误差
- 使用权重裁剪进一步优化量化精度

数学原理：
对于每个通道i，寻找最优缩放因子s_i，使得：
min_s_i ||f(x, W) - f(x, round(W/s_i) * s_i)||²
"""

# 导入必要的库
import torch  # PyTorch主库，张量操作和自动微分
import torch.nn as nn  # PyTorch神经网络模块
from typing import Tuple, List  # 类型注解支持
from awq.utils.utils import get_best_device  # 获取最佳计算设备（GPU/CPU）
from awq.modules.act import ScaledActivation  # 带缩放的激活函数模块
from awq.utils.module import get_op_by_name, set_op_by_name  # 按名称获取和设置模块操作
# 导入各种模型特定的归一化层
from transformers.models.bloom.modeling_bloom import BloomGelu  # Bloom模型的GELU激活
from transformers.models.llama.modeling_llama import LlamaRMSNorm  # Llama模型的RMS归一化
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm  # Gemma模型的RMS归一化
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm  # Gemma2模型的RMS归一化
from transformers.models.cohere.modeling_cohere import CohereLayerNorm  # Cohere模型的层归一化
from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation  # 各种GELU激活函数变体

# ==================== 支持的归一化层类型 ====================
# 这些归一化层可以用于缩放操作，作为前置操作层
allowed_norms = [
    nn.LayerNorm,        # 标准层归一化
    LlamaRMSNorm,       # Llama模型的RMS归一化
    GemmaRMSNorm,       # Gemma模型的RMS归一化
    Gemma2RMSNorm,      # Gemma2模型的RMS归一化
    CohereLayerNorm,    # Cohere模型的层归一化
]

# ==================== 支持的激活函数类型 ====================
# 这些激活函数可以用于缩放操作，通常需要特殊处理
allowed_act_fns = [
    nn.GELU,                # 标准GELU激活函数
    BloomGelu,             # Bloom模型的GELU变体
    NewGELUActivation,     # 新版GELU激活函数
    PytorchGELUTanh,       # PyTorch的tanh-GELU变体
    GELUActivation,        # 标准GELU激活函数
]


@torch.no_grad()
def apply_clip(module, clip_list: Tuple[str, torch.Tensor]):
    """
    应用权重裁剪到指定层

    权重裁剪是AWQ量化的重要优化步骤，通过限制权重的取值范围来减少量化误差。
    该函数根据预先计算的最优裁剪阈值对线性层的权重进行裁剪操作。

    Args:
        module: 包含待裁剪层的父模块
        clip_list: 裁剪配置列表，每个元素为(层名, 最大值)的元组
                  - 层名(str): 要裁剪的线性层名称
                  - 最大值(torch.Tensor): 每个通道的最大绝对值阈值

    Returns:
        None: 直接在原模块中修改权重数据

    算法步骤：
        1. 通过名称获取目标线性层
        2. 将层和裁剪值移动到最佳设备
        3. 重塑权重张量以匹配裁剪值的维度
        4. 应用裁剪：clamp(weight, -max_val, max_val)
        5. 恢复原始形状并移动回CPU
    """
    for name, max_val in clip_list:  # 遍历所有需要裁剪的层
        layer: nn.Linear = get_op_by_name(module, name)  # 通过名称获取线性层
        layer.to(get_best_device())  # 移动到最佳计算设备（GPU优先）
        max_val = max_val.to(layer.weight.device)  # 确保裁剪值在同一设备上

        org_shape = layer.weight.shape  # 保存原始形状用于后续恢复

        # 重塑权重张量以匹配裁剪值的维度结构
        # 通常将out_channels维度重塑为(group_size, channels_per_group)
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)

        # 应用裁剪操作：将权重限制在[-max_val, max_val]范围内
        # 这有助于减少极值权重对量化精度的影响
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)

        # 恢复权重的原始形状
        layer.weight.data = layer.weight.data.reshape(org_shape)

        layer.cpu()  # 移回CPU以节省GPU内存


def apply_scale(module, scales_list, input_feat_dict=None):
    """
    应用缩放因子到神经网络层

    这是AWQ量化的核心函数，根据前置操作层的类型选择适当的缩放策略。
    支持多种层类型的缩放：线性层、归一化层、激活函数层等。

    Args:
        module: 包含待缩放层的父模块
        scales_list: 缩放配置列表，每个元素为(prev_op_name, layer_names, scales)
                   - prev_op_name: 前置操作层名称
                   - layer_names: 需要缩放的层名称列表
                   - scales: 缩放因子张量
        input_feat_dict: 可选的输入特征字典，用于准备后续的裁剪操作

    Returns:
        None: 直接修改模块中的层权重

    缩放策略：
        1. 线性层 -> 多个线性层: scale_fc_fcs()
        2. 线性层 -> 单个线性层: scale_fc_fc()
        3. 归一化层 -> 线性层: scale_ln_fcs()
        4. 激活函数层 -> 线性层: scale_gelu_fc()

    数学原理：
        对于每个缩放组，修改前置层的权重W_prev和目标层的权重W_target：
        W_prev' = W_prev / scales
        W_target' = W_target * scales
        这样保持了等价性的同时优化了量化精度
    """
    for prev_op_name, layer_names, scales in scales_list:  # 遍历所有缩放配置
        # ==================== 获取层引用 ====================
        prev_op = get_op_by_name(module, prev_op_name)  # 获取前置操作层
        layers = [get_op_by_name(module, name) for name in layer_names]  # 获取目标层列表

        # ==================== 设备管理 ====================
        best_device = get_best_device()  # 获取最佳计算设备
        prev_op.to(best_device)  # 移动前置层到最佳设备
        for layer in layers:
            layer.to(best_device)  # 移动所有目标层到最佳设备
        scales.to(best_device)  # 移动缩放因子到最佳设备

        # ==================== 缩放策略选择 ====================
        # 策略1: 线性层 -> 多个线性层 (如: 前一层 -> [q_proj, k_proj, v_proj])
        if (
            isinstance(prev_op, nn.Linear)  # 前置层是线性层
            and type(layers) == list         # 目标层是列表
            and isinstance(layers[0], nn.Linear)  # 第一个目标层是线性层
        ):
            scale_fc_fcs(prev_op, layers, scales)  # 应用多目标线性层缩放

        # 策略2: 线性层 -> 单个线性层 (如: v_proj -> o_proj)
        elif isinstance(prev_op, nn.Linear):
            assert len(layers) == 1  # 确保只有一个目标层
            scale_fc_fc(prev_op, layers[0], scales)  # 应用单目标线性层缩放

        # 策略3: 归一化层 -> 线性层 (如: layernorm -> linear)
        elif (
            any(isinstance(prev_op, t) for t in allowed_norms)  # 前置层是支持的归一化层
            or "rmsnorm" in str(prev_op.__class__).lower()      # 或者是RMS归一化
        ):
            scale_ln_fcs(prev_op, layers, scales)  # 应用归一化层缩放

        # 策略4: 激活函数层 -> 线性层 (如: gelu -> linear)
        elif any(isinstance(prev_op, t) for t in allowed_act_fns):  # 前置层是支持的激活函数
            new_module = ScaledActivation(prev_op, scales)  # 创建带缩放的激活函数
            set_op_by_name(module, prev_op_name, new_module)  # 替换原激活函数
            scale_gelu_fc(prev_op, layers[0], scales)  # 应用激活函数缩放

        else:
            # 不支持的前置层类型
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # ==================== 输入特征调整 ====================
        # 如果提供了输入特征字典，对相关特征进行相应缩放
        # 这是为了准备后续的权重裁剪操作
        if input_feat_dict is not None:
            for layer_name in layer_names:  # 遍历所有目标层名
                if layer_name in input_feat_dict:  # 跳过未被量化的模块
                    inp = input_feat_dict[layer_name]  # 获取输入特征
                    # 对输入特征除以缩放因子，保持数值等价性
                    inp.div_(scales.view(1, -1).to(inp.device))

        # ==================== 内存管理 ====================
        # 将所有张量移回CPU以释放GPU内存
        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()


@torch.no_grad()
def scale_ln_fcs(ln, fcs: List[nn.Linear], scales: torch.Tensor):
    """
    应用归一化层到线性层的缩放

    该函数实现归一化层（如LayerNorm、RMSNorm）与其后续线性层的缩放操作。
    这是AWQ算法中处理归一化-线性层对的核心函数。

    Args:
        ln: 归一化层（LayerNorm、RMSNorm等）
        fcs: 后续线性层列表（可以是单个层或多个层）
        scales: 缩放因子张量，形状为(通道数,)

    Returns:
        None: 直接修改层的权重参数

    特殊处理：
        - Gemma系列模型：RMSNorm使用(1 + weight)的形式
        - 带偏置的归一化层：同时缩放权重和偏置
        - 数值稳定性检查：确保缩放后无NaN值

    数学原理：
        对于归一化层LN和线性层FC：
        LN.weight' = LN.weight / scales
        LN.bias' = LN.bias / scales (如果存在)
        FC.weight' = FC.weight * scales
        这样保持了整体输出的等价性
    """
    if not isinstance(fcs, list):  # 确保fcs是列表形式
        fcs = [fcs]

    scales = scales.to(ln.weight.device)  # 确保缩放因子在同一设备上

    # ==================== 特殊处理Gemma系列模型 ====================
    # GemmaRMSNorm与Llama的实现不同，它将(1 + weight)乘以输出，
    # 而不是直接使用weight。因此需要特殊的处理方式。
    if isinstance(ln, GemmaRMSNorm) or isinstance(ln, Gemma2RMSNorm):
        # 对于Gemma：weight_new = (1 + weight_old) / scales - 1
        ln.weight += 1          # 加1得到实际乘数
        ln.weight.div_(scales)  # 除以缩放因子
        ln.weight -= 1          # 减1恢复为权重参数形式
    else:
        # 对于其他模型：直接除以缩放因子
        ln.weight.div_(scales)

    # ==================== 处理带偏置的归一化层 ====================
    # 如果归一化层有偏置项，也需要相应缩放
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    # ==================== 缩放后续线性层 ====================
    # 对每个后续线性层的权重乘以相应的缩放因子
    for fc in fcs:
        # scales.view(1, -1)确保广播到正确的维度
        fc.weight.mul_(scales.view(1, -1))

    # ==================== 数值稳定性检查 ====================
    # 确保所有参数都不包含NaN值，防止数值异常
    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0, f"NaN detected in layer norm parameters!"
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0, f"NaN detected in linear layer parameters!"


@torch.no_grad()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    """
    应用线性层到线性层的缩放

    该函数实现两个连续线性层之间的缩放操作，常用于处理
    attention输出投影或MLP中的连续线性变换。

    Args:
        fc1: 第一个线性层（前置层）
        fc2: 第二个线性层（目标层）
        scales: 缩放因子张量

    Returns:
        None: 直接修改两个线性层的权重参数

    缩放策略：
        - fc1的输出通道（最后几行）除以缩放因子
        - fc2的输入通道（所有列）乘以缩放因子
        - 如果fc1有偏置，也相应除以缩放因子

    应用场景：
        - v_proj -> o_proj (注意力输出路径)
        - up_proj -> down_proj (MLP路径)

    数学原理：
        fc1.weight[-n:]' = fc1.weight[-n:] / scales
        fc2.weight' = fc2.weight * scales
        这样保持了 fc2(fc1(x)) 的输出等价性
    """
    # 参数类型检查，确保输入是线性层
    assert isinstance(fc1, nn.Linear), "fc1 must be a Linear layer"
    assert isinstance(fc2, nn.Linear), "fc2 must be a Linear layer"

    scales = scales.to(fc1.weight.device)  # 确保缩放因子在同一设备上

    # ==================== 缩放第一个线性层 ====================
    # 只缩放fc1的输出通道（权重的最后几行）
    # -scales.size(0)表示从倒数第n行开始
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))

    # 如果fc1有偏置，也相应进行缩放
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    # ==================== 缩放第二个线性层 ====================
    # 缩放fc2的输入通道（权重的所有列）
    fc2.weight.mul_(scales.view(1, -1))

    # ==================== 数值稳定性检查 ====================
    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0, f"NaN detected in fc1 parameters!"
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0, f"NaN detected in fc2 parameters!"


@torch.no_grad()
def scale_fc_fcs(fc1: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    """
    应用线性层到多个线性层的缩放

    该函数实现一个线性层到多个并行线性层的缩放操作，
    常用于处理QKV投影或MLP门控-上投影的并行结构。

    Args:
        fc1: 第一个线性层（前置层）
        fcs: 后续线性层列表（多个目标层）
        scales: 缩放因子张量

    Returns:
        None: 直接修改所有线性层的权重参数

    缩放策略：
        - fc1的输出通道（最后几行）除以缩放因子
        - 所有fc的输入通道（所有列）乘以缩放因子
        - 如果fc1有偏置，也相应除以缩放因子

    应用场景：
        - layernorm -> [q_proj, k_proj, v_proj] (QKV投影)
        - layernorm -> [gate_proj, up_proj] (MLP门控)

    数学原理：
        fc1.weight[-n:]' = fc1.weight[-n:] / scales
        for each fc in fcs:
            fc.weight' = fc.weight * scales
        这样保持了并行输出的等价性
    """
    if not isinstance(fcs, list):  # 确保fcs是列表形式
        fcs = [fcs]

    scales = scales.to(fc1.weight.device)  # 确保缩放因子在同一设备上

    # ==================== 缩放第一个线性层 ====================
    # 缩放fc1的输出通道（权重的最后几行）
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))

    # 如果fc1有偏置，也相应进行缩放
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    # ==================== 缩放所有后续线性层 ====================
    # 对每个后续线性层进行相同的缩放操作
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    # ==================== 数值稳定性检查 ====================
    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0, f"NaN detected in fc1 parameters!"
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0, f"NaN detected in fc parameters!"


@torch.no_grad()
def scale_gelu_fc(gelu: allowed_act_fns, fc: nn.Linear, scales: torch.Tensor):
    """
    应用激活函数到线性层的缩放

    该函数实现激活函数层到其后续线性层的缩放操作。
    与其他缩放函数不同，这里只缩放线性层，因为激活函数
    本身会被ScaledActivation替换。

    Args:
        gelu: 激活函数层（GELU系列）
        fc: 后续线性层
        scales: 缩放因子张量

    Returns:
        None: 直接修改线性层的权重参数

    特殊说明：
        - 激活函数本身不会直接修改，而是被ScaledActivation替换
        - 只有后续的线性层需要乘以缩放因子
        - 这种不对称处理是为了保持激活函数的数值稳定性

    应用场景：
        - gelu -> linear (MLP中的激活函数后接线性变换)

    数学原理：
        由于激活函数会被ScaledActivation(gelu, scales)替换，
        只需要：
        fc.weight' = fc.weight * scales
        这样保持了 fc(gelu(x)) -> fc'(scaled_gelu(x)) 的等价性
    """
    # 参数类型检查
    assert any(isinstance(gelu, t) for t in allowed_act_fns), "gelu must be one of allowed activation functions"
    assert isinstance(fc, nn.Linear), "fc must be a Linear layer"

    # ==================== 缩放线性层 ====================
    # 只缩放线性层的权重，激活函数会被ScaledActivation处理
    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    # ==================== 数值稳定性检查 ====================
    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0, f"NaN detected in fc parameters!"
