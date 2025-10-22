# 导入第三方库
import transformers  # HuggingFace Transformers库
import torch  # PyTorch主库
import inspect  # 检查代码对象的工具
import logging  # 日志记录
import functools  # 函数工具模块
import torch.nn as nn  # PyTorch神经网络模块
from tqdm import tqdm  # 进度条显示
from typing import Dict, List, Optional  # 类型注解
from collections import defaultdict  # 默认字典
 

# 导入AWQ内部模块
from awq.utils.calib_data import get_calib_dataset  # 获取校准数据集
from awq.quantize.scale import apply_scale, apply_clip  # 应用缩放和裁剪
from awq.utils.utils import clear_memory, get_best_device  # 内存清理和设备选择
from awq.modules.linear import (  # 导入各种量化线性层
    WQLinear_GEMM,    # GEMM量化线性层
    WQLinear_GEMV,    # GEMV量化线性层
    WQLinear_Marlin,  # Marlin量化线性层
    WQLinear_GEMVFast,  # 快速GEMV量化线性层
)
from awq.utils.module import (  # 导入模块工具函数
    append_str_prefix,  # 添加字符串前缀
    get_op_name,       # 获取操作名称
    get_named_linears, # 获取命名线性层
    set_op_by_name,    # 按名称设置操作
    exclude_layers_to_not_quantize,  # 排除不需要量化的层
)

from awq.models.base import BaseAWQForCausalLM


class AwqQuantizer:
    """
    AWQ（Activation-aware Weight Quantization）量化器类

    该类实现了AWQ量化算法，通过激活感知的权重量化技术，
    在保持模型精度的同时显著减少模型大小和推理开销。
    """

    def __init__(
        self,
        awq_model:BaseAWQForCausalLM,  # AWQ模型实例
        model,      # 原始模型
        tokenizer,  # 分词器
        w_bit,      # 量化位数
        group_size, # 量化组大小
        zero_point, # 是否使用零点量化
        version,    # 量化版本
        calib_data, # 校准数据
        split,      # 数据集分割
        text_column, # 文本列名
        duo_scaling, # 是否使用双重缩放
        modules_to_not_convert=None,  # 不转换的模块列表
        export_compatible=False,       # 是否导出兼容
        apply_clip=True,               # 是否应用裁剪
        n_parallel_calib_samples=None, # 并行校准样本数
        max_calib_samples=128,         # 最大校准样本数
        max_calib_seq_len=512,         # 最大校准序列长度
        max_chunk_memory=1024 * 1024 * 1024,  # 最大块内存（1GB）
    ) -> None:
        """
        初始化AWQ量化器

        Args:
            awq_model: AWQ模型包装器
            model: 要量化的PyTorch模型
            tokenizer: 分词器实例
            w_bit: 量化位数（如4位量化）
            group_size: 量化组大小
            zero_point: 是否使用零点量化
            version: 量化算法版本
            calib_data: 校准数据集
            split: 数据集分割方式
            text_column: 文本数据的列名
            duo_scaling: 是否使用权重和激活双重缩放
            modules_to_not_convert: 不进行量化的模块列表
            export_compatible: 是否生成导出兼容的权重
            apply_clip: 是否应用权重裁剪优化
            n_parallel_calib_samples: 并行处理的校准样本数
            max_calib_samples: 最大校准样本数量
            max_calib_seq_len: 最大校准序列长度
            max_chunk_memory: 最大内存块大小
        """
        # 初始化基本属性
        self.awq_model = awq_model                    # AWQ模型实例
        self.model = model                            # 原始模型
        self.tokenizer = tokenizer                    # 分词器
        self.w_bit = w_bit                            # 量化位数
        self.group_size = group_size                  # 量化组大小
        self.zero_point = zero_point                  # 零点量化标志
        self.version = version                        # 量化版本
        self.calib_data = calib_data                  # 校准数据
        self.split = split                            # 数据分割
        self.text_column = text_column                # 文本列名
        self.duo_scaling = duo_scaling                # 双重缩放标志
        self.export_compatible = export_compatible    # 导出兼容标志
        self.apply_clip = apply_clip                  # 应用裁剪标志
        self.n_parallel_calib_samples = n_parallel_calib_samples  # 并行校准样本数
        self.max_calib_samples = max_calib_samples    # 最大校准样本数
        self.max_calib_seq_len = max_calib_seq_len    # 最大校准序列长度
        self.max_chunk_memory = max_chunk_memory      # 最大块内存

        # 初始化不转换的模块列表（默认为空列表）
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )

        # 初始化量化所需的数据：模块列表、模块参数、输入数据
        self.modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        """
        对张量进行伪量化操作

        Args:
            w: 要量化的权重张量

        Returns:
            tuple: (量化后的权重, 缩放因子, 零点)
        """
        org_w_shape = w.shape  # 保存原始形状
        # 如果使用分组量化，检查并重塑张量
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.group_size})!"
            w = w.reshape(-1, self.group_size)  # 重塑为分组形式
        assert w.dim() == 2  # 确保是2D张量
        assert torch.isnan(w).sum() == 0  # 确保没有NaN值

        # 零点量化模式
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)  # 每组的最大值
            min_val = w.amin(dim=1, keepdim=True)  # 每组的最小值
            max_int = 2**self.w_bit - 1  # 量化范围上限
            min_int = 0  # 量化范围下限
            scales = (max_val - min_val).clamp(min=1e-5) / max_int  # 计算缩放因子
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)  # 计算零点
            # 应用量化：量化 -> 反量化
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)  # 重塑零点形状
        else:
            # 对称量化模式（无零点）
            max_val = w.abs().amax(dim=1, keepdim=True)  # 最大绝对值
            max_val = max_val.clamp(min=1e-5)  # 防止除零
            max_int = 2 ** (self.w_bit - 1) - 1  # 正数最大值
            min_int = -(2 ** (self.w_bit - 1))  # 负数最小值
            scales = max_val / max_int  # 计算缩放因子
            zeros = None  # 无零点
            # 应用量化：直接舍入并裁剪
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        # 检查结果的有效性
        assert torch.isnan(scales).sum() == 0  # 确保缩放因子无NaN
        assert torch.isnan(w).sum() == 0  # 确保量化权重无NaN

        # 恢复形状并返回结果
        scales = scales.view(org_w_shape[0], -1)  # 重塑缩放因子
        w = w.reshape(org_w_shape)  # 重塑权重张量

        return w, scales, zeros  # 返回量化权重、缩放因子和零点

    def pseudo_dequantize_tensor( ##没有被使用
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        """
        对线性层进行伪反量化操作

        Args:
            w: 线性层
            scales: 缩放因子
            zeros: 零点（可选）

        Returns:
            torch.Tensor: 反量化后的权重张量
        """
        # 计算重复次数（用于将分组缩放因子扩展到完整权重）
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)  # 扩展缩放因子

        # 执行反量化
        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)  # 扩展零点
            w = (w.weight.data - zeros) * scales  # 零点反量化：先减去零点再乘以缩放因子
        else:
            w = w.weight.data * scales  # 对称反量化：直接乘以缩放因子

        return w  # 返回反量化后的权重

    def quantize(self):# 核心函数
        """
        执行AWQ量化的主要方法

        该方法遍历模型的所有层，对每层执行以下步骤：
        1. 获取输入特征
        2. 计算和应用最佳缩放因子
        3. 计算和应用最佳裁剪阈值
        4. 应用量化操作
        """
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # [步骤0]: 将模块和输入移动到正确的设备
            common_device = next(self.modules[i].parameters()).device  # 获取当前模块设备
            if common_device is None or str(common_device) == "cpu":
                # 如果在CPU上，尝试移动到GPU
                if torch.cuda.is_available():
                    best_device = "cuda:" + str(i % torch.cuda.device_count())  # 轮询分配GPU
                else:
                    best_device = get_best_device()  # 获取最佳可用设备

                self.modules[i] = self.modules[i].to(best_device)  # 移动模块到目标设备
                common_device = next(self.modules[i].parameters()).device  # 更新设备信息

            # 将位置ID移动到目标设备
            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            # 将注意力掩码移动到目标设备
            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            # 将输入数据移动到目标设备
            self.inps = self.inps.to(common_device)

            # 每次移动到新模块时都需要移动旋转位置编码
            # Transformers 4.45.0 将旋转编码移到模型定义中，参考PR：
            # https://github.com/huggingface/transformers/pull/32617
            self.awq_model.move_embed(self.model, common_device)

            # Transformers >= 4.48.0 要求在前向传播之前计算位置嵌入
            if (
                transformers.__version__ >= "4.48.0"
                and self.module_kwargs.get("position_embeddings") is None
            ):
                self.module_kwargs["position_embeddings"] = self.model.model.rotary_emb(
                    self.inps, self.module_kwargs["position_ids"]
                )

            # 对于Transformers >= 4.48.0，如果没有注意力掩码则设置为None
            if (transformers.__version__ >= "4.48.0"
                and self.module_kwargs.get('attention_mask') is None):
                self.module_kwargs['attention_mask'] = None

            # 将模块参数中的元组项也移动到目标设备
            for k, v in self.module_kwargs.items():
                # 检查是否为包含位置嵌入的元组
                if isinstance(v, tuple):
                    self.module_kwargs[k] = tuple(
                        item.to(common_device) if isinstance(item, (torch.Tensor, nn.Module))
                        else item for item in v
                    )

            # [步骤1]: 获取层，提取线性模块，收集输入特征
            named_linears = get_named_linears(self.modules[i])  # 获取当前模块的所有命名线性层

            # 过滤掉不需要量化的线性层
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            # 获取输入特征
            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()  # 清理内存以释放空间

            # [步骤2]: 计算并应用缩放因子列表
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )  # 获取缩放配置
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]  # 为每个层搜索最佳缩放因子
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)  # 应用缩放因子
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )  # 添加模块名称前缀

            # [步骤3]: 计算并应用裁剪列表
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )  # 搜索最佳裁剪阈值
                apply_clip(self.modules[i], clip_list)  # 应用裁剪
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )  # 添加模块名称前缀

            # [步骤4]: 量化权重
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)  # 应用量化

            clear_memory()  # 清理内存

    def pack(self):
        """
        打包量化模型权重，使其兼容特定的推理内核

        该方法用于在export_compatible模式下完成实际的权重量化
        """
        for i in tqdm(range(len(self.modules)), desc="Packing"):
            named_linears = get_named_linears(self.modules[i])  # 获取命名线性层
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert  # 过滤不需要量化的层
            )
            self._apply_quant(self.modules[i], named_linears)  # 应用量化
            clear_memory()  # 清理内存

    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        """
        对指定模块应用量化操作

        Args:
            module: 要量化的模块
            named_linears: 命名线性层字典
        """
        for name, linear_layer in named_linears.items():
            # 注意：如果线性层使用.cpu().float()会导致困惑度轻微下降
            linear_layer = linear_layer.to(get_best_device()).half()  # 移动到最佳设备并转为半精度

            # 对权重进行伪量化
            linear_layer.weight.data, scales, zeros  = self.pseudo_quantize_tensor(
                linear_layer.weight.data
            )

            # 根据量化版本选择相应的量化线性层
            if self.version == "gemm":
                scales = scales.t().contiguous()  # 转置缩放因子
                if zeros is not None:
                    zeros = zeros.t().contiguous()  # 转置零点
                q_linear_module = WQLinear_GEMM

            elif self.version == "gemv":
                q_linear_module = WQLinear_GEMV

            elif self.version == "marlin":
                q_linear_module = WQLinear_Marlin

            elif self.version == "gemv_fast":
                q_linear_module = WQLinear_GEMVFast

            else:
                raise ValueError(f"Unknown version {self.version}")

            # 创建量化线性层
            q_linear = q_linear_module.from_linear(
                linear=linear_layer,           # 原始线性层
                w_bit=self.w_bit,              # 量化位数
                group_size=self.group_size,    # 组大小
                init_only=False,              # 不是仅初始化
                scales=scales,                 # 缩放因子
                zeros=zeros,                   # 零点
            )

            # 替换原始线性层
            linear_layer.cpu()  # 原始层移回CPU
            q_linear.to(next(module.parameters()).device)  # 量化层移到模块设备
            set_op_by_name(module, name, q_linear)  # 按名称设置操作
            clear_memory()  # 清理内存

    @torch.no_grad()  # 关闭梯度计算以节省内存
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        """
        模块前向传播方法，支持内存高效的批处理

        Args:
            x: 输入张量
            module: 要执行前向传播的模块
            module_kwargs: 模块参数字典

        Returns:
            torch.Tensor: 模块输出结果
        """
        if self.n_parallel_calib_samples is None:
            # 如果没有设置并行校准样本数，一次性处理所有样本
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]  # 如果是元组，取第一个元素；比如(output, attention_weights, guard_values)
        else:
            # 内存高效地处理所有校准样本，但每次只处理n_parallel_calib_samples个样本
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)  # 分割输入
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)  # 处理部分输入

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]  # 如果是元组，取第一个元素

                module_output.append(partial_output.cpu())  # 移到CPU并添加到输出列表

            module_output = torch.cat(module_output, dim=0)  # 拼接所有输出

        return module_output  # 返回模块输出

    @torch.no_grad()  # 关闭梯度计算以节省内存
    def _search_best_scale(
        self,
        module,                  # 模块
        prev_op,                 # 前置操作
        layers: List[nn.Linear], # 线性层列表
        inp: torch.Tensor,       # 输入张量
        module2inspect=None,     # 要检查的模块
        kwargs={},               # 额外参数
    ):
        """
        搜索最佳缩放因子

        该方法通过计算权重和激活的统计信息来找到最优的缩放因子，
        以最小化量化误差。

        Returns:
            tuple: (前置操作名称, 层名称元组, 最佳缩放因子)
        """
        # 如果没有指定要检查的模块，使用第一个层
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        # 移除不支持的参数
        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # 将输入移动到正确的设备
        inp = inp.to(next(module2inspect.parameters()).device)

        # [步骤1]: 计算标准化权重的每通道均值
        # 将所有层权重连接在一起
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape  # 保存原始形状
        # 将权重重塑为按量化组组织的形状
        weight = weight.view(-1, self.group_size)
        # 计算每个量化组内权重的相对幅度，
        # 并单独重新缩放每个组，使每个组的权重在0-1范围内
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # 将重新缩放的权重矩阵调整回原始维度
        w_scale = w_scale.view(org_shape)
        # 获取每个输出通道的平均重新缩放幅度
        w_mean = w_scale.mean(0)
        clear_memory(weight)  # 清理内存

        # [步骤2]: 使用分块计算输入激活的每通道均值
        # 将输入移到CPU以避免内存泄漏
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])  # 展平并取绝对值
        num_elements = inp_flat.size(0)  # 元素数量
        num_channels = inp_flat.size(1)   # 通道数量
        element_size_bytes = inp_flat.element_size() * 2  # 乘以2用于FP32

        # 根据max_chunk_memory动态计算块大小
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # 使用float32进行求和计算
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)

        # 分块处理输入数据
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)  # 计算均值
        clear_memory(x_sum)  # 清理内存

        # [步骤3]: 计算模块输出
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)  # 清理参数
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            # 裁剪到数据类型的有效范围
            fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)

        # [步骤4]: 计算损失并找到最佳缩放因子
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        # 返回操作名称和最佳缩放因子
        return (
            get_op_name(module, prev_op),  # 前置操作名称
            tuple([get_op_name(module, m) for m in layers]),  # 层名称元组
            best_scales,  # 最佳缩放因子
        )

    def _compute_best_scale(
        self,
        x: torch.Tensor,                   # 输入张量
        w_mean: torch.Tensor,              # 权重均值
        x_mean: torch.Tensor,              # 输入均值
        module2inspect: torch.nn.Module,   # 要检查的模块
        linears2scale: List[nn.Linear],    # 要缩放的线性层列表
        fp16_output: torch.Tensor,         # FP16输出
        kwargs: Dict={},                   # 额外参数
    ):
        """
        计算损失并选择最佳缩放因子

        损失函数公式：
        L(s) = || Q(W * s) (s^-1 * X) - W * X ||

        其中：
        Q: 权重量化函数 | pseudo_quantize_tensor(W * s)
        X: 校准数据集的输入 | X
        W: FP16格式的原始权重 | layer
        s: 每通道缩放因子 | s^-1 * X

        Args:
            x: 输入张量
            w_mean: 权重均值
            x_mean: 输入均值
            module2inspect: 要检查的模块
            linears2scale: 要缩放的线性层列表
            fp16_output: FP16输出
            kwargs: 额外参数

        Returns:
            torch.Tensor: 最佳缩放因子
        """
        n_grid = 20  # 网格搜索的点数
        history = []  # 历史损失记录
        best_ratio = -1  # 最佳比例
        best_scales = None  # 最佳缩放因子
        best_error = float("inf")  # 最佳误差

        # 保存原始状态字典
        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device  # 获取设备
        x_mean = x_mean.view(-1).to(device)  # 重塑并移动到设备
        w_mean = w_mean.view(-1).to(device)  # 重塑并移动到设备

        # 网格搜索最佳缩放比例
        for ratio in range(n_grid):
            # 创建新的缩放因子
            ratio = ratio / n_grid

            # 注意：根据论文，s^-1 * x 在这里融合
            if self.duo_scaling:
                # 双重缩放：同时考虑权重和激活
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                # 仅激活缩放
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)

            # 归一化缩放因子
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)  # 重塑为视图并移动到设备

            # 避免溢出的缩放值
            scales[torch.isinf(scales)] = 1  # 无穷值设为1
            scales[torch.isnan(scales)] = 1  # NaN值设为1

            # Q(W * s) - 对缩放后的权重进行量化
            for fc in linears2scale:
                fc.weight.mul_(scales_view)  # 应用缩放
                # 量化并反向缩放
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            # W * X - 计算量化后的模块输出
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            # 裁剪到数据类型范围
            int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)

            # 计算均方误差（L2范数）
            loss = self._compute_loss(fp16_output, int_w_output, device)

            # 记录历史损失
            history.append(loss)
            # 如果当前损失更小，更新最佳结果
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()

            # 恢复原始状态
            module2inspect.load_state_dict(org_sd)

        # 检查是否找到了有效的缩放因子
        if best_ratio == -1:
            logging.debug(history)
            raise Exception("未找到有效的缩放因子")

        # 确保最佳缩放因子没有NaN值
        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()  # 返回最佳缩放因子（分离梯度并移到CPU）

    @torch.no_grad()  # 关闭梯度计算以节省内存
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,   # FP16格式的原始输出
        int_w_output: torch.Tensor,   # 量化后的输出
        device: torch.device,        # 计算设备
    ):
        """
        计算量化损失

        该方法计算原始FP16输出与量化输出之间的均方误差（MSE），
        使用分块计算以避免内存溢出。

        Returns:
            float: 归一化后的损失值
        """
        loss = 0.0  # 初始化损失
        fp16_output_flat = fp16_output.view(-1)  # 展平FP16输出
        int_w_output_flat = int_w_output.view(-1)  # 展平量化输出
        num_elements = fp16_output_flat.size(0)    # 元素总数
        element_size_bytes = fp16_output.element_size()  # 元素大小（字节）

        # 根据max_chunk_memory动态计算块大小
        # 将max_chunk_memory除以元素大小的两倍
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)  # 确保不超过总元素数

        # 将计算分割成块
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)  # FP16输出块
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)  # 量化输出块

        # 计算每个块的损失
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            # 计算MSE：差值的平方和
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss  # 累加损失

        # 通过总元素数归一化损失
        loss /= num_elements

        return loss  # 返回归一化损失

    @torch.no_grad()  # 关闭梯度计算以节省内存
    def _search_best_clip(self, layer, named_linears, input_feat):
        """
        搜索最佳裁剪阈值

        该方法为每个线性层找到最佳的权重裁剪阈值，
        以最小化量化误差。某些层（如查询、键）会被跳过。

        Args:
            layer: 当前层
            named_linears: 命名线性层字典
            input_feat: 输入特征字典

        Returns:
            list: 裁剪列表，每个元素为(层名, 最大值)的元组
        """
        clip_list = []  # 初始化裁剪列表
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]  # 避免裁剪的层名模式

        # 遍历所有命名线性层
        for name in named_linears:
            # 由于qk bmm（查询键批矩阵乘法），很难精确裁剪
            if any([_ in name for _ in avoid_clipping]):
                continue  # 跳过查询和键相关层

            named_linears[name].to(get_best_device())  # 移动到最佳设备
            # 计算该层的最佳裁剪值
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))  # 添加到裁剪列表
            named_linears[name].cpu()  # 移回CPU

        return clip_list  # 返回裁剪列表

    @torch.no_grad()  # 关闭梯度计算以节省内存
    def _compute_best_clip(
        self,
        w: torch.Tensor,        # 权重张量
        input_feat: torch.Tensor, # 输入特征
        n_grid=20,               # 网格搜索点数
        max_shrink=0.5,          # 最大收缩比例
        n_sample_token=512,      # 采样token数量
    ):
        """
        计算最佳裁剪阈值

        该方法通过网格搜索找到最佳的权重裁剪阈值，
        在保持精度的同时减少量化误差。

        Returns:
            torch.Tensor: 最佳裁剪阈值
        """
        assert w.dim() == 2  # 确保权重是2D张量
        org_w_shape = w.shape  # 保存原始形状
        # 权重形状变换：[co, ci] -> [co, 1, n_group, group size]
        # 输入特征变换：[n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]  # 获取组大小
        input_feat = input_feat.view(-1, input_feat.shape[-1])  # 展平输入特征
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)  # 重塑为4D

        # 计算输入特征的步长（最小为1）
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]  # 采样输入特征

        # 重塑权重为4D以便分批处理
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        # 输出通道批处理大小，防止OOM
        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64
        assert org_w_shape[0] % oc_batch_size == 0  # 确保可整除
        w_all = w  # 保存所有权重
        best_max_val_all = []  # 最佳最大值列表

        # 分批处理输出通道
        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            # 计算原始最大值：[co, 1, n_group, 1]
            org_max_val = w.abs().amax(dim=-1, keepdim=True)

            best_max_val = org_max_val.clone()  # 初始化最佳最大值
            min_errs = torch.ones_like(org_max_val) * 1e9  # 初始化最小误差
            input_feat = input_feat.to(w.device)  # 移动输入特征到权重设备
            # 计算原始输出：[co, n_token, n_group]
            org_out = (input_feat * w).sum(dim=-1)

            # 网格搜索最佳裁剪阈值
            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)  # 计算当前最大值
                min_val = -max_val  # 对称裁剪
                cur_w = torch.clamp(w, min_val, max_val)  # 应用裁剪
                q_w = self.pseudo_quantize_tensor(cur_w)[0]  # 量化裁剪后的权重
                cur_out = (input_feat * q_w).sum(dim=-1)  # 计算裁剪后的输出

                # 计算误差：[co, 1, n_group, 1]
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w  # 删除临时变量释放内存
                del cur_out
                cur_best_idx = err < min_errs  # 找到更好的索引
                min_errs[cur_best_idx] = err[cur_best_idx]  # 更新最小误差
                best_max_val[cur_best_idx] = max_val[cur_best_idx]  # 更新最佳最大值
            best_max_val_all.append(best_max_val)  # 添加到列表

        # 拼接所有最佳最大值
        best_max_val = torch.cat(best_max_val_all, dim=0)

        # 清理内存
        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)  # 返回压缩后的最佳最大值

    def init_quant(self, n_samples=128, max_seq_len=512):
        """
        初始化量化过程

        该方法获取模型层、准备校准数据集，并捕获第一层的输入和参数，
        为后续的量化过程做准备。

        Args:
            n_samples: 校准样本数量
            max_seq_len: 最大序列长度

        Returns:
            tuple: (模块列表, 层参数字典, 输入张量)
        """
        # 获取模型的所有层
        modules = self.awq_model.get_model_layers(self.model)
        # 获取校准数据集
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)  # 连接所有样本

        inps = []  # 输入列表
        layer_kwargs = {}  # 层参数字典

        best_device = get_best_device()  # 获取最佳设备
        modules[0] = modules[0].to(best_device)  # 将第一层移到最佳设备
        self.awq_model.move_embed(self.model, best_device)  # 移动嵌入层

        # 获取第0层的输入和参数
        # with_kwargs仅在PyTorch 2.0中支持
        # 目前使用Catcher技巧
        class Catcher(nn.Module):
            """捕获器类，用于捕获第一层的输入和参数"""
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # 假设第一个输入是隐藏状态
                if len(args) > 0:
                    hidden_states = args[0]
                    del args  # 删除其他参数
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)  # 取第一个参数

                inps.append(hidden_states)  # 保存隐藏状态
                layer_kwargs.update(kwargs)  # 更新层参数
                raise ValueError  # 提前退出以中断后续推理

        # 修补第0层以捕获输入和参数
        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # 配合提前退出
            pass
        modules[0] = modules[0].module  # 恢复原始层

        # 使用`prepare_inputs_for_generation`方法更新层参数，
        # 该方法会处理所有内容以避免意外错误
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # 移除不需要的input_ids
        layer_kwargs.pop("input_ids")

        del samples  # 删除样本
        inps = inps[0]  # 获取第一个输入

        modules[0] = modules[0].cpu()  # 将第0层移回CPU
        self.awq_model.move_embed(self.model, "cpu")  # 将嵌入层移回CPU

        clear_memory()  # 清理内存

        # 处理注意力掩码
        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )
        elif "qwen" in self.awq_model.model_type:
            layer_kwargs["attention_mask"] = None

        return modules, layer_kwargs, inps  # 返回模块、参数和输入

    def _get_input_feat(self, layer, named_linears):
        """
        获取输入特征

        该方法通过注册前向钩子来捕获所有线性层的输入特征，
        为后续的缩放和裁剪计算提供数据。

        Args:
            layer: 当前层
            named_linears: 命名线性层字典

        Returns:
            dict: 输入特征字典
        """
        # 首先，获取所有线性层的输入特征
        def cache_input_hook(m, x, y, name, feat_dict):
            """缓存输入特征的钩子函数"""
            x = x[0]  # 获取第一个参数（输入）
            x = x.detach().cpu()  # 分离梯度并移到CPU
            feat_dict[name].append(x)  # 添加到特征字典

        input_feat = defaultdict(list)  # 创建默认字典
        handles = []  # 钩子句柄列表

        # FIXME: Mixtral的变通方法，使用block_sparse_moe输入特征
        if self.awq_model.model_type == "mixtral":
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,  # 添加稀疏MoE模块
            }

        # DeepSeek V2/V3模型的特殊处理
        if self.awq_model.model_type == "deepseek_v2" or self.awq_model.model_type == "deepseek_v3":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,  # 添加MLP模块
            }

        # Qwen3 MoE模型的特殊处理
        if self.awq_model.model_type == "qwen3_moe":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,  # 添加MLP模块
            }

        # 为每个线性层注册前向钩子
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # 多GPU情况下的设备对齐

        # 获取输出作为下一层的输入
        # 清理参数，以防使用包含模块不支持的参数的transformers版本
        # 对于trust_remote_code模型很有用
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        # 执行前向传播以捕获输入特征
        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        # 移除所有钩子
        for h in handles:
            h.remove()

        # 现在解决缩放和裁剪问题
        def cat_and_assert(k, v):
            """连接特征并验证有效性"""
            x = torch.cat(v, dim=0)  # 连接特征
            assert x.shape[0] != 0, (
                f"{k} 的维度为零。这可能是因为没有数据通过（例如MoE中的专家未激活）。"
                "尝试增加max_calib_samples（警告：这会显著增加量化时间和内存使用。）"
            )
            return x

        # 处理所有输入特征
        input_feat = {k: cat_and_assert(k, v) for k, v in input_feat.items()}

        return input_feat  # 返回输入特征字典

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        清理模块参数，移除不支持的参数以避免不同版本transformers之间的兼容性问题

        该方法检查模块的forward函数签名，只保留模块支持的参数，
        这对于处理不同版本的transformers库和trust_remote_code模型很有用。

        Args:
            inputs_kwargs (dict): 要传递给模型层的输入参数字典
            module (torch.nn.Module): 要量化的目标模块

        Returns:
            dict: 清理后的参数字典，只包含模块支持的参数
        """
        # 获取模块forward函数的参数签名
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}  # 创建清理后的参数字典
        # 遍历输入参数，只保留模块支持的参数
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs  # 返回清理后的参数字典
