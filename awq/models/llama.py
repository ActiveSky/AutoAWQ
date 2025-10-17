"""
Llama模型AWQ量化实现模块

该模块为Llama架构的大语言模型提供AWQ（Activation-aware Weight Quantization）
量化支持。Llama是由Meta开发的主流开源大语言模型架构，具有优异的性能
和广泛的应用场景。

模块功能：
- 实现Llama模型的AWQ量化策略
- 提供Llama特定的层融合优化
- 支持注意力机制和MLP组件的量化
- 优化推理性能和内存效率

核心组件：
1. LlamaAWQForCausalLM: Llama模型的AWQ量化主类
2. LlamaFuser: Llama模型的层融合器

量化策略：
- 注意力输入层：input_layernorm -> [q_proj, k_proj, v_proj]
- 注意力输出层：v_proj -> o_proj (条件性融合)
- MLP第一层：post_attention_layernorm -> [gate_proj, up_proj]
- MLP第二层：up_proj -> down_proj

使用示例：
```python
from awq import AutoAWQForCausalLM
from awq.models.llama import LlamaAWQForCausalLM

# 加载和量化Llama模型
model = AutoAWQForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
model.quantize(tokenizer, quant_config={'w_bit': 4, 'q_group_size': 128})
model.save_quantized('llama-7b-awq')
```
"""

# 导入必要的库
import tqdm  # 进度条显示库，用于可视化层融合过程
from typing import List, Tuple  # 类型注解支持，提高代码可读性
from .base import BaseAWQForCausalLM  # 导入AWQ基础模型类
from awq.utils.fused_utils import fuse_qkv  # 导入QKV融合工具函数
from awq.modules.fused.block import LlamaLikeBlock  # 导入Llama类结构的融合块
from awq.modules.fused.model import LlamaLikeModel  # 导入Llama类结构的融合模型
from transformers.models.llama.modeling_llama import (  # 导入原始HuggingFace Llama模型组件
    LlamaDecoderLayer as OldLlamaDecoderLayer,  # 原始Llama解码器层
    LlamaForCausalLM as OldLlamaForCausalLM,   # 原始Llama因果语言模型
)
from awq.modules.fused.norm import FasterTransformerRMSNorm  # 导入加速的RMS归一化层


class LlamaAWQForCausalLM(BaseAWQForCausalLM):
    """
    Llama模型的AWQ量化实现类

    继承自BaseAWQForCausalLM，为Llama架构的模型提供AWQ量化支持。
    该类定义了Llama模型特定的量化参数、融合方法和层配置策略。

    Attributes:
        layer_type (str): 解码器层的类型标识符
        max_seq_len_key (str): 最大序列长度配置的键名
    """

    layer_type = "LlamaDecoderLayer"  # 指定Llama解码器层类型
    max_seq_len_key = "max_position_embeddings"  # 配置文件中最大序列长度的键名

    @staticmethod
    def fuse_layers(model: OldLlamaForCausalLM):
        """
        融合Llama模型层以提高推理效率

        该方法创建LlamaFuser实例并执行层融合操作，将QKV注意力机制
        融合为单一操作，并使用优化的归一化层替换原始层。

        Args:
            model (OldLlamaForCausalLM): 待融合的原始Llama模型

        Returns:
            None: 直接在原模型上执行融合操作
        """
        fuser = LlamaFuser(model)  # 创建层融合器实例
        fuser.fuse_transformer()  # 执行Transformer层融合

    @staticmethod
    def get_model_layers(model: OldLlamaForCausalLM):
        """
        获取Llama模型的所有解码器层

        提取模型中的所有LlamaDecoderLayer，用于量化和融合操作。

        Args:
            model (OldLlamaForCausalLM): 原始Llama模型

        Returns:
            List[LlamaDecoderLayer]: 模型中所有解码器层的列表
        """
        return model.model.layers  # 返回模型的所有解码器层

    @staticmethod
    def get_act_for_scaling(module: OldLlamaDecoderLayer):
        """
        获取Llama层的缩放配置信息

        Llama模型的激活函数通常不需要进行额外的缩放处理，
        因此返回不可缩放的配置。这是Llama架构的特定行为。

        Args:
            module (OldLlamaDecoderLayer): Llama解码器层

        Returns:
            Dict[str, bool]: 缩放配置字典，is_scalable为False表示不可缩放
        """
        return dict(is_scalable=False)  # Llama层的激活函数不支持缩放

    @staticmethod
    def move_embed(model: OldLlamaForCausalLM, device: str):
        """
        将嵌入层移动到指定设备

        在量化和推理过程中，需要将嵌入相关的组件移动到目标设备
       （CPU/GPU）以确保计算的一致性和效率。

        Args:
            model (OldLlamaForCausalLM): Llama模型实例
            device (str): 目标设备名称（如'cpu'、'cuda:0'等）

        Returns:
            None: 直接修改模型中的嵌入层设备位置
        """
        model.model.embed_tokens = model.model.embed_tokens.to(device)  # 移动token嵌入层到目标设备
        model.model.rotary_emb = model.model.rotary_emb.to(device)  # 移动旋转位置编码到目标设备

    @staticmethod
    def get_layers_for_scaling(module: OldLlamaDecoderLayer, input_feat, module_kwargs):
        """
        获取需要进行缩放处理的层配置

        该方法定义了Llama模型中哪些线性层需要进行AWQ量化中的缩放处理。
        它按照Llama架构的结构定义了4个主要的缩放组，每个组包含
        前置操作层和需要进行缩放的目标层。

        Args:
            module (OldLlamaDecoderLayer): Llama解码器层实例
            input_feat (Dict): 输入特征字典，包含各层的激活值
            module_kwargs (Dict): 模块参数字典，用于前向传播

        Returns:
            List[Dict]: 缩放层配置列表，每个配置包含：
                - prev_op: 前置操作层（通常是归一化层或前一层）
                - layers: 需要进行缩放的目标层列表
                - inp: 输入特征数据
                - module2inspect: 要检查的模块（可选）
                - kwargs: 模块参数（可选）
        """
        layers = []

        # ==================== 第一组：注意力输入层 ====================
        # 前置操作：输入层归一化 (input_layernorm)
        # 目标层：QKV投影层 (q_proj, k_proj, v_proj)
        # 这些层接收归一化后的输入，需要进行一致的缩放处理
        layers.append(
            dict(
                prev_op=module.input_layernorm,  # 前置层：输入归一化
                layers=[
                    module.self_attn.q_proj,  # 查询投影层
                    module.self_attn.k_proj,  # 键投影层
                    module.self_attn.v_proj,  # 值投影层
                ],
                inp=input_feat["self_attn.q_proj"],  # Q投影层的输入特征
                module2inspect=module.self_attn,    # 要检查的模块：注意力层
                kwargs=module_kwargs,               # 前向传播参数
            )
        )

        # ==================== 第二组：注意力输出层 ====================
        # 前置操作：值投影层 (v_proj)
        # 目标层：输出投影层 (o_proj)
        # 注意：只有当V和O投影权重形状相同时才应用此缩放
        # 参考: https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,  # 前置层：值投影
                    layers=[module.self_attn.o_proj], # 目标层：输出投影
                    inp=input_feat["self_attn.o_proj"], # 输出投影层的输入特征
                )
            )

        # ==================== 第三组：MLP第一层 ====================
        # 前置操作：注意力后归一化 (post_attention_layernorm)
        # 目标层：MLP的门控投影和上投影层 (gate_proj, up_proj)
        # 这些层并行处理注意力输出，需要一致的缩放
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,  # 前置层：注意力后归一化
                layers=[module.mlp.gate_proj, module.mlp.up_proj],  # 目标层：门控和上投影
                inp=input_feat["mlp.gate_proj"],  # 门控投影层的输入特征
                module2inspect=module.mlp,        # 要检查的模块：MLP层
            )
        )

        # ==================== 第四组：MLP第二层 ====================
        # 前置操作：MLP上投影层 (up_proj)
        # 目标层：MLP下投影层 (down_proj)
        # 完成MLP的前向传播路径
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,       # 前置层：MLP上投影
                layers=[module.mlp.down_proj],    # 目标层：MLP下投影
                inp=input_feat["mlp.down_proj"],  # 下投影层的输入特征
            )
        )

        return layers  # 返回所有缩放层配置


class LlamaFuser:
    """
    Llama模型层融合器

    该类负责将原始Llama模型中的多个层融合为优化的单一操作，
    主要包括QKV注意力机制的融合和归一化层的优化。
    融合后的模型具有更好的推理性能和内存效率。

    Attributes:
        model (OldLlamaForCausalLM): 待融合的原始Llama模型
        llama_blocks (List[Tuple[str, OldLlamaDecoderLayer]]):
            模型中所有Llama解码器层的名称和实例列表
    """

    def __init__(self, model: OldLlamaForCausalLM):
        """
        初始化Llama层融合器

        创建融合器实例并识别模型中所有的Llama解码器层，
        为后续的融合操作做准备。

        Args:
            model (OldLlamaForCausalLM): 待融合的原始Llama模型
        """
        self.model = model  # 保存模型引用

        # 识别模型中所有的LlamaDecoderLayer
        # 使用类名匹配来找到所有解码器层
        self.llama_blocks: List[Tuple[str, OldLlamaDecoderLayer]] = [
            (name, module)  # (层名称, 层实例)的元组
            for name, module in self.model.named_modules()  # 遍历所有命名模块
            if "LlamaDecoderLayer".lower() in module.__class__.__name__.lower()  # 匹配LlamaDecoderLayer类名
        ]

    def fuse_transformer(self):
        """
        执行Transformer层的融合操作

        该方法是层融合的核心，它遍历模型的所有LlamaDecoderLayer，
        将每个层中的QKV注意力机制融合为单一操作，并用优化的
        FasterTransformerRMSNorm替换原始的归一化层。

        融合过程包括：
        1. QKV投影层融合：将q_proj、k_proj、v_proj合并为单一操作
        2. 归一化层优化：用FasterTransformerRMSNorm替换原始层
        3. 创建LlamaLikeBlock：封装融合后的组件
        4. 构建新模型：用LlamaLikeModel替换原始模型结构

        Returns:
            None: 直接修改self.model.model为融合后的新模型
        """
        blocks = []  # 存储融合后的块列表

        # 遍历每个Llama解码器层进行融合
        module: OldLlamaDecoderLayer  # 类型注解
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            # 获取当前层所在的设备（CPU/GPU）
            device = next(iter(module.state_dict().values())).device

            # ==================== QKV注意力融合 ====================
            # 将查询、键、值的三个独立投影层融合为单一操作
            # 这可以显著减少内存访问次数和计算开销
            qkv = fuse_qkv(
                module,                      # 当前解码器层
                module.self_attn.q_proj,     # 查询投影层
                module.self_attn.k_proj,     # 键投影层
                module.self_attn.v_proj,     # 值投影层
            )

            # ==================== 优化归一化层 ====================
            # 用FasterTransformer优化的RMS归一化层替换原始实现
            # 提供更好的性能和数值稳定性
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight,          # 输入归一化权重
                module.input_layernorm.variance_epsilon  # 数值稳定参数
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,          # 注意力后归一化权重
                module.post_attention_layernorm.variance_epsilon, # 数值稳定参数
            )

            # ==================== 创建融合块 ====================
            # 将融合后的组件封装为LlamaLikeBlock
            blocks.append(
                LlamaLikeBlock(
                    hidden_size=self.model.config.hidden_size,              # 隐藏层维度
                    n_heads=self.model.config.num_attention_heads,          # 注意力头数
                    n_kv_heads=self.model.config.num_key_value_heads,      # 键值注意力头数
                    qkv_layer=qkv,                                          # 融合后的QKV层
                    o_proj=module.self_attn.o_proj,                        # 输出投影层
                    mlp=module.mlp,                                         # MLP层（保持原样）
                    norm_1=norm_1,                                         # 优化的第一归一化层
                    norm_2=norm_2,                                         # 优化的第二归一化层
                    dev=device,                                            # 设备位置
                    max_seq_len=self.model.config.max_seq_len,             # 最大序列长度
                    rope_theta=self.model.config.rope_theta,               # RoPE旋转参数
                )
            )

        # ==================== 构建新模型 ====================
        # 用融合后的块构建新的LlamaLikeModel
        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,    # 词汇表大小
            blocks,                          # 融合后的所有解码器块
            self.model.model.embed_tokens,   # token嵌入层（保持不变）
            self.model.model.norm,           # 最终归一化层（保持不变）
        )
        # 设置blocks属性以便后续访问
        setattr(self.model.model, "blocks", self.model.model.blocks)
