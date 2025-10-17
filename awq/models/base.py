# 导入标准库
import os  # 操作系统接口
import gc  # 垃圾回收机制
import warnings  # 警告控制

# 导入PyTorch相关库
import torch  # PyTorch主库
import torch.nn as nn  # PyTorch神经网络模块

# 导入第三方库
from tqdm import tqdm  # 进度条显示
from typing import List, Union, Dict  # 类型注解
from typing_extensions import Doc, Annotated  # 扩展类型注解
from huggingface_hub import snapshot_download, save_torch_state_dict  # HuggingFace Hub工具

# 导入AWQ线性层模块
from awq.modules.linear import (
    WQLinear_GEMM,  # GEMM量化线性层
    WQLinear_GEMV,  # GEMV量化线性层
    WQLinear_IPEX,  # Intel Extension for PyTorch量化线性层
    WQLinear_Marlin,  # Marlin量化线性层
    WQLinear_Exllama,  # ExLlama v1量化线性层
    WQLinear_ExllamaV2,  # ExLlama v2量化线性层
    WQLinear_GEMVFast,  # 快速GEMV量化线性层
    marlin_post_init,  # Marlin后处理初始化
    exllama_post_init,  # ExLlama后处理初始化
    exllamav2_post_init,  # ExLlama v2后处理初始化
    ipex_post_init,  # IPEX后处理初始化
)

# 导入AWQ工具模块
from awq.utils.module import (
    get_named_linears,  # 获取命名线性层
    set_op_by_name,  # 按名称设置操作
    exclude_layers_to_not_quantize,  # 排除不需要量化的层
    try_import,  # 尝试导入模块
)

# 导入AWQ实用工具
from awq.utils.utils import get_best_device, ipex_available, triton_available  # 设备检测和可用性检查

# 导入Transformers库组件
from transformers import (
    AutoConfig,  # 自动配置
    PreTrainedModel,  # 预训练模型基类
    PretrainedConfig,  # 预训练配置
    AutoProcessor,  # 自动处理器
    BaseImageProcessor,  # 基础图像处理器
    ProcessorMixin,  # 处理器混入类
    PreTrainedTokenizer,  # 预训练分词器
)

# 导入Accelerate库的大模型处理
from accelerate.big_modeling import (
    init_empty_weights,  # 初始化空权重
    load_checkpoint_and_dispatch,  # 加载检查点并分发
)

# 导入AWQ内部模块
from awq.models._config import AwqConfig  # AWQ配置类
from awq.modules.act import ScaledActivation  # 缩放激活函数
from awq.quantize.quantizer import AwqQuantizer  # AWQ量化器
from awq.utils.module import get_named_linears, set_op_by_name  # 模块工具（重复导入）


# 由于支持来自transformers的不同AutoModelForxxx类，需要定义一个自定义映射字典
# 将模型类型映射到相应的Transformers自动模型类
TRANSFORMERS_AUTO_MAPPING_DICT = {
    "mpt": "AutoModelForCausalLM",                    # MPT系列模型 - 因果语言模型
    "llama": "AutoModelForCausalLM",                  # LLaMA系列模型 - 因果语言模型
    "opt": "AutoModelForCausalLM",                    # OPT系列模型 - 因果语言模型
    "RefinedWeb": "AutoModelForCausalLM",            # RefinedWeb模型 - 因果语言模型
    "RefinedWebModel": "AutoModelForCausalLM",       # RefinedWeb模型（另一种命名） - 因果语言模型
    "exaone": "AutoModelForCausalLM",                # ExaOne模型 - 因果语言模型
    "falcon": "AutoModelForCausalLM",                # Falcon系列模型 - 因果语言模型
    "bloom": "AutoModelForCausalLM",                 # BLOOM系列模型 - 因果语言模型
    "gptj": "AutoModelForCausalLM",                  # GPT-J模型 - 因果语言模型
    "gpt_bigcode": "AutoModelForCausalLM",           # GPT-BigCode模型 - 因果语言模型
    "mistral": "AutoModelForCausalLM",               # Mistral系列模型 - 因果语言模型
    "mixtral": "AutoModelForCausalLM",               # Mixtral模型 - 因果语言模型
    "gpt_neox": "AutoModelForCausalLM",              # GPT-NeoX模型 - 因果语言模型
    "aquila": "AutoModelForCausalLM",                # Aquila模型 - 因果语言模型
    "Yi": "AutoModelForCausalLM",                    # Yi系列模型 - 因果语言模型
    "qwen": "AutoModelForCausalLM",                  # Qwen系列模型 - 因果语言模型
    "baichuan": "AutoModelForCausalLM",              # Baichuan系列模型 - 因果语言模型
    "llava": "AutoModelForVision2Seq",               # LLaVA模型 - 视觉到序列模型
    "qwen2": "AutoModelForCausalLM",                 # Qwen2系列模型 - 因果语言模型
    "qwen2_vl": "AutoModelForVision2Seq",            # Qwen2-VL模型 - 视觉到序列模型
    "qwen3": "AutoModelForCausalLM",                 # Qwen3系列模型 - 因果语言模型
    "qwen3_moe": "AutoModelForCausalLM",             # Qwen3 MoE模型 - 因果语言模型
    "gemma": "AutoModelForCausalLM",                 # Gemma系列模型 - 因果语言模型
    "gemma2": "AutoModelForCausalLM",                # Gemma2系列模型 - 因果语言模型
    "stablelm": "AutoModelForCausalLM",              # StableLM模型 - 因果语言模型
    "starcoder2": "AutoModelForCausalLM",            # StarCoder2模型 - 因果语言模型
    "llava_next": "AutoModelForVision2Seq",          # LLaVA-Next模型 - 视觉到序列模型
    "phi3": "AutoModelForCausalLM",                  # Phi-3模型 - 因果语言模型
    "phi3_v": "AutoModelForCausalLM",                # Phi-3 V模型 - 因果语言模型
    "cohere": "AutoModelForCausalLM",                # Cohere模型 - 因果语言模型
    "deepseek_v2": "AutoModelForCausalLM",           # DeepSeek V2模型 - 因果语言模型
    "deepseek_v3": "AutoModelForCausalLM",           # DeepSeek V3模型 - 因果语言模型
    "minicpm": "AutoModelForCausalLM",               # MiniCPM模型 - 因果语言模型
    "minicpm3": "AutoModelForCausalLM",              # MiniCPM3模型 - 因果语言模型
    "internlm2": "AutoModelForCausalLM",             # InternLM2模型 - 因果语言模型
    "qwen2_vl": "AutoModelForVision2Seq",            # Qwen2-VL模型 - 视觉到序列模型
    "qwen2_5_vl": "AutoModelForVision2Seq",          # Qwen2.5-VL模型 - 视觉到序列模型
    "qwen2_5_omni": "AutoModelForTextToWaveform",    # Qwen2.5-Omni模型 - 文本到波形模型
}


class BaseAWQForCausalLM(nn.Module):
    """
    AWQ因果语言模型基类

    这是所有AutoAWQ模型的基础类，提供了量化、加载、保存等核心功能。
    继承自nn.Module，支持PyTorch模型的标准操作。
    """

    def __init__(
        self,
        model: Annotated[PreTrainedModel, Doc("预训练或量化后的模型")],
        model_type: Annotated[str, Doc("模型类型，从config.json中获取")],
        is_quantized: Annotated[
            bool, Doc("指示当前模型是否已量化")
        ],
        config: Annotated[PretrainedConfig, Doc("模型的配置信息")],
        quant_config: Annotated[
            AwqConfig, Doc("模型的量化配置")
        ],
        processor: Annotated[
            BaseImageProcessor, Doc("可选的处理器，例如用于视觉模型")
        ],
    ):
        """
        初始化AWQ基础模型

        Args:
            model: 预训练或量化后的模型实例
            model_type: 模型类型字符串
            is_quantized: 布尔值，表示模型是否已量化
            config: 模型的配置对象
            quant_config: AWQ量化配置对象
            processor: 可选的处理器（用于多模态模型）
        """
        super().__init__()  # 调用父类初始化
        self.model: PreTrainedModel = model          # 存储实际模型
        self.model_type: str = model_type            # 存储模型类型
        self.is_quantized: bool = is_quantized        # 存储量化状态标志
        self.search_result = None                    # 搜索结果存储（暂未使用）
        self.config: PretrainedConfig = config       # 存储模型配置
        self.quant_config: AwqConfig = quant_config  # 存储量化配置
        self.processor: ProcessorMixin = processor    # 存储处理器（多模态模型使用）

    def to(self, device: Annotated[str, Doc("要将模型移动到的设备")]):
        """
        将模型移动到指定设备的工具函数

        Args:
            device: 目标设备（如"cpu", "cuda", "cuda:0"等）

        Returns:
            移动后的模型
        """
        return self.model.to(device)  # 调用内部模型的to方法

    def forward(self, *args, **kwargs):
        """
        模拟PyTorch的前向传播函数

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            模型的输出结果
        """
        return self.model(*args, **kwargs)  # 调用内部模型的forward方法

    def generate(self, *args, **kwargs):
        """
        模拟HuggingFace的生成函数

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            生成的序列
        """
        with torch.inference_mode():  # 关闭梯度计算以节省内存
            return self.model.generate(*args, **kwargs)  # 调用内部模型的generate方法

    @torch.no_grad()
    def quantize(
        self,
        tokenizer: Annotated[
            PreTrainedTokenizer, Doc("用于量化的分词器")
        ] = None,
        quant_config: Annotated[
            Dict, Doc("要使用的量化配置字典")
        ] = {},
        calib_data: Annotated[
            Union[str, List[str]],
            Doc(
                "校准数据集。可以是指向HuggingFace的字符串，也可以是预加载示例的列表"
            ),
        ] = "pileval",  # 默认使用pileval数据集
        split: Annotated[str, Doc("校准数据集的数据分割")] = "train",  # 默认使用训练集
        text_column: Annotated[str, Doc("校准数据集的文本列名")] = "text",  # 默认文本列名为"text"
        duo_scaling: Annotated[
            bool, Doc("是否同时使用权重和激活进行缩放，还是仅使用激活")
        ] = True,  # 默认使用双重缩放
        export_compatible: Annotated[
            bool,
            Doc(
                "此参数通过仅应用缩放而不实际量化到FP16来避免真实量化，"
                "用于导出兼容性"
            ),
        ] = False,
        apply_clip: Annotated[
            bool,
            Doc(
                "在量化期间是否对模型应用裁剪。某些模型在设置为False时可能表现更好"
            ),
        ] = True,  # 默认应用裁剪
        n_parallel_calib_samples: Annotated[
            int,
            Doc(
                "并行运行的校准样本数量。"
                "如果max_calib_samples足够高，大量的并行样本可能在量化过程中导致OOM。"
                "如果为None，则同时运行所有样本。"
                "您可以将其设置为较低数值以实现更节省内存的量化。"
            ),
        ] = None,  # 默认不限制并行样本数
        max_calib_samples: Annotated[
            int, Doc("运行模型的最大样本数量")
        ] = 128,  # 默认最多128个样本
        max_calib_seq_len: Annotated[
            int,
            Doc(
                "校准数据集的最大序列长度。丢弃大于max_calib_seq_len的样本"
            ),
        ] = 512,  # 默认最大序列长度为512
        max_chunk_memory: Annotated[
            int,
            Doc(
                "损失计算和每通道均值被优化为分块计算。"
                "调整此参数以增加或减少这些计算的内存使用。"
                "默认为1GB (1024 * 1024 * 1024)"
            ),
        ] = 1024  # 默认1GB内存限制
        * 1024
        * 1024,
        quantizer_cls: Annotated[
            AwqQuantizer,
            Doc("如果要自定义量化类，可以使用AwqQuantizer作为基类")
        ] = AwqQuantizer,  # 默认使用AwqQuantizer类
        **kwargs,
    ):
        """
        主要的量化函数，用于对模型进行AWQ量化

        示例:

        ```python
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model_path = "..."
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
        model.quantize(tokenizer, quant_config)
        ```
        """
        # 从字典创建AWQ配置对象
        self.quant_config: AwqConfig = AwqConfig.from_dict(quant_config)

        # 如果存在不需要转换的模块列表，将其添加到量化配置中
        if hasattr(self, "modules_to_not_convert"):
            self.quant_config.modules_to_not_convert = self.modules_to_not_convert

        # 创建量化器实例
        self.quantizer = quantizer_cls(
            self,  # AWQ模型实例
            self.model,  # 实际模型
            tokenizer,  # 分词器
            self.quant_config.w_bit,  # 量化位数
            self.quant_config.q_group_size,  # 量化组大小
            self.quant_config.zero_point,  # 是否使用零点
            self.quant_config.version,  # 量化版本
            calib_data,  # 校准数据
            split,  # 数据分割
            text_column,  # 文本列
            duo_scaling,  # 双重缩放
            modules_to_not_convert=self.quant_config.modules_to_not_convert,  # 不转换的模块
            export_compatible=export_compatible,  # 导出兼容性
            apply_clip=apply_clip,  # 应用裁剪
            n_parallel_calib_samples=n_parallel_calib_samples,  # 并行校准样本数
            max_calib_samples=max_calib_samples,  # 最大校准样本数
            max_calib_seq_len=max_calib_seq_len,  # 最大校准序列长度
            max_chunk_memory=max_chunk_memory,  # 最大块内存
            **kwargs,  # 其他参数
        )
        # 执行量化过程
        self.quantizer.quantize()

        # 标记模型已量化
        self.is_quantized = True

    @torch.no_grad()  # 关闭梯度计算以节省内存
    def pack(self):
        """
        打包量化模型权重，使其兼容CUDA推理的工具函数

        注意：如果使用相同的quant_path，save_quantized会覆盖现有权重。

        示例:

        ```python
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            export_compatible=True
        )
        model.save_quantized(...)  # 生成GGUF/其他兼容权重
        model.pack(...) # 使模型兼容CUDA
        model.save_quantized(...)  # 生成CUDA兼容权重
        ```
        """
        self.quantizer.pack()  # 调用量化器的打包方法

    @staticmethod
    def fuse_layers(model):
        """
        静态方法：融合模型层以提高推理速度

        Args:
            model: 要融合的模型

        Note:
            这是一个基础方法，具体实现由各个子类覆盖
        """
        pass  # 基础实现为空，由子类覆盖

    def save_quantized(
        self,
        save_dir: Annotated[str, Doc("保存模型的目录路径")],
        safetensors: Annotated[
            bool, Doc("是否将模型保存为safetensors格式或torch文件")
        ] = True,  # 默认使用safetensors格式
        shard_size: Annotated[
            str, Doc("将大模型分割成多个块的分片大小")
        ] = "5GB",  # 默认分片大小为5GB
    ):
        """
        保存量化模型到指定目录

        Args:
            save_dir: 保存模型的目录路径
            safetensors: 是否使用safetensors格式保存
            shard_size: 模型分片大小
        """
        # 如果路径以斜杠结尾，移除最后的斜杠
        save_dir = save_dir[:-1] if save_dir[-1] == "/" else save_dir

        # 保存模型
        # 创建一个空模块类用于保存基础结构
        class EmptyModule(nn.Module):
            def __init__(self):
                super(EmptyModule, self).__init__()

            def forward(self, x):
                return x

        # 使用空的状态字典保存模型和配置文件
        self.model.config.quantization_config = self.quant_config.to_transformers_dict()  # 设置量化配置
        self.model.generation_config.do_sample = True  # 设置采样参数
        self.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())  # 保存基础结构

        # 如果是视觉Transformer模型，还需要保存处理器
        if self.processor is not None:
            self.processor.save_pretrained(save_dir)  # 保存处理器配置

        # 移除空的状态字典文件
        default_paths = [
            f"{save_dir}/model.safetensors",  # safetensors格式文件
            f"{save_dir}/pytorch_model.bin",  # PyTorch格式文件
        ]
        for path in default_paths:
            if os.path.exists(path):
                os.remove(path)  # 删除空的状态文件

        # 使用HuggingFace的工具保存实际的模型权重
        save_torch_state_dict(
            state_dict=self.model.state_dict(),  # 模型状态字典
            save_directory=save_dir,  # 保存目录
            max_shard_size=shard_size,  # 分片大小
            safe_serialization=safetensors,  # 安全序列化
            force_contiguous=True,  # 强制连续内存
            shared_tensors_to_discard=self.model._tied_weights_keys,  # 共享权重键
        )

    @classmethod  # 类方法装饰器
    def from_pretrained(
        self,
        model_path: Annotated[str, Doc("HuggingFace模型路径或本地模型路径")],
        model_type: Annotated[str, Doc("模型类型，从config.json中加载")],
        torch_dtype: Annotated[
            torch.dtype,
            Doc(
                "加载模型的数据类型。除float16外可能不兼容其他数值类型"
            ),
        ] = torch.float16,  # 默认使用float16
        trust_remote_code: Annotated[
            bool,
            Doc(
                "对于尚未集成到transformers中的HuggingFace仓库很有用"
            ),
        ] = True,  # 默认信任远程代码
        safetensors: Annotated[
            bool, Doc("是否下载/加载safetensors格式而不是torch权重格式")
        ] = True,  # 默认使用safetensors
        device_map: Annotated[
            Union[str, Dict],
            Doc(
                "将传递给transformers模型加载方法的设备映射"
            ),
        ] = "auto",  # 默认自动分配设备
        download_kwargs: Annotated[
            Dict,
            Doc("用于配置下载模型参数"),
        ] = None,  # 默认无额外下载参数
        low_cpu_mem_usage: Annotated[
            bool,
            Doc("从transformers加载时是否使用低CPU内存模式")
        ] = True,  # 默认使用低CPU内存
        use_cache: Annotated[
            bool,
            Doc("是否在transformers中使用use_cache参数")
        ] = False,  # 默认不使用缓存
        **model_init_kwargs: Annotated[
            Dict,
            Doc(
                "在初始化期间传递给模型的额外关键字参数"
            ),
        ],  # 其他模型初始化参数
    ):
        """
        初始化预训练模型的方法，通常使用FP16格式

        Returns:
            BaseAWQForCausalLM: 加载的AWQ模型实例
        """
        # 获取权重路径和量化配置
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            "",  # 空模型文件名（表示加载全部）
            safetensors,
            trust_remote_code=trust_remote_code,
            download_kwargs=download_kwargs,
        )

        # 根据模型类型获取对应的transformers类名
        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)  # 获取类对象

        processor = None  # 初始化处理器为None
        # 如果是多模态模型，需要加载处理器
        if target_cls_name == "AutoModelForVision2Seq" or target_cls_name == "AutoModelForTextToWaveform":
            processor = AutoProcessor.from_pretrained(model_weights_path)

        # 设置低CPU内存使用参数
        if model_init_kwargs.get("low_cpu_mem_usage") is None:
            model_init_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

        # 设置缓存使用参数（仅对非多模态模型）
        if model_init_kwargs.get("use_cache") is None and not ((target_cls_name == "AutoModelForVision2Seq") or (target_cls_name == "AutoModelForTextToWaveform")):
            model_init_kwargs["use_cache"] = use_cache

        # 如果不是量化模型，必须使用AutoModelForCausalLM加载
        model = target_cls.from_pretrained(
            model_weights_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            use_safetensors=safetensors,
            device_map=device_map,
            **model_init_kwargs,
        )

        model.eval()  # 设置模型为评估模式

        # 返回AWQ模型实例
        return self(
            model,  # 实际模型
            model_type,  # 模型类型
            is_quantized=False,  # 标记为未量化
            config=config,  # 模型配置
            quant_config=quant_config,  # 量化配置
            processor=processor,  # 处理器
        )

    @classmethod
    def from_quantized(
        self,
        model_path: Annotated[str, Doc("A Huggingface path or local path to a model.")],
        model_type: Annotated[str, Doc("The model type, loaded from config.json.")],
        model_filename: Annotated[
            str, Doc("Load a specific model's filename by specifying this argument.")
        ] = "",
        max_seq_len: Annotated[
            int,
            Doc(
                "The maximum sequence cached sequence length of the model. Larger values may increase loading time and memory usage."
            ),
        ] = None,
        torch_dtype: Annotated[
            torch.dtype,
            Doc(
                "The dtype to load the model as. May not work with other values than float16."
            ),
        ] = torch.float16,
        trust_remote_code: Annotated[
            bool,
            Doc(
                "Useful for Huggingface repositories that have not been integrated into transformers yet."
            ),
        ] = True,
        safetensors: Annotated[
            bool, Doc("Whether to download/load safetensors instead of torch weights.")
        ] = True,
        fuse_layers: Annotated[
            bool,
            Doc(
                "Whether to use fused/optimized combination of layers for increased speed."
            ),
        ] = True,
        use_exllama: Annotated[
            bool, Doc("Whether to map the weights to ExLlamaV1 kernels.")
        ] = False,
        use_exllama_v2: Annotated[
            bool, Doc("Whether to map the weights to ExLlamaV2 kernels.")
        ] = False,
        use_ipex: Annotated[
            bool, Doc("Whether to map the weights to ipex kernels for CPU and XPU device.")
        ] = False,
        device_map: Annotated[
            Union[str, Dict],
            Doc(
                "A device map that will be passed onto the model loading method from transformers."
            ),
        ] = "balanced",
        max_memory: Annotated[
            Dict[Union[int, str], Union[int, str]],
            Doc(
                'A dictionary device identifier to maximum memory which will be passed onto the model loading method from transformers. For example：{0: "4GB",1: "10GB"'
            ),
        ] = None,
        offload_folder: Annotated[
            str,
            Doc("The folder ot offload the model to."),
        ] = None,
        download_kwargs: Annotated[
            Dict,
            Doc("Used for configure download model"),
        ] = None,
        **config_kwargs: Annotated[
            Dict,
            Doc(
                "Additional kwargs that are passed to the config during initialization."
            ),
        ],
    ):
        """A method for initialization of a quantized model, usually in INT4."""
        # [STEP 1-2] Load weights path and configs
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            model_filename,
            safetensors,
            trust_remote_code,
            max_seq_len=max_seq_len,
            download_kwargs=download_kwargs,
            **config_kwargs,
        )

        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)

        # [STEP 3] Load model
        with init_empty_weights():
            model = target_cls.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )

        best_device = get_best_device()
        if best_device == "cpu" or (best_device == "xpu:0" and not triton_available):
            use_ipex = True
        if use_ipex and not ipex_available:
            raise ImportError(
                "Please install intel_extension_for_pytorch with "
                "`pip install intel_extension_for_pytorch` for 'ipex' kernel!"
            )
        # Prepare WQLinear layers, replace nn.Linear
        self._load_quantized_modules(
            self,
            model,
            quant_config,
            quant_config.version,
            use_exllama=use_exllama,
            use_exllama_v2=use_exllama_v2,
            use_ipex=use_ipex,
        )

        model.tie_weights()

        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            model,
            checkpoint=model_weights_path,
            device_map=device_map,
            max_memory=max_memory,
            no_split_module_classes=[self.layer_type],
            offload_folder=offload_folder,
            dtype=torch_dtype,
        )

        # Dispath to devices
        awq_ext, msg = try_import("awq_ext")
        if fuse_layers:
            if best_device in ["mps", "cuda:0"] and awq_ext is None:
                warnings.warn("Skipping fusing modules because AWQ extension is not installed." + msg)
            else:
                self.fuse_layers(model)

        if use_ipex:
            # repack qweight to match the ipex kernel.
            model = ipex_post_init(model)
        elif quant_config.version == "marlin":
            model = marlin_post_init(model)
        elif use_exllama:
            # creates q4 handle
            model = exllama_post_init(model)
        elif use_exllama_v2:
            # creates q4 handle and allocates scratch spaces wrt max_input_len and max_batch_size
            model = exllamav2_post_init(
                model,
                max_input_len=max_seq_len or 2048,
                max_batch_size=int(os.getenv("AWQ_BATCH_SIZE", 1)),
            )

        model.eval()

        return self(
            model,
            model_type,
            is_quantized=True,
            config=config,
            quant_config=quant_config,
            processor=None,
        )

    def _load_config(
        self,
        model_path,
        model_filename,
        safetensors=True,
        trust_remote_code=True,
        max_seq_len=4096,
        download_kwargs=None,
        **config_kwargs,
    ):
        # [STEP 1] Download model if path is not a directory
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*", "optimizer.pt", "*.onnx*"]
            if safetensors:
                ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
            else:
                ignore_patterns.append("*.safetensors*")

            if download_kwargs is None:
                download_kwargs = {}

            if "ignore_patterns" in download_kwargs:
                download_kwargs_ignore_patterns = download_kwargs.pop("ignore_patterns")

                if isinstance(download_kwargs_ignore_patterns, str):
                    ignore_patterns.append(download_kwargs_ignore_patterns)
                elif isinstance(download_kwargs_ignore_patterns, list):
                    ignore_patterns.extend(download_kwargs_ignore_patterns)

            model_path = snapshot_download(
                model_path, ignore_patterns=ignore_patterns, **download_kwargs
            )

        if model_filename != "":
            model_weights_path = model_path + f"/{model_filename}"
        else:
            model_weights_path = model_path

        # [STEP 2] Load config and set sequence length
        # TODO: Create BaseAWQConfig class
        quant_config = AwqConfig.from_pretrained(model_path)

        # Load model config and set max generation length
        if max_seq_len is None and hasattr(self, "max_seq_len_key"):
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = getattr(config, self.max_seq_len_key, 2048)
            # To add the generate support for Multi-modal models as well
            if hasattr(config, "text_config"):
                config.text_config.max_seq_len = getattr(
                    config, self.max_seq_len_key, 2048
                )
        else:
            max_seq_len = 2048 if max_seq_len is None else max_seq_len
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **config_kwargs
            )
            config.max_seq_len = max_seq_len

        return model_weights_path, config, quant_config

    def _load_quantized_modules(
        self, model, quant_config, version, use_exllama, use_exllama_v2, use_ipex=False
    ):
        # Real quantization of weights
        assert not (
            version == "gemv" and (use_exllama or use_exllama_v2 or use_ipex)
        ), "Exllama kernels only support GEMM version."

        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Filter out the linear layers we don't want to include
            named_linears = exclude_layers_to_not_quantize(
                named_linears, quant_config.modules_to_not_convert
            )

            # Replace activation functions
            self._scale_activations(self, layer)

            # Replace nn.Linear with WQLinear
            for name, module in named_linears.items():
                if use_ipex:
                    q_linear_module = WQLinear_IPEX
                elif version == "marlin":
                    q_linear_module = WQLinear_Marlin
                elif use_exllama:
                    q_linear_module = WQLinear_Exllama
                elif use_exllama_v2:
                    q_linear_module = WQLinear_ExllamaV2
                elif version == "gemm":
                    q_linear_module = WQLinear_GEMM
                elif version == "gemv":
                    q_linear_module = WQLinear_GEMV
                elif version == "gemv_fast":
                    q_linear_module = WQLinear_GEMVFast


                q_linear = q_linear_module.from_linear(
                    module, quant_config.w_bit, quant_config.q_group_size, True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

            if not use_ipex:
                torch.cuda.empty_cache()
            gc.collect()

    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer)

        if scale_dict["is_scalable"]:
            if not isinstance(scale_dict["scale_layer"], ScaledActivation):
                param = next(layer.parameters())

                # get activation scale
                scale_like = torch.ones(
                    scale_dict["scale_shape"], dtype=param.dtype, device=param.device
                )

                # scale activation
                scaled_act = ScaledActivation(scale_dict["scale_layer"], scale_like)
                set_op_by_name(layer, scale_dict["scale_name"], scaled_act)
