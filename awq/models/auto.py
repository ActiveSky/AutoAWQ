# 导入必要的库
import os
import torch
import logging
from transformers import AutoConfig
from awq.models import *  # 导入所有AWQ模型实现
from awq.models.base import BaseAWQForCausalLM  # 导入AWQ基础模型类


# AWQ因果语言模型映射表，将模型类型名称映射到对应的AWQ实现类
AWQ_CAUSAL_LM_MODEL_MAP = {
    "mpt": MptAWQForCausalLM,                    # MPT系列模型
    "llama": LlamaAWQForCausalLM,                # LLaMA系列模型
    "opt": OptAWQForCausalLM,                    # OPT系列模型
    "RefinedWeb": FalconAWQForCausalLM,          # RefinedWeb模型（Falcon的一种）
    "RefinedWebModel": FalconAWQForCausalLM,     # RefinedWeb模型的另一种命名
    "exaone": ExaoneAWQForCausalLM,              # ExaOne模型
    "falcon": FalconAWQForCausalLM,              # Falcon系列模型
    "bloom": BloomAWQForCausalLM,                # BLOOM系列模型
    "gptj": GPTJAWQForCausalLM,                  # GPT-J模型
    "gpt_bigcode": GptBigCodeAWQForCausalLM,     # GPT-BigCode模型（StarCoder的前身）
    "mistral": MistralAWQForCausalLM,            # Mistral系列模型
    "mixtral": MixtralAWQForCausalLM,            # Mixtral模型（MoE架构）
    "gpt_neox": GPTNeoXAWQForCausalLM,           # GPT-NeoX模型
    "aquila": AquilaAWQForCausalLM,              # Aquila模型
    "Yi": YiAWQForCausalLM,                      # Yi系列模型
    "qwen": QwenAWQForCausalLM,                  # Qwen系列模型
    "baichuan": BaichuanAWQForCausalLM,          # Baichuan系列模型
    "llava": LlavaAWQForCausalLM,                # LLaVA多模态模型
    "qwen2": Qwen2AWQForCausalLM,                # Qwen2系列模型
    "qwen3": Qwen3AWQForCausalLM,                # Qwen3系列模型
    "qwen3_moe": Qwen3MoeAWQForCausalLM,         # Qwen3 MoE模型
    "gemma": GemmaAWQForCausalLM,                # Gemma系列模型
    "gemma2": Gemma2AWQForCausalLM,              # Gemma2系列模型
    "stablelm": StableLmAWQForCausalLM,          # StableLM模型
    "starcoder2": Starcoder2AWQForCausalLM,      # StarCoder2模型
    "llava_next": LlavaNextAWQForCausalLM,       # LLaVA-Next多模态模型
    "phi3": Phi3AWQForCausalLM,                  # Phi-3模型
    "phi3_v": Phi3VAWQForCausalLM,               # Phi-3-V视觉模型
    "cohere": CohereAWQForCausalLM,              # Cohere模型
    "deepseek_v2": DeepseekV2AWQForCausalLM,     # DeepSeek V2模型
    "deepseek_v3": DeepseekV3AWQForCausalLM,     # DeepSeek V3模型
    "minicpm": MiniCPMAWQForCausalLM,            # MiniCPM模型
    "internlm2": InternLM2AWQForCausalLM,        # InternLM2模型
    "minicpm3": MiniCPM3AWQForCausalLM,          # MiniCPM3模型
    "qwen2_vl": Qwen2VLAWQForCausalLM,           # Qwen2-VL视觉语言模型
    "qwen2_5_vl": Qwen2_5_VLAWQForCausalLM,      # Qwen2.5-VL视觉语言模型
    "qwen2_5_omni": Qwen2_5_OmniAWQForConditionalGeneration  # Qwen2.5-Omni多模态模型
}


def check_and_get_model_type(model_dir, trust_remote_code=True, **model_init_kwargs):
    """
    检查并获取模型类型

    Args:
        model_dir (str): 模型目录路径
        trust_remote_code (bool): 是否信任远程代码，默认为True
        **model_init_kwargs: 模型初始化的额外参数

    Returns:
        str: 模型类型名称

    Raises:
        TypeError: 当模型类型不被支持时抛出异常
    """
    # 从预训练模型加载配置
    config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code, **model_init_kwargs
    )
    # 检查模型类型是否在支持列表中
    if config.model_type not in AWQ_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


class AutoAWQForCausalLM:
    """
    AutoAWQ因果语言模型类，用于自动检测和加载相应的AWQ量化模型

    该类提供了统一的接口来加载不同类型的大语言模型，并自动应用AWQ量化技术。
    支持从预训练模型和量化模型两种方式加载。
    """

    def __init__(self):
        """
        禁止直接实例化，必须使用类方法from_quantized或from_pretrained
        """
        raise EnvironmentError(
            "You must instantiate AutoAWQForCausalLM with\n"
            "AutoAWQForCausalLM.from_quantized or AutoAWQForCausalLM.from_pretrained"
        )

    @classmethod
    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        safetensors=True,
        device_map=None,
        download_kwargs=None,
        low_cpu_mem_usage=True,
        use_cache=False,
        **model_init_kwargs,
    ) -> BaseAWQForCausalLM:
        """
        从预训练模型加载AWQ模型

        Args:
            model_path (str): 预训练模型路径
            torch_dtype (str): torch数据类型，默认为"auto"
            trust_remote_code (bool): 是否信任远程代码，默认为True
            safetensors (bool): 是否使用safetensors格式，默认为True
            device_map: 设备映射策略，默认为None
            download_kwargs: 下载相关参数，默认为None
            low_cpu_mem_usage (bool): 是否使用低CPU内存模式，默认为True
            use_cache (bool): 是否使用缓存，默认为False
            **model_init_kwargs: 模型初始化的额外参数

        Returns:
            BaseAWQForCausalLM: 加载的AWQ模型实例
        """
        # 检查并获取模型类型
        model_type = check_and_get_model_type(
            model_path, trust_remote_code, **model_init_kwargs
        )
        # 根据模型类型调用相应的from_pretrained方法
        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path,
            model_type,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            safetensors=safetensors,
            device_map=device_map,
            download_kwargs=download_kwargs,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_cache=use_cache,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
        self,
        quant_path,
        quant_filename="",
        max_seq_len=2048,
        trust_remote_code=True,
        fuse_layers=True,
        use_exllama=False,
        use_exllama_v2=False,
        use_ipex=False,
        batch_size=1,
        safetensors=True,
        device_map="balanced",
        max_memory=None,
        offload_folder=None,
        download_kwargs=None,
        **config_kwargs,
    ) -> BaseAWQForCausalLM:
        """
        从量化模型加载AWQ模型

        Args:
            quant_path (str): 量化模型路径
            quant_filename (str): 量化文件名，默认为空字符串
            max_seq_len (int): 最大序列长度，默认为2048
            trust_remote_code (bool): 是否信任远程代码，默认为True
            fuse_layers (bool): 是否融合层以提升推理速度，默认为True
            use_exllama (bool): 是否使用ExLlama内核，默认为False
            use_exllama_v2 (bool): 是否使用ExLlama v2内核，默认为False
            use_ipex (bool): 是否使用Intel Extension for PyTorch，默认为False
            batch_size (int): 批处理大小，默认为1
            safetensors (bool): 是否使用safetensors格式，默认为True
            device_map (str): 设备映射策略，默认为"balanced"
            max_memory: 最大内存限制，默认为None
            offload_folder: 卸载文件夹路径，默认为None
            download_kwargs: 下载相关参数，默认为None
            **config_kwargs: 配置相关的额外参数

        Returns:
            BaseAWQForCausalLM: 加载的AWQ量化模型实例
        """
        # 设置环境变量AWQ_BATCH_SIZE
        os.environ["AWQ_BATCH_SIZE"] = str(batch_size)
        # 检查并获取模型类型
        model_type = check_and_get_model_type(quant_path, trust_remote_code)

        # 检查是否使用了已弃用的max_new_tokens参数
        if config_kwargs.get("max_new_tokens") is not None:
            max_seq_len = config_kwargs["max_new_tokens"]
            logging.warning(
                "max_new_tokens argument is deprecated... gracefully "
                "setting max_seq_len=max_new_tokens."
            )

        # 根据模型类型调用相应的from_quantized方法
        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            quant_path,
            model_type,
            quant_filename,
            max_seq_len,
            trust_remote_code=trust_remote_code,
            fuse_layers=fuse_layers,
            use_exllama=use_exllama,
            use_exllama_v2=use_exllama_v2,
            use_ipex=use_ipex,
            safetensors=safetensors,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            download_kwargs=download_kwargs,
            **config_kwargs,
        )
