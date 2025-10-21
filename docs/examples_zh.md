# 示例

## 基本量化

AWQ执行零点量化，精度可达到4位整数。
您也可以指定其他比特率，如3位，但这些选项可能缺少用于运行推理的内核。

注意事项：

- 一些模型如Falcon仅兼容组大小64。
- 要使用Marlin，必须指定零点为False且版本为Marlin。

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_path = 'mistral-instruct-v0.2-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 量化
model.quantize(tokenizer, quant_config=quant_config)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'模型已量化并保存在 "{quant_path}"')
```

### 自定义数据

这包括一个加载wikitext或dolly的示例函数。
注意，目前所有超过512长度的样本都会被丢弃。

```python
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'lmsys/vicuna-7b-v1.5'
quant_path = 'vicuna-7b-v1.5-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 定义数据加载方法
def load_dolly():
    data = load_dataset('databricks/databricks-dolly-15k', split="train")

    # 连接数据
    def concatenate_data(x):
        return {"text": x['instruction'] + '\n' + x['context'] + '\n' + x['response']}

    concatenated = data.map(concatenate_data)
    return [text for text in concatenated["text"]]

def load_wikitext():
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

# 量化
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext())

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'模型已量化并保存在 "{quant_path}"')
```

#### 长上下文：优化量化

对于这个示例，我们将使用HuggingFaceTB/cosmopedia-100k，因为它是一个高质量的数据集，
我们可以直接根据令牌数量进行过滤。我们将使用Qwen2 7B，这是AutoAWQ中支持的较新模型之一，性能很高。以下示例在配备
RTX 4090 24 GB显存和107 GB系统内存的机器上顺利运行。

注意：调整`n_parallel_calib_samples`、`max_calib_samples`和`max_calib_seq_len`将有助于
在自定义数据集时避免OOM。

- AWQ算法的样本效率极高，因此128-256的`max_calib_samples`应该足以量化模型。更多的样本数量可能无法实现，除非有大量
可用内存或通过PR进一步优化AWQ以实现磁盘卸载。
- 当`n_parallel_calib_samples`设置为整数时，我们会卸载到系统内存以节省GPU显存。
如果您可用的内存很少，这可能会导致系统OOM；我们正在寻求进一步优化这个问题。

```python
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'Qwen/Qwen2-7B-Instruct'
quant_path = 'qwen2-7b-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def load_cosmopedia():
    data = load_dataset('HuggingFaceTB/cosmopedia-100k', split="train")
    data = data.filter(lambda x: x["text_token_length"] >= 2048)

    return [text for text in data["text"]]

# 量化
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=load_cosmopedia(),
    n_parallel_calib_samples=32,
    max_calib_samples=128,
    max_calib_seq_len=4096
)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'模型已量化并保存在 "{quant_path}"')
```

#### 编码模型

对于这个示例，我们将使用deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct，因为它是一个优秀的编码模型。

```python
from tqdm import tqdm
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
quant_path = 'deepseek-coder-v2-lite-instruct-awq'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM" }

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def load_openhermes_coding():
    data = load_dataset("alvarobartt/openhermes-preferences-coding", split="train")

    samples = []
    for sample in data:
        responses = [f'{response["role"]}: {response["content"]}' for response in sample["chosen"]]
        samples.append("\n".join(responses))

    return samples

# 量化
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=load_openhermes_coding(),
    # 如有必要请修改这些参数：
    # n_parallel_calib_samples=32,
    # max_calib_samples=128,
    # max_calib_seq_len=4096
)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'模型已量化并保存在 "{quant_path}"')
```

### 视觉语言模型

AutoAWQ支持一些视觉语言模型。目前，我们支持LLaVa 1.5和LLaVa 1.6（next）。

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'llava-hf/llama3-llava-next-8b-hf'
quant_path = 'llama3-llava-next-8b-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 量化
model.quantize(tokenizer, quant_config=quant_config)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'模型已量化并保存在 "{quant_path}"')
```

### GGUF导出

这会计算AWQ缩放因子并将其应用于模型，而无需运行实际量化。
这保持了AWQ的质量，因为权重已应用但跳过了量化以使其与其他框架兼容。

逐步操作：

- `quantize()`: 计算AWQ缩放因子并应用它们
- `save_pretrained()`: 以FP16保存非量化模型
- `convert.py`: 将Huggingface FP16权重转换为GGUF FP16权重
- `quantize`: 运行GGUF量化以获得实际量化权重，在这种情况下为4位。

```python
import os
import subprocess
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'mistralai/Mistral-7B-v0.1'
quant_path = 'mistral-awq'
llama_cpp_path = '/workspace/llama.cpp'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 6, "version": "GEMM" }

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 量化
# 注意：我们避免打包权重，因此在量化后无法在AutoAWQ中使用此模型。
# 保存的模型是FP16但已应用AWQ缩放因子。
model.quantize(
    tokenizer,
    quant_config=quant_config,
    export_compatible=True
)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'模型已量化并保存在 "{quant_path}"')

# GGUF转换
print('正在将模型转换为GGUF...')
llama_cpp_method = "q4_K_M"
convert_cmd_path = os.path.join(llama_cpp_path, "convert.py")
quantize_cmd_path = os.path.join(llama_cpp_path, "quantize")

if not os.path.exists(llama_cpp_path):
    cmd = f"git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_path} && cd {llama_cpp_path} && make LLAMA_CUBLAS=1 LLAMA_CUDA_F16=1"
    subprocess.run([cmd], shell=True, check=True)

subprocess.run([
    f"python {convert_cmd_path} {quant_path} --outfile {quant_path}/model.gguf"
], shell=True, check=True)

subprocess.run([
    f"{quantize_cmd_path} {quant_path}/model.gguf {quant_path}/model_{llama_cpp_method}.gguf {llama_cpp_method}"
], shell=True, check=True)
```

### 自定义量化器 (Qwen2 VL示例)

下面，Qwen团队提供了一个如何使用自定义量化器的示例。这可以
有效地使用多模态示例量化Qwen2 VL模型。

```python
import torch
import torch.nn as nn

from awq import AutoAWQForCausalLM
from awq.utils.qwen_vl_utils import process_vision_info
from awq.quantize.quantizer import AwqQuantizer, clear_memory, get_best_device

# 指定量化路径和超参数
model_path = "Qwen/Qwen2-VL-7B-Instruct"
quant_path = "qwen2-vl-7b-instruct"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(
    model_path, attn_implementation="flash_attention_2"
)

# 我们通过扩展AwqQuantizer来定义自己的量化器。
# 主要区别在于样本处理方式
# 量化过程初始化时。
class Qwen2VLAwqQuantizer(AwqQuantizer):
    def init_quant(self, n_samples=None, max_seq_len=None):
        modules = self.awq_model.get_model_layers(self.model)
        samples = self.calib_data

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # 获取第0层的输入和kwargs
        # with_kwargs仅在PyTorch 2.0中支持
        # 暂时使用这个Catcher技巧
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # 假设forward的第一个输入是隐藏状态
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # 提前退出以中断后续推理

        def move_to_device(obj: torch.Tensor | nn.Module, device: torch.device):
            def get_device(obj: torch.Tensor | nn.Module):
                if isinstance(obj, torch.Tensor):
                    return obj.device
                return next(obj.parameters()).device

            if get_device(obj) != device:
                obj = obj.to(device)
            return obj

        # 修补第0层以捕获输入和kwargs
        modules[0] = Catcher(modules[0])
        for k, v in samples.items():
            if isinstance(v, (torch.Tensor, nn.Module)):
                samples[k] = move_to_device(v, best_device)
        try:
            self.model(**samples)
        except ValueError:  # 处理提前退出
            pass
        finally:
            for k, v in samples.items():
                if isinstance(v, (torch.Tensor, nn.Module)):
                    samples[k] = move_to_device(v, "cpu")
        modules[0] = modules[0].module  # 恢复

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        return modules, layer_kwargs, inps

# 然后您需要为校准准备数据。您需要做的就是将样本放入列表中，
# 每个样本都是如下所示的典型聊天消息。您可以在`content`字段中指定文本和图像：
# dataset = [
#     # 消息0
#     [
#         {"role": "system", "content": "你是一个有用的助手。"},
#         {"role": "user", "content": "告诉我你是谁。"},
#         {"role": "assistant", "content": "我是一个名为Qwen的大型语言模型..."},
#     ],
#     # 消息1
#     [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": "file:///path/to/your/image.jpg"},
#                 {"type": "text", "text": "输出图像中的所有文本"},
#             ],
#         },
#         {"role": "assistant", "content": "图像中的文本是balabala..."},
#     ],
#     # 其他消息...
#     ...,
# ]
# 这里，我们使用字幕数据集**仅作演示**。您应该将其替换为您自己的sft数据集。
def prepare_dataset(n_sample: int = 8) -> list[list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS", split=f"train[:{n_sample}]")
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["url"]},
                    {"type": "text", "text": "为这张图片生成一个标题"},
                ],
            },
            {"role": "assistant", "content": sample["caption"]},
        ]
        for sample in dataset
    ]

dataset = prepare_dataset()

# 将数据集处理为张量
text = model.processor.apply_chat_template(dataset, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(dataset)
inputs = model.processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

# 然后只需一行代码运行校准过程
model.quantize(calib_data=inputs, quant_config=quant_config, quantizer_cls=Qwen2VLAwqQuantizer)

# 保存模型
model.model.config.use_cache = model.model.generation_config.use_cache = True
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
```

### 另一个自定义量化器 (MiniCPM3示例)

这里我们介绍来自OpenBMB的MiniCPM团队的另一个自定义量化器。我们只
修改权重剪裁机制以使量化工作。

```python
import torch
from transformers import AutoTokenizer

from awq import AutoAWQForCausalLM
from awq.quantize.quantizer import AwqQuantizer, clear_memory

class CPM3AwqQuantizer(AwqQuantizer):
    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # 计算输入特征步长（最小为1）
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]

        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # 防止OOM
        if org_w_shape[0] % oc_batch_size != 0:
            oc_batch_size = org_w_shape[0]
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

model_path = 'openbmb/MiniCPM3-4B'
quant_path = 'minicpm3-4b-awq'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM" }

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 量化
model.quantize(tokenizer, quant_config=quant_config, quantizer_cls=CPM3AwqQuantizer)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'模型已量化并保存在 "{quant_path}"')
```

## 基本推理

### GPU推理
要运行推理，通常需要设置`fuse_layers=True`以在AutoAWQ中获得声称的加速。
此外，考虑设置`max_seq_len`（默认：2048），因为这将是模型可以容纳的最大上下文。

注意事项：

- 您可以指定`use_exllama_v2=True`以在推理期间启用ExLlamaV2内核。

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

quant_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# 加载模型
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 将提示转换为令牌
prompt_template = "[INST] {prompt} [/INST]"

prompt = "你站在地球表面。 "\
        "你向南走一英里，向西走一英里，向北走一英里。 "\
        "你最终正好回到起点。你在哪儿？"

tokens = tokenizer(
    prompt_template.format(prompt=prompt),
    return_tensors='pt'
).input_ids.cuda()

# 生成输出
generation_output = model.generate(
    tokens,
    streamer=streamer,
    max_new_tokens=512
)
```

### CPU推理
要使用CPU运行推理，应指定`use_ipex=True`。ipex是CPU的后端，包括操作符的内核。ipex是intel_extension_for_pytorch包。

```python
from awq import AutoAWQForCausalLM

quant_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
# 加载模型
model = AutoAWQForCausalLM.from_quantized(quant_path, use_ipex=True)
```

### Transformers

您也可以使用AutoModelForCausalLM加载AWQ模型，只需确保已安装AutoAWQ。
请注意，并非所有模型在从transformers加载时都具有融合模块。
查看更多[文档](https://huggingface.co/docs/transformers/main/en/quantization/awq)。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 注意：必须从PR安装直到合并
# pip install --upgrade git+https://github.com/younesbelkada/transformers.git@add-awq
model_id = "casperhansen/mistral-7b-instruct-v0.1-awq"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 将提示转换为令牌
text = "[INST] 使用Huggingface transformers库的基本步骤是什么？ [/INST]"

tokens = tokenizer(
    text,
    return_tensors='pt'
).input_ids.cuda()

# 生成输出
generation_output = model.generate(
    tokens,
    streamer=streamer,
    max_new_tokens=512
)
```

### vLLM

您也可以在[vLLM](https://github.com/vllm-project/vllm)中加载AWQ模型。

```python
import asyncio
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

model_path = "casperhansen/mixtral-instruct-awq"

# 提示
prompt = "你站在地球表面。 "\
         "你向南走一英里，向西走一英里，向北走一英里。 "\
         "你最终正好回到起点。你在哪儿？",

prompt_template = "[INST] {prompt} [/INST]"

# 采样参数
sampling_params = SamplingParams(
    repetition_penalty=1.1,
    temperature=0.8,
    max_tokens=512
)

# 分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 用于流式传输的异步引擎参数
engine_args = AsyncEngineArgs(
    model=model_path,
    quantization="awq",
    dtype="float16",
    max_model_len=512,
    enforce_eager=True,
    disable_log_requests=True,
    disable_log_stats=True,
)

async def generate(model: AsyncLLMEngine, tokenizer: PreTrainedTokenizer):
    tokens = tokenizer(prompt_template.format(prompt=prompt)).input_ids

    outputs = model.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=1,
        prompt_token_ids=tokens,
    )

    print("\n** 开始生成！\n")
    last_index = 0

    async for output in outputs:
        print(output.outputs[0].text[last_index:], end="", flush=True)
        last_index = len(output.outputs[0].text)

    print("\n\n** 完成生成！\n")

if __name__ == '__main__':
    model = AsyncLLMEngine.from_engine_args(engine_args)
    asyncio.run(generate(model, tokenizer))
```

### LLaVa（多模态）

AutoAWQ还支持LLaVa模型。您只需加载一个
AutoProcessor来处理提示和图像，为AWQ模型生成输入。

```python
import torch
import requests
from PIL import Image
from awq import AutoAWQForCausalLM
from transformers import AutoProcessor, TextStreamer

# 加载模型
quant_path = "casperhansen/llama3-llava-next-8b-awq"
model = AutoAWQForCausalLM.from_quantized(quant_path)
processor = AutoProcessor.from_pretrained(quant_path)
streamer = TextStreamer(processor, skip_prompt=True)

# 定义提示
prompt = """\
\system
回答问题。\
\user
<image>
这张图片显示了什么？\
\assistant
"""

# 定义图像
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

# 加载输入
inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

generation_output = model.generate(
    **inputs,
    max_new_tokens=512,
    streamer=streamer
)
```

### Qwen2 VL

下面是关于如何运行Qwen2 VL推理的示例。

```python
from awq import AutoAWQForCausalLM
from awq.utils.qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, TextStreamer

# 加载模型
quant_path = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
model = AutoAWQForCausalLM.from_quantized(quant_path)
processor = AutoProcessor.from_pretrained(quant_path)
streamer = TextStreamer(processor, skip_prompt=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "描述这张图片。"},
        ],
    }
]

# 加载输入
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generation_output = model.generate(
    **inputs,
    max_new_tokens=512,
    streamer=streamer
)
```