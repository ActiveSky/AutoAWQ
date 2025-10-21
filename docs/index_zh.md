# AutoAWQ

AutoAWQ 将易用性和快速推理速度结合在一个包中。在以下文档中，您将学习如何量化模型并运行推理。

推理速度示例（RTX 4090、Ryzen 9 7950X、64 个令牌）：

- Vicuna 7B（GEMV 内核）：198.848 令牌/秒
- Mistral 7B（GEMM 内核）：156.317 令牌/秒
- Mistral 7B（ExLlamaV2 内核）：188.865 令牌/秒
- Mixtral 46.7B（GEMM 内核）：93 令牌/秒（2x 4090）

## 安装注意事项

- 安装：`pip install autoawq`。
- 您的 torch 版本必须与构建版本匹配，即不能使用 torch 2.0.1 来配合用 2.2.0 构建的 wheel。
- 对于 AMD GPU，推理将通过 ExLlamaV2 内核运行而不使用融合层。您需要传递以下参数来在 AMD GPU 上运行：
    ```python
    model = AutoAWQForCausalLM.from_quantized(
        ...,
        fuse_layers=False,
        use_exllama_v2=True
    )
    ```
- 对于 CPU 设备，您应该使用 `pip install intel_extension_for_pytorch` 安装 intel_extension_for_pytorch。并且需要最新版本的 torch，因为“intel_extension_for_pytorch (IPEX)”是用最新版本的 torch 构建的（现在 IPEX 2.4 是用 torch 2.4 构建的）。如果从源代码构建 IPEX，则需要确保 torch 版本的一致性。并且应该为 CPU 设备使用“use_ipex=True”。
    ```python
    model = AutoAWQForCausalLM.from_quantized(
        ...,
        use_ipex=True
    )
    ```

## 支持的模型

我们支持现代大语言模型。您可以在 `awq/models` 中找到支持的 Huggingface `model_types` 列表。
