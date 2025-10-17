# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoAWQ implements the Activation-aware Weight Quantization (AWQ) algorithm for 4-bit quantization of Large Language Models. The project is officially deprecated but maintained for compatibility with existing quantized models.

## Common Commands

### Model Quantization
```bash
# Basic quantization (see examples/quantize.py for full example)
python examples/quantize.py

# Custom quantization with specific config
python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained('model_path')
tokenizer = AutoTokenizer.from_pretrained('model_path')
quant_config = {'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized('quantized_path')
"
```

### Model Inference
```bash
# Basic inference (see examples/generate.py for full example)
python examples/generate.py

# Load quantized model
python -c "
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_quantized('quantized_path', fuse_layers=True)
"
```

### Benchmarking
```bash
# Benchmark model performance
python examples/benchmark.py --model_path <model_path> --batch_size 1

# Benchmark with specific generator
python examples/benchmark.py --model_path <model_path> --generator hf --batch_size 4

# Benchmark pretrained (FP16) model
python examples/benchmark.py --model_path <model_path> --pretrained
```

### Testing
```bash
# Run quantization tests
python tests/test_quantization.py

# Run Intel CPU/IPEX tests
python tests/test_ipex_cpu.py

# Run dequantization tests
python tests/test_dequantization.py
```

### Installation
```bash
# Default installation (Triton kernels only)
pip install -e .

# With optimized kernels
pip install -e .[kernels]

# Intel CPU optimization
pip install -e .[cpu]

# Development dependencies
pip install -e .[dev]
```

## Architecture Overview

### Core Components

**AwqQuantizer** (`awq/quantize/quantizer.py`): Main quantization engine that implements the AWQ algorithm
- Performs activation-aware weight quantization through 4-step process
- Handles scaling factor computation, clipping optimization, and weight quantization
- Supports memory-efficient chunked computation for large models

**BaseAWQForCausalLM** (`awq/models/base.py`): Base class for all AWQ model implementations
- Provides unified interface for loading pretrained and quantized models
- Handles model packing, saving, and basic inference operations
- Manages device placement and memory optimization

**Model Implementations** (`awq/models/`): Model-specific implementations for 35+ model families
- Each model family (Llama, Mistral, Qwen, etc.) has its own implementation class
- Handles model architecture quirks and layer naming conventions
- Implements scaling layer identification and input feature extraction

**Quantization Kernels** (`awq/modules/linear.py`): Multiple quantization backends
- GEMM: General Matrix-Matrix Multiplication (better for batch sizes > 1)
- GEMV: General Matrix-Vector Multiplication (faster for batch size = 1)
- Marlin: Optimized kernel for specific hardware
- ExLlama/ExLlamaV2: High-performance inference kernels

### Quantization Process

1. **Initialization**: Load model, prepare calibration dataset, capture layer inputs using Catcher pattern
2. **Input Feature Collection**: Register forward hooks to capture activations for each linear layer
3. **Scale Factor Optimization**: Grid search for optimal per-channel scaling factors that minimize quantization error
4. **Clipping Optimization**: Optional weight clipping to further reduce quantization error
5. **Weight Quantization**: Apply 4-bit quantization with computed scales and optional zero points

### Memory Management

The codebase uses extensive memory optimization techniques:
- Chunked computation to avoid OOM during large tensor operations
- Explicit memory cleanup with `clear_memory()` calls
- Device-aware tensor placement for multi-GPU setups
- Batching strategies for calibration data processing

### Multi-Model Support

AutoAWQ supports 35+ model families through specialized implementations:
- **Text Models**: Llama, Mistral, Mixtral, Qwen, DeepSeek, Gemma, etc.
- **Vision-Language Models**: LLaVA, Qwen2-VL, MiniCPM-VL
- **Code Models**: StarCoder2, CodeLlama, DeepSeek-Coder
- **MoE Models**: Mixtral, Qwen3-MoE, DeepSeek-V2/V3

Each implementation handles:
- Layer naming conventions and module structure
- Scaling layer identification for optimal quantization
- Model-specific forward pass patterns
- Special handling for attention mechanisms and position embeddings

## Development Notes

### Model Types and Versions

AWQ supports multiple quantization versions:
- **GEMM**: Better for batch sizes > 1, larger contexts
- **GEMV**: Faster for batch size = 1, smaller contexts
- **GEMV Fast**: Optimized GEMV implementation
- **Marlin**: Hardware-optimized kernel

### Configuration Parameters

Key quantization parameters:
- `w_bit`: Quantization bits (typically 4)
- `q_group_size`: Group size for quantization (typically 128)
- `zero_point`: Whether to use zero-point quantization
- `version`: Quantization kernel version
- `max_calib_samples`: Number of calibration samples (default: 128)
- `max_calib_seq_len`: Maximum calibration sequence length (default: 512)

### Memory Efficiency Features

- `max_chunk_memory`: Controls memory usage during computation (default: 1GB)
- `n_parallel_calib_samples`: Parallel processing for calibration data
- `apply_clip`: Optional weight clipping optimization
- `export_compatible`: Generate weights compatible with export formats

### Model-Specific Considerations

- **Mixtral**: Requires special MoE expert handling
- **DeepSeek V2/V3**: Complex attention patterns need custom scaling
- **Vision Models**: Additional processor and image handling
- **CPU/IPEX**: Special kernels for Intel hardware optimization