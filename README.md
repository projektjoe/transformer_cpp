# transformer_cpp

## Overview

LLaMA2 model inference using pure C++ STD only. This project aims to provide a transparent, debuggable approach to understanding LLMs through a sequential, readable codebase (since I found parallelism to be difficult to interpret).

## Motivation

The primary goal is to:
- Deeply understand LLM internals
- Facilitate research, optimization and performance improvements.
- Create a clean, step-by-step implementation for learning and debugging

## Installation

Start by downloading the Llama2 model https://huggingface.co/meta-llama/Llama-2-7b

### Export Model and Tokenizer
1. Export the tokenizer to binary format:
```bash
python utils/export_tokenizer.py --input /path/to/tokenizer.model --output /path/to/tokenizer.bin
```

2. Export the quantized model:
```bash
python utils/export_model.py --model_path /path/to/llama/weights/folder --output /path/to/model_q80.bin
```

### Inference

```bash
mkdir build
cd build
cmake .. && make
./transformer_cpp <prompt> <tokenizer_path> <model_path> 
```

Concrete example:
```
./transformer_cpp hi ../llama2_weights/tokenizer.bin ../llama2_weights/llama2_q80.bin
```

## Credits

- Tokenizer and weight loading adapted from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c)
- Core implementation written from scratch

## Roadmap

### TODO
- [x] Quantization
- [x] Greedy sampling
- [x] KV caching

- [ ] Advanced sampling techniques
  - Top-k sampling
  - Top-n sampling
- [ ] Performance optimizations
  - Speculative decoding
  - Parallelizing attention heads
  - Optimized matrix multiplication
  - Parallel token processing
- [ ] Advanced features
  - Backward propagation
  - CUDA matrix multiplication kernel

## Getting Started

### Prerequisites
- C++ compiler with C++17 support
- CMake

### Building the Project
```bash
git clone https://github.com/yourusername/transformer_cpp.git
cd transformer_cpp
mkdir build && cd build
cmake ..
make
```

## Contributing

Contributions, issues, and feature requests are welcome!
