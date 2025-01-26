# transformer_cpp

## Overview

LLaMA2 model inference using pure C++ STD only. This project aims to provide a transparent, debuggable approach to understanding LLMs through a sequential, readable codebase (since I found parallelism to be difficult to interpret).

## Motivation

The primary goal is to:
- Deeply understand LLM internals
- Facilitate research, optimization and performance improvements.
- Create a clean, step-by-step implementation for learning and debugging

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

## License

MIT License

Copyright (c) 2025 projectjoe
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.