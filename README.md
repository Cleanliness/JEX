# JEX
JAX-based LLM inference

# Timeline/benchmarking
Hardware:
- RTX 3060 12GB
- 16GB RAM

Benchmarking parameters:
- 1 batch size
- 256 sequence length
- Gemma 1 2B
- Mean time over 100 runs

| Version | Tokens/s | Notes |
| --- | --- | --- |
| v0 | 59.8865 | Initial impl |
| v0.1 | 61.96 | Added KV cache |

To beat:
- Deepmind's Flax impl: OOM
- HF transformers: TODO
- KerasNLP (JAX backend): TODO
- exllamav2: TODO
    - This might be unfair since it doesn't do bf16
- vLLM: TODO

# Implemented
Gemma 1
- [x] RMSNorm+1
- [x] Multi-query attention 
- [x] Multi-head attention
- [x] Causal attention
- [x] Positional embedding
- [x] Tokenizer
- [x] Correctness
- [x] Masking
- [x] KV Cache

# TODO

Gemma 2 
- [ ] Sliding window attention
- [ ] Attention soft capping (-30, 30)
- [ ] Logit soft capping (-50, 50)
- [ ] double layernorms

Llama 2/3
- [ ] Regular RMSnorm

QOL
- [ ] clean up
    - [x] move benchmarks/tests into other scripts
    - [x] simple inference example
- [ ] model API
- [ ] choose inference dtype

Later
- [ ] FlashAttention (CUDA/Triton kernels)
- [ ] Weight Quantization
- [ ] Grouped Query attention
- [ ] More exotic quantization schemes (HQQ)
