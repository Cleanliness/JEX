# JEX
JAX-based LLM inference

# Timeline/benchmarking
Hardware:
- RTX 3060 12GB
- 16GB RAM

All benchmarks run on Gemma 2B

To beat:
- Deepmind's Flax impl: OOM
- HF transformers: TODO
- KerasNLP (JAX backend): TODO
- exllamav2: TODO
    - This might be unfair since it doesn't do bf16
- vLLM: TODO

History:
- v0: 59.8865 tokens/second

# TODO
gemma 2B
- [x] Layernorm
- [x] Multi-query attention 
- [x] Multi-head attention
- [x] Causal attention
- [x] Positional embedding
- [x] Tokenizer
- [ ] KV Cache

Later
- [ ] FlashAttention (CUDA and Triton kernels)
- [ ] Weight Quantization
