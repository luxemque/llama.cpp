# Parallel RPC loading

This branch of [llama.cpp](https://github.com/ggml-org/llama.cpp) implements a set of changes to the RPC loading path that cut cold model load time on multi-node clusters by ~10×.

Documentation — quick start, env vars, the full speedup table, the diagnostic write-up, and the archive of per-run artifacts — lives in the companion repo:

> **https://github.com/luxemque/turboload.llama.cpp**
