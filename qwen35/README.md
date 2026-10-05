# Mirai Qwen3.5 on Asahi

Work in progress: the loader and packed CPU kernels support the exact
[`trymirai/Qwen3.5-0.8B-M`](https://huggingface.co/trymirai/Qwen3.5-0.8B-M)
checkpoint at revision `c12202e4c764e559960827761566aaa1fd15a87a`.
Model weights remain outside the repository in `~/.cache/ane-qwen35/model/`.
Place `config.json`, `model.safetensors` and `tokenizer.json` there, or select
a directory with `--model`.

```sh
./qwen35/first-run.sh inspect --check-hashes
./qwen35/first-run.sh generate --prompt 'What is 2 + 2?' --max-tokens 32
./qwen35/first-run.sh generate --precision bf16 --prompt 'Hello'
./qwen35/first-run.sh bench --raw --prompt 'Hello' --max-tokens 32 --output /tmp/qwen35.json
./qwen35/.venv/bin/python -m unittest qwen35.tests -v
```

Mirai M uses asymmetric 4-bit weights, groups of 32, BF16 scales, packed
4-bit zero points and signed 32-element Hadamard transforms. This differs
from GGUF Q4_0 and Mirai S's trellis codec. Body matrices store scales and
zero points in group/output order; embeddings use output/group order.
The loader retains the original packed weights and normalizes only those
small parameter tables. Native matvec accumulates in FP32. DeltaNet's
convolution and recurrent states use FP32. `--precision bf16` reproduces
the vendor's BF16 activation boundaries; FP32 is the default.

There are 18 recurrent layers and six attention layers, width 1024 and
FFN width 3584. Attention uses eight query heads, two KV heads and partial
64-dimensional RoPE. The tokenizer and chat template come from the exact
checkpoint; generation currently uses greedy selection. Prompt ingestion
is sequential, and earlier prompt tokens skip the vocabulary projection.

The independent NumPy checks cover distinct group/output parameter values,
both sides of the transform, tied embedding readout, and an explicit
recurrent-state update. They do not establish full-model parity. An optional
Uzu CPU oracle can be built with `tools/build_reference.py`; its instrumentation
exports decoder logits without changing the decoder or its kernels.
Full-model validation, ANE integration and performance profiling remain in
progress. [Source revisions and hashes](provenance/sources.json) are retained.
