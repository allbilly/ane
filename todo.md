# Tiny encoder benchmark

Current task complete: reproduce the saved macOS tiny encoder timing on Asahi
and update the README comparison.

| Encoder + CPU cross-K/V | Median |
|---|---:|
| macOS ANE + Accelerate — saved | 17.17 ms |
| macOS ANE + OpenBLAS — saved | 28.65 ms |
| Asahi ANE + OpenBLAS — latest rerun | **32.13 ms** |

Full 1,500-position encoder context, zero-mel input and four workers. Timing
includes input staging, ANE execution and CPU cross-K/V. Loading, compilation
and decoding are excluded. The Asahi rerun used two persistent contexts, two
warmups and five measured calls per context; measured range: 31.98–32.48 ms.
The earlier 32.10 ms result is consistent with this rerun.

- [x] Reconstruct the existing dumped kernel from the matching tiny checkpoint.
- [x] Run the same native encoder timing harness on Asahi with OpenBLAS.
- [x] Match the saved macOS zero-mel encoder output byte for byte on all 23
  separate replay checks: three warmups and twenty measured calls.
- [x] Record default-poll driver profiles and update the README benchmark.

[Asahi timing/output receipt](whisper/results/pr3905-asahi-tiny-profiled-20261008.json)
· [saved macOS timings](whisper/results/pr3905-m1-20261008/host-encode-only.json)
· [README table](README.md)

The older decoder, paired-control, Qwen and broader fixture investigations have
been removed from this active checklist. Their existing code, results and docs
are retained. Speech numerical accuracy is a separate result, described beside
the README table; it does not add prerequisites to this encoder timing task.
