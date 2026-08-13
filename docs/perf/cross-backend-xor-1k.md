# cross-backend comparison

Produced by `cargo_full run --bin xtask -- bench-crossback <program>`. ms
values are CPU-reference oracle wall-clock per call. GPU release
evidence comes from the dedicated CUDA/WGPU benchmark suites.

| program | wgpu | spirv | secondary_text | native_module | cpu-ref |
|---------|------|-------|-----|-------|---------|
| `xor-1k` | n/a | n/a | n/a | n/a | 0.012 |
