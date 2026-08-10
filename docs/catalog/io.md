# `io` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

2 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `io.dma_from_nvme` | `runtime` | `io` | (fd:i32, offset:u64, length:u64) -> (handle:GpuBufferHandle) | `default` | reference=false inputs=false expected=false tolerance=0 ULP | cuda:experimental<br>reference:not_applicable<br>wgpu:experimental | none declared | leaf |
| `io.write_back_to_nvme` | `runtime` | `io` | (handle:GpuBufferHandle, fd:i32, offset:u64) -> () | `default` | reference=false inputs=false expected=false tolerance=0 ULP | cuda:experimental<br>reference:not_applicable<br>wgpu:experimental | none declared | leaf |
