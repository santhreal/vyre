# `mem` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

2 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mem.unmap` | `runtime` | `mem` | (handle:GpuBufferHandle) -> () | `default` | reference=false inputs=false expected=false tolerance=0 ULP | cuda:experimental<br>reference:not_applicable<br>wgpu:experimental | none declared | leaf |
| `mem.zerocopy_map` | `runtime` | `mem` | (fd:i32) -> (handle:GpuBufferHandle) | `default` | reference=false inputs=false expected=false tolerance=0 ULP | cuda:experimental<br>reference:not_applicable<br>wgpu:experimental | none declared | leaf |
