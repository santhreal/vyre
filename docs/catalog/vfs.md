# `vfs` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

1 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-primitives::vfs::resolve` | `intrinsic` | `vfs` | 0:include_hashes:ReadOnly:U32<br>1:out_file_buffers:ReadWrite:U32<br>2:global_dma_pool:ReadOnly:U32 | `parsing`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::vfs::resolve |
