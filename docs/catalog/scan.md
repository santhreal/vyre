# `scan` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

1 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-libs::scan::substring_search` | `libs` | `scan` | 0:haystack:ReadOnly:U32<br>1:needle:ReadOnly:U32<br>2:matches:ReadWrite:U32 | `matching` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::scan::substring_search |
