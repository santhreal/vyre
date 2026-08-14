# `opt` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

1 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-primitives::opt::homotopy_euler_predictor` | `intrinsic` | `opt` | 0:x_curr:ReadOnly:U32<br>1:v:ReadOnly:U32<br>2:dt_scaled:ReadOnly:U32<br>3:x_pred:ReadWrite:U32 | `opt`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::opt::homotopy_euler_predictor |
