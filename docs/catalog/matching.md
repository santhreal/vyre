# `matching` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

5 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-libs::matching::aho_corasick` | `libs` | `scan` | 0:haystack:ReadOnly:U32<br>1:transitions:ReadOnly:U32<br>2:accept:ReadOnly:U32<br>3:matches:ReadWrite:U32 | `matching` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::matching::aho_corasick |
| `vyre-libs::matching::compact_hits` | `libs` | `matching` | 0:out_hits:ReadOnly:U32<br>1:out_cursor:ReadOnly:U32<br>2:hit_buffer_live_length:ReadWrite:U32 | `matching` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::matching::compact_hits |
| `vyre-libs::matching::cooperative_dfa` | `libs` | `matching` | 0:input:ReadOnly:U32<br>1:transitions:ReadOnly:U32<br>2:accept_mask:ReadOnly:U32<br>3:matches:ReadWrite:U32 | `matching` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::matching::cooperative_dfa |
| `vyre-libs::matching::emit_hit` | `libs` | `matching` | 0:rule_id:ReadOnly:U32<br>1:file_id:ReadOnly:U32<br>2:span_start:ReadOnly:U32<br>3:span_len:ReadOnly:U32<br>4:out_hits:ReadWrite:U32<br>5:out_cursor:ReadWrite:U32<br>6:hit_buffer_overflow_count:ReadWrite:U32 | `matching` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-libs::matching::emit_hit |
| `vyre-primitives::matching::bracket_match` | `intrinsic` | `matching` | 0:kinds:ReadOnly:U32<br>1:stack:ReadWrite:U32<br>2:match_pairs:ReadWrite:U32 | `matching`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::matching::bracket_match |
