# `text` operations

This page is generated from `docs/generated/OP_SCHEMA.json`. The JSON schema is the authority. Regenerate this view with `cargo_full run --bin xtask -- catalog`.

6 operations are registered in this subsystem.

| operation | tier | category | signature | features | oracle | backend support | laws | composition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vyre-primitives::text::byte_histogram_256` | `primitive` | `text` | 0:bytes:ReadOnly:U32<br>1:histogram:ReadWrite:U32 | `text`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::text::byte_histogram_256 |
| `vyre-primitives::text::char_class` | `primitive` | `text` | 0:source:ReadOnly:U32<br>1:table:ReadOnly:U32<br>2:classified:ReadWrite:U32 | `text`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::text::char_class |
| `vyre-primitives::text::encoding_classify` | `primitive` | `text` | 0:histogram:ReadOnly:U32<br>1:encoding:ReadWrite:U32 | `text`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::text::encoding_classify<br>&nbsp;&nbsp;vyre-primitives::reduce::range_counts_u32<br>&nbsp;&nbsp;vyre-primitives::text::utf8_shape_counts |
| `vyre-primitives::text::line_index` | `primitive` | `text` | 0:source:ReadOnly:U32<br>1:__lines_line_start_flags:ReadWrite:U32<br>2:lines:ReadWrite:U32<br>0:__lines_guarded_scan_a:Workgroup:U32<br>0:__lines_guarded_scan_b:Workgroup:U32 | `text`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre.program.root (internal)<br>&nbsp;&nbsp;vyre-primitives::text::line_index::line_start_flags (internal)<br>&nbsp;&nbsp;vyre-primitives::reduce::multi_block_prefix_scan_inclusive_sum::guarded_single_block (internal) |
| `vyre-primitives::text::utf8_shape_counts` | `primitive` | `text` | 0:histogram:ReadOnly:U32<br>1:out:ReadWrite:U32 | `text`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::text::utf8_shape_counts |
| `vyre-primitives::text::utf8_validate` | `primitive` | `text` | 0:source:ReadOnly:U32<br>1:classes:ReadWrite:U32 | `text`<br>`inventory-registry` | reference=true inputs=true expected=true tolerance=0 ULP | cuda:supported<br>reference:supported<br>wgpu:supported | none declared | vyre-primitives::text::utf8_validate |
