# vyre-libs Feature Matrix

Applies to Vyre 0.7.2.

`vyre-libs` uses feature flags for footprint control, not for product
shape. The matrix has three layers:

| Layer | Features | Rule |
| --- | --- | --- |
| Defaults | `default` | Load-bearing consumer surface. CI must keep this compiling without parser-specific feature opt-ins. |
| Aggregates | `math`, `nn`, `matching`, `crypto`, `parsing` | Compatibility rollups. Add new granular features here only when the aggregate contract should include them. |
| Granular dialects | `math-linalg`, `matching-nfa`, `c-parser`, etc. | Smallest selectable units. Benches/tests must declare `required-features` for any granular feature they need. |

Operational features:

| Feature | Use |
| --- | --- |
| `bench` | Benchmark-only helper APIs and layout introspection. Not part of normal consumer builds. |
| `rule` | Rule-builder surface. |
| `vyre_wgpu`, `vyre_driver_wgpu` | Compatibility aliases for optional wgpu integration. |

## matching features and their source homes

The `matching-*` features gate modules under `vyre-libs/src/scan/`. The
prefix is historical: it names the dialect, not a directory. There is no
`vyre-libs/src/matching/`. The features are not renamed, because a rename is
a semver break for every consumer that lists them.

| Feature | Implies | Source home |
| --- | --- | --- |
| `matching-dfa` | `cpu-parity` | `scan/dfa/`, `scan/classic_ac/` |
| `matching-substring` | `cpu-parity`, `matching-dfa` | `scan/substring/` |
| `matching-nfa` | `matching-dfa` | `scan/nfa/`, `scan/scan_program.rs` |
| `matching-regex` | `matching-nfa`, `dep:regex-syntax` | `scan/regex_compile.rs`, `scan/regex_dfa.rs`, `scan/regex_region_admission.rs`, `scan/regex_anchored_window.rs`, `scan/fused_region_evidence.rs` |
| `matching` | `matching-substring`, `matching-dfa` | Aggregate rollup with no source of its own. |

`scan/builders.rs`, `scan/hit_buffer.rs`, and `scan/post_process.rs` are
ungated and build under every configuration.

`scan/classic_ac/bounded_ranges/` owns the Aho-Corasick transition step, the
output-link span, the region-search prologue, the candidate end gate, and the
shared DFA buffer declarations. Every other builder under `scan/` projects
from it. `vyre-libs/tests/scan_ac_transition_walk_single_owner.rs` fails if a
second copy of that walk appears.

CI policy:

- Default `cargo check -p vyre-libs` covers the consumer surface.
- Parser tests must opt in with `c-parser`, `go-parser`, or `python-parser`.
- Bench targets must keep `required-features` in `vyre-libs/Cargo.toml`; do not rely on workspace-wide defaults.
- New granular features need one line in this table and one focused check command in the owning PR.
