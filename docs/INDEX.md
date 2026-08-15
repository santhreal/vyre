<!-- Generated from docs/DOCS.toml by scripts/docs_manifest.py. Do not edit. -->
# Documentation Authority and Lifecycle

Source: [`docs/DOCS.toml`](DOCS.toml).

Each active page declares its audience, owner, authority source, kind, and
generation mode. Generated pages also declare the generator. Superseded and
archived pages remain lifecycle evidence and are excluded from navigation.

## Documentation owners

| Owner | Authority |
| --- | --- |
| `architecture` | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| `benchmark` | [`optimization/BENCH_TARGETS.toml`](optimization/BENCH_TARGETS.toml) |
| `docs-governance` | [`DOCS.toml`](DOCS.toml) |
| `foundation` | [`../vyre-foundation/src/lib.rs`](../vyre-foundation/src/lib.rs) |
| `historical` | [`DOCS.toml`](DOCS.toml) |
| `operation-registry` | [`../vyre-foundation/src/operation.rs`](../vyre-foundation/src/operation.rs) |
| `optimization` | [`optimization/OWNERSHIP.toml`](optimization/OWNERSHIP.toml) |
| `public-facade` | [`../vyre/src/lib.rs`](../vyre/src/lib.rs) |
| `release-tooling` | [`../scripts/release_docs.py`](../scripts/release_docs.py) |
| `runtime` | [`../vyre-runtime/src/lib.rs`](../vyre-runtime/src/lib.rs) |
| `safetensors-adapter` | [`../vyre-safetensors/src/lib.rs`](../vyre-safetensors/src/lib.rs) |
| `testing` | [`testing/TESTING.toml`](testing/TESTING.toml) |

## Cargo-derived workspace facts

- Workspace packages: 35.
- Shipped library, binary, and example targets: 74.
- Source: `cargo metadata --no-deps --format-version 1`.

## Lifecycle counts

- current: 1.
- generated: 1.
- superseded: 0.
- archived: 0.

## Pages

| Status | Audience | Owner | Kind | Page | Authority | Generation |
| --- | --- | --- | --- | --- | --- | --- |
| `current` | `extension` | `architecture` | `ownership` | `ARCHITECTURE.md` | self | manual |
| `generated` | `contributor` | `docs-governance` | `governance` | `INDEX.md` | [DOCS.toml](DOCS.toml) | generated: [../scripts/docs_manifest.py](../scripts/docs_manifest.py) |
