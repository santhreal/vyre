# vyre-model-compiler

Independent downstream neural model compiler package consuming the Vyre public facade.

## Purpose

`vyre-model-compiler` translates frontier transformer architecture definitions and checkpoint parameter manifests into domain-neutral intermediate representation:
- Autoregressive prefill and decode execution envelopes
- Multi-head and grouped-query attention mechanisms
- Mixture-of-experts routing and feed-forward blocks
- Parameter tensor validation and typed resource datasets

Whole-model pipelines are lowered to validated `ProgramGraph` instances without domain vocabulary, schedule templates, or target hints reaching compiler internals.

## Dependencies

The package depends only on published consumer SDK crates:
- `vyre`: frontend intermediate representation, request validation, compiler facade, and resource admission
- `vyre-libs`: neural network composition dialects including `nn-activation`, `nn-linear`, `nn-norm`, `nn-attention`, `nn-moe`, and `nn-inference`

Zero production dependencies exist on internal engine crates.

## Running

Run the model compiler driver:

```bash
cargo run --manifest-path consumers/vyre-model-compiler/Cargo.toml
```

## Testing

Execute the test suite:

```bash
cargo test --manifest-path consumers/vyre-model-compiler/Cargo.toml
```

The test suite proves:
- `dependency_direction.rs`: dynamically parses `docs/CRATE_OWNERSHIP.toml` and confirms zero forbidden `internal-engine` or `private-test-support` dependencies exist in the manifest, and verifies dependency direction via `cargo metadata`.
- `seam_audit.rs`: verifies that all named model architecture families compile through the neutral compiler seam.
- `prefill_decode_pipeline.rs`: tests multi-stage prefill and decode pipeline generation, checkpoint parameter validation, and resource dataset ingestion.
- `absence_proofs.rs`: verifies the complete absence of domain vocabulary or model-specific identifiers in emitted artifacts.
- `named_configs_coverage.rs`: validates coverage across all declared architecture configurations.
