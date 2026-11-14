# vyre-foundation-fuzz

LibFuzzer targets exercising untrusted input parsers and decoders in `vyre-foundation`.

## Purpose

- Fuzzes `VIR0` wire decoding (`decoder`, `program_wire`) and registry TOML parsing (`registry_toml`) under `cargo fuzz`.
- Verifies that malformed byte streams return structured errors without panicking or reading out of bounds.

## Boundaries

- Standalone fuzz workspace excluded from the root Cargo workspace (`crates/vyre-foundation/fuzz`).
- Depends on `vyre-foundation` and `libfuzzer-sys` only.
- Contains fuzz target binaries only; reusable fixtures and property strategies belong in `vyre-test-support`.
- No workspace crate may depend on `vyre-foundation-fuzz`.
