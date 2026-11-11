# vyre-graphics-app

Independent downstream interactive graphical application consumer built on the Vyre public facade.

## Purpose

`vyre-graphics-app` exercises multi-stage 2D rendering workflows against the public compiler seam:
- Viewport bounding-box culling
- Analytical vector line segment rasterization
- Glyph instance accumulation from texture atlases
- Scissor rectangle clipping
- Retained dirty-region patch blitting

Every frame is constructed as one validated `ProgramGraph` through `vyre_libs::graph_compositions::build_interactive_graphics_pipeline`. Compilation and submission proceed strictly through `vyre::compiler::compile` and `vyre::ArtifactSession`.

## Dependencies

The package depends only on published consumer SDK crates:
- `vyre`: frontend intermediate representation, request validation, compiler facade, and artifact admission
- `vyre-libs`: domain compositions including `visual`, `math-scan`, `reduce`, and `graph`

Zero production dependencies exist on internal engine crates or workspace test fixtures.

## Running

Run the interactive graphics benchmark harness:

```bash
cargo run --manifest-path consumers/vyre-graphics-app/Cargo.toml -- 100
```

## Testing

Execute the test suite:

```bash
cargo test --manifest-path consumers/vyre-graphics-app/Cargo.toml
```

The test suite proves:
- `public_seam_isolation.rs`: dynamically validates against `docs/CRATE_OWNERSHIP.toml` that no forbidden `internal-engine` or `private-test-support` dependencies exist in consumer manifests, and verifies that graph nodes use only standard intermediate representation opcodes.
- `exact_pixel_parity.rs`: compares rendered rasterization, blend, text, and scissor clip buffers against independent reference execution.
- `workload_matrix.rs`: validates cold startup, rapid window resize, burst pointer events, retained state reset, and simulated device recovery.
