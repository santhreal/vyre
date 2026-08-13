# vyre Architecture

The architecture documentation lives at [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

That file is the single source. It covers the workspace boundaries, the
operation categories (A compositions, C hardware intrinsics, banned B
interpreters), the target operation-crate structure, the region chain
invariant, the artifact pipeline, the megakernel four-owner boundary, the
two-layer optimization architecture, and the conformance and release
evidence rules that used to live here.

For the shorter tour, start with the [`README.md`](README.md). For the reasoning
behind the design rather than its shape, read [`THESIS.md`](THESIS.md).
