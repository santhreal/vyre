# Vyre Consumer Showcase

## First public integration: Keyhog

The first public Vyre integration is Keyhog.
Use it as a practical example of a full workload flowing through:

- C/Rust parser pipeline output
- Graph/dataflow composition in `vyre-libs`
- Conformance-checked execution paths
- CUDA-backed and WGPU-backed runners
- Conformance evidence and regression reporting

Repo: https://github.com/santhsecurity/keyhog

Use this path if you want a real end-to-end reference for:

- parser output shaping into Vyre IR,
- conformance gate integration,
- and production-oriented deployment patterns for GPU-first analysis workflows.
