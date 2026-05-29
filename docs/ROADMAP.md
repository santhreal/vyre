# vyre Roadmap (current priorities)

This roadmap tracks what is open today. It is a working contract for this codebase, not a historical changelog.

## Active (current cycle)

- **Release path stability**
  - Keep CUDA as the primary release backend path and WGPU as the portable GPU path.
  - Keep `vyre-runtime`, `vyre-driver-cuda`, and `vyre-driver-wgpu` execution claims evidence-backed through conformance and adversarial suites.

- **Organization and separation cleanup**
  - Maintain single-authority rules for foundational ops and keep higher-tier crates thin.
  - Continue moving duplicated logic toward one authority with clear wiring/dispatch layers where applicable.
  - Keep docs and graphs reflecting actual crate layout.

- **Testing scale**
  - Favor generated/adversarial/property-based coverage for high-risk paths.
  - Keep conformance evidence tied to real workloads (parse, dataflow, graph, frontier, parser pipelines, optimizer passes).

- **Frontier work**
  - Keep parsing and runtime stacks concrete and measurable.
  - Keep benchmark evidence tied to throughput and convergence behavior.

## Open backlog (non-blocking unless marked)

- **Backend expansion:** Metal + DXIL path work begins once current CUDA/WGPU parity surfaces are stable.
- **Backend and IO-first innovations:** collective ops, IO-initiated GPU parse pipelines, and scaling experiments beyond current data sizes.
- **GPU-first formalism:** broader formal verification integration for algebraic laws and conformance certificates.
- **Frontend breadth:** broaden C parser and Rust frontends from beta confidence to production confidence through corpus breadth and failure taxonomies.

## Planned, not started here

- **Wasm/WebGPU package distribution** for browser-friendly demos.
- **New hardware backends** beyond Vulkan/CUDA where they preserve the same contracts.

The intent is to make every open item executable from this repo and to keep the gap between roadmap and reality visible in the doc status matrix.
