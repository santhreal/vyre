# vyre-emit

Emits target kernel source from verified `KernelDescriptor` values.

## Purpose

- `naga`: WGSL through a Naga module.
- `ptx`: PTX text, with optional NVRTC compilation (`nvrtc`).
- `spirv`: SPIR-V words.
- `metal`: Metal Shading Language text.

Each target is a cargo feature of the same name. A build enables only the targets it uses.

## Boundaries

- Depends on `vyre-foundation` and `vyre-compiler`.
- Input is a `KernelDescriptor` produced by `vyre_compiler::lower`. The crate does not read `Program` IR and does not lower.
- Output is source or bytecode. The crate does not acquire devices, allocate memory, launch kernels or cache pipelines; that is `vyre-runtime` and the drivers.
- An emitter makes no optimization decision the compiler did not record in the descriptor.
