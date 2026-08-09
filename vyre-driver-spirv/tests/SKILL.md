# tests/SKILL.md  -  vyre-driver-spirv

Read `../../.internals/skills/testing/SKILL.md` first for the category contract.

## Purpose

`vyre-driver-spirv` is the target adapter and runtime owner for SPIR-V devices.
`Program` input enters through `vyre_lower::lower_verified`; descriptor
serialization is delegated to `vyre-emit-spirv`.

## Critical invariants

- The adapter never implements a second descriptor-to-SPIR-V writer.
- Identical `Program` input produces identical SPIR-V words.
- Invalid programs fail at verified lowering before target serialization.
- Runtime admission and dispatch errors remain actionable.

## Adversarial surface

- Invalid workgroup geometry.
- Unsupported neutral descriptor operations.
- Target payloads rejected during runtime admission.

## Cross-crate contracts

- Consumes `Program` values through `vyre_lower::lower_verified`.
- Delegates descriptor serialization to `vyre-emit-spirv`.
- Owns backend registration and Vulkan execution.

## Bench targets

- `emit_spv`: adapter throughput across small, medium, and large programs.

## Fuzz targets

- `emit_spv_fuzz`: arbitrary programs never panic and always return `Result`.

## What NOT to test here

- Descriptor-to-SPIR-V construction owned by `vyre-emit-spirv`.
- Naga writer parity owned by the emitter crate.

## Running

```bash
./cargo_full test -p vyre-driver-spirv
./cargo_full test -p vyre-driver-spirv --test adversarial
./cargo_full test -p vyre-driver-spirv --test property
./cargo_full test -p vyre-driver-spirv --test gap
./cargo_full test -p vyre-driver-spirv --test integration
./cargo_full bench -p vyre-driver-spirv
```
