# vyre-runtime

Executes compiled vyre artifacts on registered devices.

## Purpose

- Defines the device, target compiler, materializer, binding, submission and completion contracts each backend implements.
- Holds the backend registry and selects a registered backend for an artifact.
- Admits artifacts (`artifact_admission::ArtifactSession`), binds host or resident resources, submits work and reads results back.
- Owns residency, resident work queues, recovery, pipeline caches, scheduling and IO (`uring`, GPUDirect).

## Boundaries

- Depends on `vyre-foundation`, `vyre-spec`, `vyre-compiler` and `vyre-libs`.
- Contains no lowering and no target code generation. It consumes artifacts from `vyre-compiler` and modules from `vyre-emit` through the backends.
- Backend crates (`vyre-driver-*`) implement its contracts. It never names a concrete backend.
- No CPU fallback. A missing or failing device is an error with a corrective action.
- Host bytes are borrowed until upload and moved on readback. A copy on the dispatch path is a defect.
