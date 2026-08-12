# vyre-runtime

Artifact admission, persistent resident queues, model residency, and Linux
zero-copy IO for Vyre.

## What this crate provides

The runtime authenticates compiler envelopes, materializes target payloads,
binds artifact ABI resources, submits typed work, and rematerializes the same
artifact after device loss.

| Module | Purpose |
|--------|---------|
| `artifact_admission` | Envelope authentication, exact target selection, materialization, and retained sessions |
| `persistent_executor` | Resident queue submission over retained artifact bindings |
| `resident_work_queue` | Ring protocol, host mirrors, queue sizing, IO, and telemetry |
| `pipeline_cache` | Content-addressed storage keyed by neutral artifact digest |
| `checkpoint` | Bounded safetensors metadata and sharded checkpoint verification |
| `model_residency` | Immutable weights, artifact instances, admission budgets, and generation-checked sequence state |
| `uring` | Linux registered-buffer and direct NVMe ingest |

## Quick start

```rust
use vyre_driver::backend::BackendRegistration;
use vyre_runtime::resident_work_queue::{self, ResidentWorkQueue};
use vyre_runtime::{PersistentExecutor, ResidentQueueState};

fn run(
    backend: &'static BackendRegistration,
    envelope: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let initial = ResidentQueueState {
        control: ResidentWorkQueue::try_encode_control(false, 1, 0)?,
        ring: ResidentWorkQueue::try_encode_empty_ring(256)?,
        debug_log: ResidentWorkQueue::try_encode_empty_debug_log(
            resident_work_queue::debug::RECORD_CAPACITY,
        )?,
        io_queue: resident_work_queue::io::try_encode_empty_io_queue(
            resident_work_queue::io::IO_SLOT_COUNT,
        )?,
    };
    let executor = PersistentExecutor::from_bytes(backend, envelope, initial.clone())?;
    let completed = executor.submit_and_wait(initial)?;
    assert!(!completed.state.control.is_empty());
    Ok(())
}
```

## Validate checkpoint metadata

Open the local shard index before you allocate device memory:

```rust
use std::path::Path;
use vyre_runtime::checkpoint::{
    ExpectedShardDigest, SafetensorDtype, SafetensorRequirement,
    ShardedSafetensorIndex,
};

fn validate_model(trusted_shard_blake3: [u8; 32])
    -> Result<[u8; 32], Box<dyn std::error::Error>>
{
    let index = ShardedSafetensorIndex::open(
        "/models/example",
        "/models/example/model.safetensors.index.json",
    )?;
    index.validate_requirements([SafetensorRequirement {
        name: "model.language_model.embed_tokens.weight",
        dtype: SafetensorDtype::BF16,
        shape: &[248_320, 5_120],
    }])?;
    let identity = index.verify_shards([ExpectedShardDigest {
        shard: Path::new("model.safetensors-00001-of-00001.safetensors"),
        blake3: trusted_shard_blake3,
    }])?;
    Ok(identity.content_digest())
}
```

The loader reads only the bounded index and shard headers. It rejects malformed
metadata, duplicate names, invalid ranges, path traversal, root-escaping
symlinks, missing mapped tensors, unmapped shard tensors, and model requirement
drift. The metadata manifest digest covers the exact index bytes and each
validated shard metadata identity. `verify_shards` then streams every complete
shard through a fixed-size buffer, requires the exact trusted digest set, and
returns the immutable full-checkpoint identity you admit to a device.

## Own model and sequence residency

Admit verified weights once, then give each sequence separate mutable state:

```rust
use std::sync::Arc;
use vyre_driver::backend::{ArtifactInstance, VyreBackend};
use vyre_runtime::model_residency::{
    ArtifactInstanceBinding, ImmutableWeightUpload, ModelAdmission,
    ModelResidency, ModelResidencyKey, SequenceStateSpec,
};

fn run_sequence(
    backend: Arc<dyn VyreBackend>,
    instance: Arc<dyn ArtifactInstance>,
    checkpoint_digest: [u8; 32],
    weight_bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let key = ModelResidencyKey {
        checkpoint_digest,
        artifact_digest: instance.artifact().0,
    };
    let budget = u64::try_from(weight_bytes.len())?
        .checked_add(4096)
        .ok_or_else(|| std::io::Error::other("model residency budget overflow"))?;
    let residency = ModelResidency::new(backend, budget);
    residency.admit_model(ModelAdmission {
        key,
        weights: vec![ImmutableWeightUpload {
            name: "embedding.weight",
            bytes: weight_bytes,
            blake3: *blake3::hash(weight_bytes).as_bytes(),
        }],
        artifacts: vec![ArtifactInstanceBinding::new("decode", instance, 0)],
    })?;
    let sequence = residency.start_sequence(
        key,
        &[SequenceStateSpec {
            name: "kv_cache",
            byte_len: 4096,
        }],
    )?;
    let reset_sequence = residency.reset_sequence(sequence)?;
    residency.finish_sequence(reset_sequence)?;
    residency.evict_model(key)?;
    Ok(())
}
```

Cold admission verifies each weight digest before allocation. A warm admission
reuses only an exact checkpoint and artifact key. Allocation or upload failure
rolls back all earlier resources. Sequence cancellation, completion, and reset
release or zero mutable state, and generation checks reject stale leases.
Eviction refuses a model while any sequence remains active.

<!-- BEGIN GENERATED CRATE CONTRACT -->
## Crate contract

This section is generated by `python3 scripts/crate_readmes.py --write` from
the crate manifest, release train, ownership registry, and crate-guide metadata.

### Purpose

Own compile-to-materialize orchestration, artifact sessions, recovery, persistence, residency, scheduling, caches, telemetry, readback, and IO.

### Boundaries

The `runtime` owner maintains this `runtime` crate at `vyre-runtime`.
Its allowed internal production dependencies are: `vyre-driver`, `vyre-foundation`, `vyre-megakernel`, `vyre-self-substrate`.
Any other normal or build dependency requires an ownership-registry change.

### Minimal real example

Run the checked-in behavior from `vyre-runtime/examples/vyre_runtime_release_surface.rs`:

```console
CARGO_BUILD_JOBS=1 ./cargo_full run -p vyre-runtime --example vyre_runtime_release_surface
```

### Features

- Manifest features: `c-frontend-adapter`, `default`, `megakernel-batch`, `remote`, `remote-cache`, `self-substrate-adapters`, `subgroup-ops`, `uring-cmd-nvme`
- Default feature members: None

### Errors and unsupported behavior

Invalid plans, stale artifacts, unavailable selected backends, IO failures, and illegal state transitions are operator-visible errors.

### Testing

Use [`docs/testing/vyre-runtime.md`](../docs/testing/vyre-runtime.md) for exact commands, Cargo targets, hardware
requirements, evidence outputs, expected skips, and failure semantics.

### Release status

This crate is an active experimental runtime surface in the 0.7.2 workspace. Its public contracts follow the Vyre release train.

### Ownership

`docs/CRATE_OWNERSHIP.toml` is authoritative for this crate's responsibility
and allowed internal edges. Regenerate `docs/CRATE_GRAPH.md` and
`docs/OWNERSHIP.md` after changing that registry.

### License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license

at your option. See the workspace `LICENSE-APACHE` and `LICENSE-MIT` files.

<!-- END GENERATED CRATE CONTRACT -->
