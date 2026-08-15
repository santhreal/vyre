# vyre-runtime  -  architecture

The runtime layer that wires per-backend dispatchers into a single
substrate-neutral surface. It owns the resident work queue, the
authenticated artifact session, the pipeline cache, replay, backend
routing, and Linux io_uring ingest.

Artifact compilation and target selection are not here. They live in
`vyre-megakernel`, which produces the artifact envelope this crate
authenticates and executes.

## Modules

### `resident_work_queue/`
The persistent kernel: the wire protocol every tenant follows, the host
mirrors of it, the scheduling policy, and the in-kernel IO queue.

- `protocol/` and `protocol.rs`  -  ring slot words, opcodes, control
  block, and the codec every tenant encodes against.
- `builder/` and `builder.rs`  -  the program builders, from the default
  sharded loop to the finite one-pass and JIT variants.
- `handlers.rs`  -  opcode dispatch emitter for caller-supplied handlers.
- `scheduler/` and `scheduler.rs`  -  priority partitioning and fairness
  budgets across ring slots.
- `planner/`  -  launch geometry, grid sizing, barrier placement, and
  fusion decisions for a dispatch.
- `policy/` and `policy.rs`  -  admission and adoption policy for resident
  work, including the LRU tick cache behind it.
- `io/`  -  the IO queue protocol for in-kernel async loads and the poll
  and completion halves of it.
- `telemetry/` and `telemetry.rs`  -  ring state observation, evidence
  records, and the sketch that bounds their cost.
- `readback.rs`, `resident.rs`, `ring.rs`, `task.rs`  -  host-side mirrors
  of the queue and its slots.
- `speculation.rs`  -  the speculative-execution adoption verdict paired
  with the policy that proposes it.
- `workspace_adapter.rs`, `workspace_layout.rs`  -  consumer-owned
  resident workspace binding.

### `artifact_admission.rs`
Authenticates an artifact envelope and admits exactly the target payload
format the registered backend requires.

### `persistent_executor.rs`
Authenticated persistent execution over retained artifact bindings.

### `pipeline_cache/`
Content-addressed authenticated artifact cache above each backend's own
cache. The key derives from the neutral artifact identity alone.

### `recovery.rs`
Structured artifact-session recovery: classifies a backend error into a
retry class without parsing messages or recompiling.

### `replay.rs`
Circular on-disk log of every published ring slot, so a later run can
diff epoch-by-epoch execution against a live backend.

### `resource_residency.rs`
Backend-neutral immutable-resource and mutable-state residency.

### `routing/`
Backend selection: given a program and adapter capabilities, pick a
backend that supports every operation the program uses.

### `scheduler.rs`
Multi-GPU work partitioning across runtime backends.

### `tenant/`
Multiple clients share one resident kernel per GPU by publishing into
different opcode partitions, keyed by the `tenant_id` field the ring
protocol already carries.

### `uring/`
Linux io_uring integration and zero-copy SSD ingest. Compiled out on
macOS and Windows.

## Public types

- **`PipelineError`**  -  the one runtime error surface, including the
  first-class `DrainIncomplete` variant that reports an under-drained
  dispatch rather than letting a partial hit set pass as complete.
- **`ResidentWorkQueue`**  -  stateless owner of queue encoding and
  decoding.
- **`RingSlotTransition`**  -  the one public path to a slot state change.
- **`UringCompletionPump`** and **`UringPollState`**  -  the optional
  io_uring completion pump and its observable state.

## Integration points

- Downstream fused-dispatch paths route through this layer when they opt
  into resident execution.
- `vyre-aot` calls into this for the runtime side of the AOT artifact
  loader.
