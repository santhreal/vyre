# RFC 0005: Persistent megakernel and ring-buffer submission

Last verified: 2026-08-04

Status: **Superseded**

This RFC records the historical motivation for persistent GPU submission. Its
original device-bytecode-interpreter design is not the current Vyre
architecture. Use [`../megakernel-wiring.md`](../megakernel-wiring.md) and
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) for current ownership and execution
contracts.

## Historical motivation

Per-program launches have fixed driver, queue, and synchronization costs.
Workloads made from many small dependent operations can spend more time in
submission than in useful device work. A resident kernel with a bounded queue
can amortize that cost and keep scheduler state on the device.

The original proposal therefore explored:

- one long-lived device kernel;
- ring-buffer submission;
- explicit queue identities for multiple tenants;
- device-visible completion states;
- comparison against ordinary dispatch for correctness and latency.

That motivation remains valid. Persistent scheduling is implemented under
`vyre-runtime/src/megakernel/`.

## Superseded design

The original RFC proposed sending VIR bytes to a general interpreter inside the
resident kernel. The kernel would decode arbitrary IR tags and execute them.
That design is superseded for three reasons.

1. It creates a Category B execution engine. Vyre does not support a general
   host or device opcode interpreter as an alternative to typed lowering.
2. It duplicates validation and lowering semantics inside a persistent kernel.
   Every IR change would require synchronized interpreter changes.
3. It weakens backend ownership. Concrete targets would share an interpreter
   implementation while still needing target-specific device behavior.

Current persistent execution starts from a validated typed `Program`, derives a
runtime descriptor and backend requirements, and uses the selected concrete
driver for device lowering and dispatch.

## Current resolution

The current ownership split is:

- `vyre-foundation` owns typed IR, validation, and semantic optimization.
- `vyre-lower` owns backend-neutral lowering analysis and descriptors.
- `vyre-driver` owns backend-neutral launch, capability, routing, cache, and
  evidence contracts.
- `vyre-runtime/src/megakernel/` owns persistent queue protocol, scheduling,
  resident execution coordination, readback, telemetry, and recovery.
- Concrete drivers own target lowering and device dispatch.
- `vyre-aot` owns artifact packaging.
- `vyre-megakernel` is a current workspace member. It compiles typed program
  graphs into canonical static and persistent artifacts without owning backend
  dispatch. Vyre does not support a general interpreter inside the kernel.

## Queue invariants retained from this RFC

The persistent runtime keeps the useful protocol requirements:

- Publication initializes a complete slot before making it visible.
- One claimant owns a published slot.
- Completion makes outputs visible before reuse.
- Capacity, offsets, state transitions, and tenant identity are validated.
- Unsupported capabilities and malformed descriptors fail explicitly.
- Timeout, recovery, and device failure produce terminal operator-visible
  results.

Exact slot constants and codecs live in `vyre-runtime/src/megakernel/protocol/`.
They are intentionally not copied into this historical RFC.

## Multi-tenancy

Tenant isolation remains a runtime requirement. A descriptor names its tenant
and resources. Scheduling validates ownership before publication and before
readback. One tenant cannot use another tenant's queue slot or buffer identity.

Fairness policy belongs in `vyre-runtime/src/megakernel/scheduler.rs` and
`policy.rs`. A concrete backend does not define a separate fairness model.

## Backend scope

Persistent execution is capability-gated. CUDA is the preferred release route
on the NVIDIA evidence host. WGPU is the portable GPU route. SPIR-V is a
registered dispatch route. Metal is active on supported Apple targets. A
backend that cannot honor a persistent requirement returns an explicit error.
It does not route through the reference interpreter or ordinary dispatch
silently.

Current executable backend state is recorded in
`release/evidence/backends/backend-matrix.json`. Operation support is recorded
in `docs/optimization/OP_MATRIX.toml`.

## Verification contract

A persistent-path change proves:

1. Standard and persistent routes produce the same declared output bytes for an
   eligible program.
2. Queue publication, claim, completion, reuse, and recovery transitions are
   deterministic under adversarial scheduling.
3. Tenant resources remain isolated.
4. Unsupported requirements return the documented error.
5. Performance evidence preserves raw samples and matches the current source
   fingerprint.

The original latency estimates in this RFC were design targets. They are not
release claims. Only current benchmark artifacts may support a measured claim.

## Alternatives

### Ordinary dispatch only

Ordinary dispatch remains the baseline path. It is simpler and supports programs
that are not eligible for resident execution. It does not amortize repeated
submission for small compatible work.

### Backend graph APIs

Concrete graph APIs can reduce launch overhead for a fixed graph. They remain
backend-owned Layer 2 or runtime integration mechanisms. They do not define the
portable typed megakernel artifact.

### One interpreter per backend

This was rejected with the general interpreter design. It multiplies semantic
implementations and conflicts with typed lowering.

## Historical outcome

The RFC established the need for persistent submission and explicit queue
semantics. The bytecode-interpreter mechanism was replaced by typed program
planning, a shared runtime protocol, concrete backend lowering, and the current
`vyre-megakernel` workspace member that freezes canonical megakernel artifacts.
