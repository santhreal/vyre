//! Barrier scope parity between the PTX and WGSL emitters.
//!
//! WHY: both emitters used to collapse `Acquire`, `Release`, `AcqRel` and
//! `SeqCst` onto one construct. Naga emitted `STORAGE | WORK_GROUP` for all
//! four, and PTX emitted `bar.sync 0` for all four. A workgroup-scratch
//! reduction round therefore paid a device-scope storage fence it never needed,
//! and a global release fence paid full CTA convergence. Both are silent: the
//! kernel is correct and slower, so no test that only checks results can see it.
//!
//! The class this closes is one ordering losing its distinct scope again. The
//! variant space is read out of `MemoryOrdering::from_wire_tag`, so adding an
//! ordering to that enum turns this file red until a scope decision is recorded
//! for it in both emitters.
//!
//! What it does not catch: whether the narrowed fence is sufficient on real
//! hardware. That is the address-space analysis contract, pinned here only by
//! the three body shapes below, and by the golden corpora in each emitter.

use vyre_foundation::ir::{DataType, MemoryOrdering};
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op, shared_rw};
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

/// Workgroup scratch bindings live above the host-visible slot range, which
/// `vyre_lower::verify` enforces. Read from the lowering constant rather than
/// written out, so a change to the range moves this fixture with it.
const SCRATCH_SLOT: u32 = vyre_lower::WORKGROUP_SLOT_BASE;

/// Every ordering the memory model assigns a wire tag to. Read from the decoder
/// rather than written out, so a new variant appears here without an edit.
fn every_ordering() -> Vec<MemoryOrdering> {
    (0u8..=u8::MAX)
        .filter_map(|tag| MemoryOrdering::from_wire_tag(tag).ok())
        .collect()
}

/// The scope each barrier-valid ordering must lower to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// A memory fence with no thread convergence: PTX `membar.gl`, WGSL
    /// `STORAGE`.
    Fence,
    /// A full barrier within the issuing workgroup: PTX `bar.sync 0`, WGSL flags
    /// taken from the address spaces the barrier orders.
    WorkgroupBarrier,
    /// Not an instruction on either backend.
    Refused,
}

/// Only defined for orderings `MemoryOrdering::is_valid_for_barrier` accepts.
/// Every caller filters on that first, so no arm here is unexercised.
fn expected_scope(ordering: MemoryOrdering) -> Scope {
    match ordering {
        MemoryOrdering::Acquire | MemoryOrdering::Release | MemoryOrdering::AcqRel => Scope::Fence,
        MemoryOrdering::SeqCst => Scope::WorkgroupBarrier,
        MemoryOrdering::GridSync => Scope::Refused,
        MemoryOrdering::Relaxed => {
            panic!("Relaxed is not a barrier ordering; filter on is_valid_for_barrier first")
        }
        other => panic!(
            "Fix: memory ordering {other:?} has no recorded barrier scope. Map it in \
             vyre_emit_naga::emitter::op_lookup::barrier_flags and in the Barrier arm of \
             vyre_emit_ptx::emitter::dispatch, then record the decision in expected_scope."
        ),
    }
}

/// Barrier body that touches storage on both sides and nothing else.
fn storage_only(ordering: MemoryOrdering) -> KernelDescriptor {
    descriptor("barrier_storage_only")
        .slot(global_rw(0, DataType::U32, "buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0)])
                .op(op(KernelOpKind::LocalInvocationId, [0], 0))
                .op(lit(0, 1))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1]))
                .op(effect(KernelOpKind::Barrier { ordering }, []))
                .op(op(KernelOpKind::LoadGlobal, [0, 0], 2))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 2])),
        )
        .build()
}

/// Barrier body that touches workgroup scratch on both sides and nothing else.
fn scratch_only(ordering: MemoryOrdering) -> KernelDescriptor {
    descriptor("barrier_scratch_only")
        .slot(shared_rw(SCRATCH_SLOT, DataType::U32, 64, "tile"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0)])
                .op(op(KernelOpKind::LocalInvocationId, [0], 0))
                .op(lit(0, 1))
                .op(effect(KernelOpKind::StoreShared, [SCRATCH_SLOT, 0, 1]))
                .op(effect(KernelOpKind::Barrier { ordering }, []))
                .op(op(KernelOpKind::LoadShared, [SCRATCH_SLOT, 0], 2))
                .op(effect(KernelOpKind::StoreShared, [SCRATCH_SLOT, 0, 2])),
        )
        .build()
}

/// Barrier body that stages storage through workgroup scratch, so both address
/// spaces are demonstrated and neither flag may be dropped.
fn both_spaces(ordering: MemoryOrdering) -> KernelDescriptor {
    descriptor("barrier_both_spaces")
        .slot(global_rw(0, DataType::U32, "buf"))
        .slot(shared_rw(SCRATCH_SLOT, DataType::U32, 64, "tile"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0)])
                .op(op(KernelOpKind::LocalInvocationId, [0], 0))
                .op(op(KernelOpKind::LoadGlobal, [0, 0], 1))
                .op(effect(KernelOpKind::StoreShared, [SCRATCH_SLOT, 0, 1]))
                .op(effect(KernelOpKind::Barrier { ordering }, []))
                .op(op(KernelOpKind::LoadShared, [SCRATCH_SLOT, 0], 2))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 2])),
        )
        .build()
}

fn emit_ptx(desc: &KernelDescriptor) -> Result<String, String> {
    let verified = vyre_lower::verify_descriptor(desc)
        .unwrap_or_else(|error| panic!("descriptor `{}` must verify: {error:?}", desc.id));
    vyre_emit_ptx::emit(&verified).map_err(|error| format!("{error}"))
}

/// Flags of every barrier in the emitted module, in emission order.
fn naga_barriers(desc: &KernelDescriptor) -> Result<Vec<naga::Barrier>, String> {
    let verified = vyre_lower::verify_descriptor(desc)
        .unwrap_or_else(|error| panic!("descriptor `{}` must verify: {error:?}", desc.id));
    let module = vyre_emit_naga::emit(&verified).map_err(|error| format!("{error}"))?;
    let mut flags = Vec::new();
    for entry in &module.entry_points {
        collect_barriers(&entry.function.body, &mut flags);
    }
    Ok(flags)
}

fn collect_barriers(block: &naga::Block, out: &mut Vec<naga::Barrier>) {
    for statement in block.iter() {
        match statement {
            naga::Statement::Barrier(flags) => out.push(*flags),
            naga::Statement::Block(inner) => collect_barriers(inner, out),
            naga::Statement::If { accept, reject, .. } => {
                collect_barriers(accept, out);
                collect_barriers(reject, out);
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_barriers(body, out);
                collect_barriers(continuing, out);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_barriers(&case.body, out);
                }
            }
            _ => {}
        }
    }
}

fn sole_barrier(flags: &[naga::Barrier], label: &str) -> naga::Barrier {
    assert_eq!(
        flags.len(),
        1,
        "Fix: {label} must emit exactly one WGSL barrier, found {flags:?}."
    );
    flags[0]
}

/// A fence ordering must not converge the CTA, and a workgroup barrier must not
/// degrade into a bare fence. Collapsing the two is the regression this pins.
#[test]
fn ptx_separates_memory_fences_from_cta_barriers() {
    for ordering in every_ordering()
        .into_iter()
        .filter(|o| o.is_valid_for_barrier())
    {
        let emitted = emit_ptx(&storage_only(ordering));
        match expected_scope(ordering) {
            Scope::Fence => {
                let ptx = emitted.unwrap_or_else(|error| {
                    panic!("Fix: fence ordering {ordering:?} must lower to PTX: {error}")
                });
                assert!(
                    ptx.contains("membar.gl;"),
                    "Fix: {ordering:?} is a memory fence and must lower to `membar.gl`."
                );
                assert!(
                    !ptx.contains("bar.sync 0;"),
                    "Fix: {ordering:?} must not converge the CTA; `bar.sync 0` over-synchronizes \
                     a fence that only orders global memory."
                );
            }
            Scope::WorkgroupBarrier => {
                let ptx = emitted
                    .unwrap_or_else(|error| panic!("Fix: {ordering:?} must lower to PTX: {error}"));
                assert!(
                    ptx.contains("bar.sync 0;"),
                    "Fix: {ordering:?} is a full workgroup barrier and must lower to `bar.sync 0`."
                );
                assert!(
                    !ptx.contains("membar.gl;"),
                    "Fix: `bar.sync 0` already fences; a redundant `membar.gl` beside it is dead \
                     cost."
                );
            }
            Scope::Refused => {
                let error = emitted.expect_err(&format!(
                    "Fix: {ordering:?} is not lowerable to a CTA-scope construct and must be \
                     refused, not silently downgraded."
                ));
                assert!(
                    error.contains("MemoryOrdering::GridSync"),
                    "Fix: refusal for {ordering:?} must name the ordering it rejected: {error}"
                );
            }
        }
    }
}

/// WGSL flags follow the address spaces the barrier orders, not the ordering
/// alone. A scratch-only round must not request a storage fence.
#[test]
fn wgsl_barrier_flags_follow_the_fenced_address_space() {
    let scratch = naga_barriers(&scratch_only(MemoryOrdering::SeqCst))
        .expect("scratch-only workgroup barrier must emit");
    assert_eq!(
        sole_barrier(&scratch, "a scratch-only SeqCst barrier"),
        naga::Barrier::WORK_GROUP,
        "Fix: a barrier whose body touches only workgroup scratch must emit WORK_GROUP alone; \
         a storage fence there is unpaid-for device-scope synchronization."
    );

    let storage = naga_barriers(&storage_only(MemoryOrdering::SeqCst))
        .expect("storage-only workgroup barrier must emit");
    assert_eq!(
        sole_barrier(&storage, "a storage-only SeqCst barrier"),
        naga::Barrier::STORAGE,
        "Fix: a barrier whose body touches only storage must emit STORAGE alone."
    );

    let mixed =
        naga_barriers(&both_spaces(MemoryOrdering::SeqCst)).expect("staged barrier must emit");
    assert_eq!(
        sole_barrier(&mixed, "a SeqCst barrier staging storage through scratch"),
        naga::Barrier::STORAGE | naga::Barrier::WORK_GROUP,
        "Fix: a barrier that orders both address spaces must keep both flags; dropping either \
         one loses a real ordering edge."
    );
}

/// The three fence orderings name global-memory visibility, so they lower to
/// `STORAGE` whatever the surrounding body touches. Sharing the SeqCst
/// address-space analysis would make a scratch-only acquire fence forget storage.
#[test]
fn wgsl_fence_orderings_always_fence_storage() {
    for ordering in every_ordering()
        .into_iter()
        .filter(|o| o.is_valid_for_barrier() && expected_scope(*o) == Scope::Fence)
    {
        for desc in [
            scratch_only(ordering),
            storage_only(ordering),
            both_spaces(ordering),
        ] {
            let label = desc.id.to_string();
            let flags = naga_barriers(&desc)
                .unwrap_or_else(|error| panic!("Fix: {ordering:?} in `{label}`: {error}"));
            assert_eq!(
                sole_barrier(&flags, &format!("{ordering:?} in `{label}`")),
                naga::Barrier::STORAGE,
                "Fix: {ordering:?} is a storage fence and must emit STORAGE regardless of the \
                 address spaces in its body."
            );
        }
    }
}

/// `GridSync` has no WGSL instruction and no cooperative launch on wgpu, so it
/// must be cut into sequential dispatches by the planner. The emitter refusal is
/// the backstop, and it must say where the cut belongs.
#[test]
fn wgsl_refuses_grid_sync_and_names_the_planner_cut() {
    let error = naga_barriers(&storage_only(MemoryOrdering::GridSync))
        .expect_err("Fix: GridSync must never lower to a workgroup-scope WGSL barrier.");
    assert!(
        error.contains("splitting"),
        "Fix: the GridSync refusal must direct the caller to dispatch splitting: {error}"
    );
}
