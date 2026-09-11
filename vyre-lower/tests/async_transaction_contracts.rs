//! The typed asynchronous-transfer surface: stated visibility, stated ring
//! slot, stated fence, and the pairing between a transfer and its wait.
//!
//! A transfer used to carry one name and nothing else, so a wait could only
//! mean "everything issued so far has landed": a bounded stage ring collapsed
//! into a full drain and a fence was chosen by whichever backend read the op.
//! These cases pin what the descriptor now states and what it refuses to state,
//! and they pin the staging that a selected pipeline performs at the lowering
//! boundary.
//!
//! What they do not catch: whether a concrete backend has a transfer mechanism
//! and a wait form for a declared transaction. That is the target's own
//! rejection, checked in the emitter crates.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_foundation::schedule::{SchedulePhaseId, SelectedSchedule};
use vyre_lower::descriptor_builder::{effect, lit};
use vyre_lower::{
    lower_physical, lower_scheduled, verify, AsyncTransaction, AsyncTransactionError,
    AsyncWaitSpec, BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody,
    KernelDescriptor, KernelOp, KernelOpKind, LiteralValue, MemoryClass, MemoryProxyFence,
    StageSlot, TransactionScope, VerifyErrorKind,
};

/// Every scope a transfer can be observed at, listed once so a new scope turns
/// the fence rules red until someone states the fence it needs.
const SCOPES: [TransactionScope; 4] = [
    TransactionScope::Invocation,
    TransactionScope::Subgroup,
    TransactionScope::Workgroup,
    TransactionScope::Device,
];

/// Every fence the descriptor can state.
const FENCES: [MemoryProxyFence; 3] = [
    MemoryProxyFence::None,
    MemoryProxyFence::Workgroup,
    MemoryProxyFence::Device,
];

fn transfer(tag: &str) -> Node {
    Node::AsyncLoad {
        source: Ident::from("pool"),
        destination: Ident::from("staged"),
        offset: Box::new(Expr::u32(0)),
        size: Box::new(Expr::u32(16)),
        tag: Ident::from(tag),
    }
}

fn wait(tag: &str) -> Node {
    Node::AsyncWait {
        tag: Ident::from(tag),
    }
}

/// A program that stages three transfers and waits for each of them.
fn three_transfers() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("pool", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1024),
            BufferDecl::storage("staged", 1, BufferAccess::ReadWrite, DataType::U32).with_count(16),
        ],
        [64, 1, 1],
        vec![
            transfer("first"),
            wait("first"),
            transfer("second"),
            wait("second"),
            transfer("third"),
            wait("third"),
        ],
    )
}

/// A two-phase schedule whose first phase runs as a three-slot pipeline.
fn pipelined_schedule() -> SelectedSchedule {
    vyre_test_support::selected_schedules::mapped_pipelined_two_phase()
}

/// A descriptor whose body is exactly the ops given, over the two global
/// bindings a transfer addresses and two literal words.
fn descriptor(ops: Vec<KernelOp>) -> KernelDescriptor {
    KernelDescriptor {
        id: "async_transaction".into(),
        bindings: BindingLayout {
            slots: vec![
                binding(0, BindingVisibility::ReadOnly, "pool"),
                binding(1, BindingVisibility::ReadWrite, "staged"),
            ],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops,
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(16)],
        },
    }
}

fn binding(slot: u32, visibility: BindingVisibility, name: &str) -> BindingSlot {
    BindingSlot {
        slot,
        element_type: DataType::U32,
        element_count: Some(16),
        memory_class: MemoryClass::Global,
        visibility,
        name: name.to_string(),
    }
}

/// Two literal words followed by one transfer and, optionally, its wait.
fn transfer_ops(issue: KernelOpKind, wait_kind: Option<KernelOpKind>) -> Vec<KernelOp> {
    let mut ops = vec![lit(0, 0), lit(1, 1), effect(issue, [0, 1, 0, 1])];
    if let Some(kind) = wait_kind {
        ops.push(effect(kind, []));
    }
    ops
}

fn kinds_of(descriptor: &KernelDescriptor) -> Vec<VerifyErrorKind> {
    match verify(descriptor) {
        Ok(()) => Vec::new(),
        Err(errors) => errors.into_iter().map(|error| error.kind).collect(),
    }
}

/// WHY: the fence is the whole reason a completed transfer is readable. Stating
/// a fence narrower than the scope the transfer claims means a reader inside
/// that scope can observe the destination before the transfer's writes, and
/// nothing downstream re-derives the requirement. The rule is checked over the
/// full scope-by-fence space so a new scope or fence cannot be added silently.
#[test]
fn a_fence_narrower_than_the_stated_visibility_is_rejected() {
    for scope in SCOPES {
        let required = scope.minimum_fence();
        for fence in FENCES {
            let spec =
                AsyncWaitSpec::fenced(AsyncTransaction::unstaged("dma".into(), scope), fence);
            let result = spec.validate();
            if fence.covers(required) {
                assert_eq!(
                    result,
                    Ok(()),
                    "Fix: {fence} orders a transfer observed at {scope} scope and must be accepted"
                );
            } else {
                assert_eq!(
                    result,
                    Err(AsyncTransactionError::FenceNarrowerThanVisibility {
                        fence,
                        visibility: scope,
                    }),
                    "Fix: {fence} cannot order a transfer observed at {scope} scope"
                );
            }
        }
        assert_eq!(
            AsyncWaitSpec::new(AsyncTransaction::unstaged("dma".into(), scope)).fence(),
            required,
            "Fix: the default wait takes the narrowest fence the visibility needs"
        );
    }
}

/// WHY: each of these declarations reads as a valid transfer and behaves as
/// none. An empty tag pairs with every wait and with no transfer; a zero-slot
/// ring states a bound of nothing; a slot outside its ring names a stage no
/// wait can complete. Each is broken on its own so no check covers for another.
#[test]
fn a_declaration_that_pairs_nothing_or_leaves_its_ring_is_rejected() {
    let empty = AsyncTransaction::unstaged("".into(), TransactionScope::Workgroup);
    assert_eq!(empty.validate(), Err(AsyncTransactionError::EmptyTag));

    let zero_ring = AsyncTransaction::staged(
        "dma".into(),
        TransactionScope::Workgroup,
        StageSlot::new(0, 0),
    );
    assert_eq!(zero_ring.validate(), Err(AsyncTransactionError::ZeroRing));

    let outside = AsyncTransaction::staged(
        "dma".into(),
        TransactionScope::Workgroup,
        StageSlot::new(3, 3),
    );
    assert_eq!(
        outside.validate(),
        Err(AsyncTransactionError::SlotOutOfRing {
            slot: 3,
            ring_slots: 3,
        })
    );

    let last = AsyncTransaction::staged(
        "dma".into(),
        TransactionScope::Workgroup,
        StageSlot::new(2, 3),
    );
    assert_eq!(
        last.validate(),
        Ok(()),
        "Fix: the last slot of a ring is a slot of that ring"
    );
}

/// WHY: the in-flight allowance is what a target reads to decide how much of
/// the ring it may leave outstanding. Deriving it from the ring depth is the
/// difference between overlap and a full drain, and an unstaged transfer must
/// state no allowance rather than an unbounded one.
#[test]
fn the_in_flight_allowance_follows_the_ring_depth() {
    let unstaged = AsyncTransaction::unstaged("dma".into(), TransactionScope::Workgroup);
    assert_eq!(unstaged.in_flight_allowed(), 0);
    assert!(!unstaged.is_staged());

    for depth in [1u16, 2, 3, 8, u16::MAX] {
        let staged = AsyncTransaction::staged(
            "dma".into(),
            TransactionScope::Workgroup,
            StageSlot::new(0, depth),
        );
        assert_eq!(
            staged.in_flight_allowed(),
            depth - 1,
            "Fix: waiting on one slot of a {depth}-slot ring leaves the other slots in flight"
        );
        assert!(staged.is_staged());
    }
}

/// WHY: a wait is the only thing that completes a transfer. A wait that names a
/// transfer the descriptor never issues leaves a target either draining an
/// unrelated transfer or fencing nothing, and a wait whose slot no issue filled
/// completes a stage that was never occupied. Both used to verify clean because
/// the pairing was a bare name nobody checked.
#[test]
fn the_verifier_rejects_a_wait_no_transfer_issues() {
    let unmatched = descriptor(transfer_ops(
        KernelOpKind::async_load("first".into()),
        Some(KernelOpKind::async_wait("second".into())),
    ));
    assert_eq!(
        kinds_of(&unmatched),
        vec![VerifyErrorKind::AsyncWaitUnmatched {
            tag: "second".into()
        }],
        "Fix: a wait must name a transfer this descriptor issues"
    );

    let staged_issue = KernelOpKind::AsyncLoad(Box::new(AsyncTransaction::staged(
        "dma".into(),
        TransactionScope::Workgroup,
        StageSlot::new(0, 3),
    )));
    let other_slot =
        KernelOpKind::AsyncWait(Box::new(AsyncWaitSpec::new(AsyncTransaction::staged(
            "dma".into(),
            TransactionScope::Workgroup,
            StageSlot::new(1, 3),
        ))));
    assert_eq!(
        kinds_of(&descriptor(transfer_ops(
            staged_issue.clone(),
            Some(other_slot)
        ))),
        vec![VerifyErrorKind::AsyncStageDisagreement { tag: "dma".into() }],
        "Fix: a wait completes the slot its transfer occupies"
    );

    let same_slot =
        KernelOpKind::AsyncWait(Box::new(AsyncWaitSpec::new(AsyncTransaction::staged(
            "dma".into(),
            TransactionScope::Workgroup,
            StageSlot::new(0, 3),
        ))));
    assert!(
        kinds_of(&descriptor(transfer_ops(staged_issue, Some(same_slot)))).is_empty(),
        "Fix: a wait for the slot its transfer occupies is well formed"
    );
}

/// WHY: an unstatable declaration must fail at the neutral boundary, not at
/// whichever backend reads it first, and a transfer states four operands: two
/// bindings, a byte offset and a byte size. The arity used to be checked
/// against two, so a transfer missing its size verified clean and every emitter
/// reported its own descriptor error instead.
#[test]
fn the_verifier_rejects_an_unstatable_transfer_declaration() {
    let empty_tag = KernelOpKind::AsyncLoad(Box::new(AsyncTransaction::unstaged(
        "".into(),
        TransactionScope::Workgroup,
    )));
    let kinds = kinds_of(&descriptor(transfer_ops(empty_tag, None)));
    assert!(
        kinds.contains(&VerifyErrorKind::AsyncTransactionUnstatable {
            reason: AsyncTransactionError::EmptyTag
        }),
        "Fix: an unpairable transfer must be reported by the neutral verifier; got {kinds:?}"
    );

    let mut ops = transfer_ops(KernelOpKind::async_load("dma".into()), None);
    ops[2].operands.truncate(3);
    assert_eq!(
        kinds_of(&descriptor(ops)),
        vec![VerifyErrorKind::OperandCountTooShort {
            expected_min: 4,
            got: 3
        }],
        "Fix: a transfer states two bindings, an offset and a size"
    );
}

/// WHY: staging is the point of the selected pipeline. Without it a wait is a
/// full drain and the ring depth the search priced buys nothing. Slots have to
/// rotate over the selected depth, a wait has to take the slot of the transfer
/// it completes, and a program lowered without a pipeline has to stay unstaged
/// rather than claim slot zero of a ring nobody selected.
#[test]
fn a_selected_pipeline_stages_every_transfer_and_its_wait() {
    let program = three_transfers();

    let unscheduled = lower_physical(&program).expect("a physical program must lower");
    for transaction in transactions(unscheduled.descriptor()) {
        assert!(
            !transaction.is_staged(),
            "Fix: a lowering with no selected pipeline states no ring slot"
        );
        assert_eq!(transaction.in_flight_allowed(), 0);
    }

    let scheduled = lower_scheduled(&program, &pipelined_schedule(), SchedulePhaseId(0))
        .expect("a validated schedule phase must lower");
    let projected = scheduled
        .schedule()
        .expect("a scheduled lowering carries its frozen facts");
    assert_eq!(projected.ring_slots, 3);

    let issued: Vec<StageSlot> = scheduled
        .descriptor()
        .ops_iter()
        .filter_map(|op| match &op.kind {
            KernelOpKind::AsyncLoad(transaction) | KernelOpKind::AsyncStore(transaction) => {
                transaction.stage()
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        issued,
        vec![
            StageSlot::new(0, 3),
            StageSlot::new(1, 3),
            StageSlot::new(2, 3)
        ],
        "Fix: slots rotate over the selected ring depth in descriptor order"
    );
    for slot in &issued {
        assert_eq!(slot.ring_slots(), 3);
    }

    let waited: Vec<StageSlot> = scheduled
        .descriptor()
        .ops_iter()
        .filter_map(|op| match &op.kind {
            KernelOpKind::AsyncWait(wait) => wait.transaction().stage(),
            _ => None,
        })
        .collect();
    assert_eq!(
        waited, issued,
        "Fix: a wait completes the slot its own transfer occupies"
    );
    assert!(
        verify(scheduled.descriptor()).is_ok(),
        "Fix: staging must leave the pairing well formed"
    );
}

fn transactions(descriptor: &KernelDescriptor) -> Vec<AsyncTransaction> {
    descriptor
        .ops_iter()
        .filter_map(|op| match &op.kind {
            KernelOpKind::AsyncLoad(transaction) | KernelOpKind::AsyncStore(transaction) => {
                Some(transaction.as_ref().clone())
            }
            KernelOpKind::AsyncWait(wait) => Some(wait.transaction().clone()),
            _ => None,
        })
        .collect()
}
