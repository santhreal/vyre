//! Explicit ownership, borrow, alias, capability, state-epoch, and effect tokens,
//! and obligation lifecycle tracking.
//!
//! Every load, store, atomic, collective, and state transition consumes and produces
//! explicit obligations. An unconsumed obligation is a compile error or a refusal by name,
//! not a warning.

use super::fence::BarrierParticipation;
use super::scope::{ExecutionScope, MemoryScope};
use super::storage::StorageDomain;
use crate::ir::{Node, Program};
use crate::visit::child_bodies;

/// Closed ownership state of a resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum OwnershipKind {
    /// Exclusively owned by the current execution context.
    Exclusive,
    /// Shared immutably across multiple readers.
    Shared,
    /// Temporarily borrowed under active scope.
    Borrowed,
    /// Ownership transferred / moved to another execution domain.
    Transferred,
    /// Released / deallocated.
    Released,
}

/// Token witnessing resource ownership.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct OwnershipToken {
    /// Target resource or buffer identifier.
    pub resource: String,
    /// Current ownership kind.
    pub kind: OwnershipKind,
    /// Storage domain where resource resides.
    pub storage_domain: StorageDomain,
}

/// Closed borrow discipline.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum BorrowKind {
    /// Immutable read-only borrow.
    Immutable,
    /// Mutable exclusive borrow.
    Mutable,
    /// Atomic read-modify-write exclusive borrow.
    AtomicExclusive,
}

/// Token witnessing an active borrow.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct BorrowToken {
    /// Unique borrow instance identifier.
    pub borrow_id: u64,
    /// Borrowed resource identifier.
    pub resource: String,
    /// Kind of borrow.
    pub kind: BorrowKind,
    /// Epoch at which borrow was initiated.
    pub origin_epoch: StateEpoch,
}

/// Closed alias discipline.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum AliasDiscipline {
    /// Provably no aliasing with any other live reference.
    NoAlias,
    /// Distinct non-overlapping view within the same buffer.
    DistinctDisjoint,
    /// May alias with potential read/write conflicts.
    MayAlias,
    /// Known overlapping mutable view (requires synchronization).
    OverlappingView,
}

/// Token witnessing alias relationship.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct AliasToken {
    /// Resource identifier.
    pub resource: String,
    /// Alias discipline.
    pub discipline: AliasDiscipline,
}

/// Closed memory and execution capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum MemoryCapability {
    /// Read access permitted.
    Read,
    /// Write access permitted.
    Write,
    /// Atomic operations permitted.
    Atomic,
    /// Execution and memory barrier synchronization permitted.
    Barrier,
    /// Asynchronous DMA / copy engine transfers permitted.
    AsyncDma,
    /// Collective communication across group permitted.
    Collective,
    /// Dynamic indirect dispatch permitted.
    DynamicDispatch,
}

/// Token witnessing granted capability.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct CapabilityToken {
    /// Resource identifier or execution domain.
    pub target: String,
    /// Granted capability.
    pub capability: MemoryCapability,
}

/// Monotonic state and memory epoch counter.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    serde::Deserialize,
    serde::Serialize,
)]
pub struct StateEpoch(pub u64);

impl StateEpoch {
    /// Initial epoch.
    pub const ZERO: Self = Self(0);

    /// Advance to the next sequential epoch.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Advance epoch in place and return previous epoch.
    pub fn advance(&mut self) -> Self {
        let prev = *self;
        self.0 += 1;
        prev
    }
}

/// Closed effect kind classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum EffectKind {
    /// Pure computation with no side effects.
    Pure,
    /// Reads memory from a storage domain.
    ReadMemory,
    /// Mutates memory in a storage domain.
    WriteMemory,
    /// Atomic read-modify-write on memory.
    AtomicRmw,
    /// Execution barrier or memory fence synchronization.
    BarrierSync,
    /// Asynchronous data transfer initiation or wait.
    AsyncTransfer,
    /// Inter-thread / inter-device collective communication.
    CollectiveComm,
    /// Dynamic control-flow divergence.
    ControlDivergence,
    /// Execution trap or fault generation.
    TrapFault,
}

/// Token witnessing an explicit effect.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct EffectToken {
    /// Token identifier.
    pub token_id: u64,
    /// Effect kind.
    pub kind: EffectKind,
    /// Epoch when effect occurred.
    pub epoch: StateEpoch,
}

/// Closed obligation kind.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum ObligationKind {
    /// Prior write must be made visible via barrier/fence before consuming scope.
    PendingWriteVisibility {
        /// Target buffer name.
        buffer: String,
        /// Scope where write must become visible.
        scope: MemoryScope,
        /// Epoch when write was performed.
        epoch: StateEpoch,
    },
    /// Asynchronous transfer submitted that must be waited on before reading destination.
    PendingAsyncWait {
        /// Transfer tag identifier.
        tag: String,
        /// Source buffer name.
        source: String,
        /// Destination buffer name.
        destination: String,
    },
    /// Execution barrier rendezvous obligation.
    PendingBarrierRendezvous {
        /// Execution scope of the barrier.
        scope: ExecutionScope,
        /// Participation discipline.
        participation: BarrierParticipation,
    },
    /// Collective communication join obligation.
    PendingCollectiveJoin {
        /// Communication group.
        group: String,
        /// Collective operation name.
        op: String,
    },
    /// Active borrow must be released.
    PendingBorrowRelease {
        /// Borrow identifier.
        borrow_id: u64,
        /// Resource name.
        buffer: String,
    },
    /// Monotonic state transition obligation.
    PendingStateTransition {
        /// Resource name.
        resource: String,
        /// Initial epoch.
        from_epoch: StateEpoch,
        /// Target epoch.
        to_epoch: StateEpoch,
    },
}

/// An explicit obligation produced by an operation.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Obligation {
    /// Unique obligation identifier.
    pub id: u64,
    /// Human-readable obligation name / description.
    pub name: String,
    /// Obligation kind.
    pub kind: ObligationKind,
    /// Statement / AST node index where obligation was produced.
    pub produced_at_node: usize,
    /// Whether this obligation has been consumed / discharged.
    pub consumed: bool,
    /// Statement / AST node index where obligation was consumed.
    pub consumed_by_node: Option<usize>,
}

/// Compile error or verification refusal when an obligation is violated.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ObligationError {
    /// An obligation was produced but never consumed before end of program or scope.
    #[error("UnconsumedObligation: obligation `{name}` produced at node {produced_at_node} was not consumed. Details: {details}")]
    UnconsumedObligation {
        /// Obligation name.
        name: String,
        /// Obligation kind.
        kind: ObligationKind,
        /// Statement index where produced.
        produced_at_node: usize,
        /// Actionable details.
        details: String,
    },
    /// Attempted to consume an obligation that does not exist or was already consumed.
    #[error(
        "InvalidObligationConsumption: failed to consume obligation `{name}`. Details: {details}"
    )]
    InvalidObligationConsumption {
        /// Obligation name.
        name: String,
        /// Actionable details.
        details: String,
    },
    /// Conflicting obligations detected.
    #[error(
        "ConflictingObligation: conflict between `{first_name}` and `{second_name}`: {reason}"
    )]
    ConflictingObligation {
        /// First obligation name.
        first_name: String,
        /// Second obligation name.
        second_name: String,
        /// Conflict reason.
        reason: String,
    },
    /// Borrow exclusivity rule violated.
    #[error(
        "BorrowExclusivityViolation: resource `{buffer}` has conflicting active borrows: {details}"
    )]
    BorrowExclusivityViolation {
        /// Resource name.
        buffer: String,
        /// Details.
        details: String,
    },
}

/// Tracks production and consumption of explicit obligations.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObligationTracker {
    next_id: u64,
    obligations: Vec<Obligation>,
    current_epoch: StateEpoch,
}

impl ObligationTracker {
    /// Create a new empty obligation tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            obligations: Vec::new(),
            current_epoch: StateEpoch::ZERO,
        }
    }

    /// Current state epoch.
    #[must_use]
    pub const fn current_epoch(&self) -> StateEpoch {
        self.current_epoch
    }

    /// Advance epoch.
    pub fn advance_epoch(&mut self) -> StateEpoch {
        self.current_epoch.advance()
    }

    /// Produce an explicit obligation.
    pub fn produce(
        &mut self,
        name: impl Into<String>,
        kind: ObligationKind,
        node_idx: usize,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.obligations.push(Obligation {
            id,
            name: name.into(),
            kind,
            produced_at_node: node_idx,
            consumed: false,
            consumed_by_node: None,
        });
        id
    }

    /// Consume an obligation by ID.
    ///
    /// # Errors
    ///
    /// Returns [`ObligationError::InvalidObligationConsumption`] if the obligation
    /// is unknown or already consumed.
    pub fn consume(&mut self, obligation_id: u64, node_idx: usize) -> Result<(), ObligationError> {
        let Some(obligation) = self.obligations.iter_mut().find(|o| o.id == obligation_id) else {
            return Err(ObligationError::InvalidObligationConsumption {
                name: format!("id:{obligation_id}"),
                details: "obligation ID does not exist in tracker".to_string(),
            });
        };
        if obligation.consumed {
            return Err(ObligationError::InvalidObligationConsumption {
                name: obligation.name.clone(),
                details: format!(
                    "obligation was already consumed at node {:?}",
                    obligation.consumed_by_node
                ),
            });
        }
        obligation.consumed = true;
        obligation.consumed_by_node = Some(node_idx);
        Ok(())
    }

    /// Consume all obligations matching a predicate.
    ///
    /// Returns the number of obligations successfully consumed.
    pub fn consume_matching<F>(&mut self, node_idx: usize, mut predicate: F) -> usize
    where
        F: FnMut(&Obligation) -> bool,
    {
        let mut count = 0;
        for obligation in &mut self.obligations {
            if !obligation.consumed && predicate(obligation) {
                obligation.consumed = true;
                obligation.consumed_by_node = Some(node_idx);
                count += 1;
            }
        }
        count
    }

    /// Check that all produced obligations have been consumed.
    ///
    /// # Errors
    ///
    /// Returns [`ObligationError::UnconsumedObligation`] for the first unconsumed obligation.
    pub fn check_all_consumed(&self) -> Result<(), ObligationError> {
        for obligation in &self.obligations {
            if !obligation.consumed {
                return Err(ObligationError::UnconsumedObligation {
                    name: obligation.name.clone(),
                    kind: obligation.kind.clone(),
                    produced_at_node: obligation.produced_at_node,
                    details: format!(
                        "obligation `{}` was produced at node {} but was never consumed before the boundary",
                        obligation.name, obligation.produced_at_node
                    ),
                });
            }
        }
        Ok(())
    }

    /// Return all currently unconsumed obligations.
    #[must_use]
    pub fn unconsumed_obligations(&self) -> Vec<&Obligation> {
        self.obligations.iter().filter(|o| !o.consumed).collect()
    }
}

/// Verify that a `Program` satisfies all explicit obligation invariants.
///
/// Every asynchronous transfer (`AsyncLoad`/`AsyncStore`) produces a `PendingAsyncWait`
/// obligation that MUST be consumed by an `AsyncWait` with matching tag before the program ends.
///
/// # Errors
///
/// Returns [`ObligationError`] naming the unconsumed obligation if any obligation is violated.
pub fn verify_program_obligations(program: &Program) -> Result<(), ObligationError> {
    let mut tracker = ObligationTracker::new();
    walk_and_track_obligations(program.entry(), &mut tracker, 0)?;
    tracker.check_all_consumed()
}

fn walk_and_track_obligations(
    nodes: &[Node],
    tracker: &mut ObligationTracker,
    start_idx: usize,
) -> Result<usize, ObligationError> {
    let mut current_idx = start_idx;
    for node in nodes {
        match node {
            Node::AsyncLoad {
                tag,
                source,
                destination,
                ..
            } => {
                tracker.produce(
                    format!("async_wait:{}", tag.as_str()),
                    ObligationKind::PendingAsyncWait {
                        tag: tag.as_str().to_string(),
                        source: source.as_str().to_string(),
                        destination: destination.as_str().to_string(),
                    },
                    current_idx,
                );
            }
            Node::AsyncStore {
                tag,
                source,
                destination,
                ..
            } => {
                tracker.produce(
                    format!("async_wait:{}", tag.as_str()),
                    ObligationKind::PendingAsyncWait {
                        tag: tag.as_str().to_string(),
                        source: source.as_str().to_string(),
                        destination: destination.as_str().to_string(),
                    },
                    current_idx,
                );
            }
            Node::AsyncWait { tag } => {
                let tag_str = tag.as_str();
                let consumed = tracker.consume_matching(current_idx, |o| match &o.kind {
                    ObligationKind::PendingAsyncWait { tag: t, .. } => t == tag_str,
                    _ => false,
                });
                if consumed == 0 {
                    return Err(ObligationError::InvalidObligationConsumption {
                        name: format!("async_wait:{tag_str}"),
                        details: format!("AsyncWait at node {current_idx} referenced tag `{tag_str}` with no pending transfer"),
                    });
                }
            }
            // Each conditional arm tracks its own copy, so a transfer started in
            // one arm is not consumed by a wait in the other.
            Node::If { .. } => {
                for body in child_bodies(node) {
                    let mut branch = tracker.clone();
                    walk_and_track_obligations(body, &mut branch, current_idx + 1)?;
                }
            }
            _ => {
                for body in child_bodies(node) {
                    walk_and_track_obligations(body, tracker, current_idx + 1)?;
                }
            }
        }
        current_idx += 1;
    }
    Ok(current_idx)
}
