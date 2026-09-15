//! Typed asynchronous transactions and their completion facts.
//!
//! An asynchronous transfer used to be one name shared by an issue and a wait.
//! Nothing else was stated, so a wait could only mean "every transfer issued so
//! far has landed": a bounded stage ring collapsed into a full drain and the
//! overlap the ring existed for was lost. A transaction now states the ring slot
//! it occupies, how wide its result becomes observable, and the fence a
//! subsequent generic read needs, so a target selects a transfer mechanism and a
//! wait form under stated facts instead of inferring both from a name.

use std::fmt;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::{
    AsyncTransaction, AsyncWaitSpec, KernelBody, KernelOpKind, MemoryProxyFence, StageSlot,
    TransactionScope,
};
use crate::Name;

/// Assign every asynchronous transfer a slot of the schedule's stage ring.
///
/// Without a staged ring a wait can only mean "every transfer has landed", so a
/// pipelined phase pays a full drain at each stage boundary and the ring buys
/// nothing. Slots rotate in descriptor order, which is the order both ends of
/// the pairing are walked in, so the same descriptor always stages the same
/// way. A transfer's wait takes the slot its issue was given; a wait with no
/// issue is left alone for the verifier to report.
pub(crate) fn stage_transactions(body: &mut KernelBody, ring_slots: u16) {
    if ring_slots == 0 {
        return;
    }
    let mut slots: FxHashMap<Name, StageSlot> = FxHashMap::default();
    let mut next = 0u16;
    assign_issues(body, ring_slots, &mut next, &mut slots);
    assign_waits(body, &slots);
}

fn assign_issues(
    body: &mut KernelBody,
    ring_slots: u16,
    next: &mut u16,
    slots: &mut FxHashMap<Name, StageSlot>,
) {
    for op in &mut body.ops {
        let (KernelOpKind::AsyncLoad(transaction) | KernelOpKind::AsyncStore(transaction)) =
            &mut op.kind
        else {
            continue;
        };
        let slot = StageSlot::new(*next % ring_slots, ring_slots);
        *next = next.wrapping_add(1);
        transaction.stage = Some(slot);
        slots.insert(transaction.tag.clone(), slot);
    }
    for child in &mut body.child_bodies {
        assign_issues(child, ring_slots, next, slots);
    }
}

fn assign_waits(body: &mut KernelBody, slots: &FxHashMap<Name, StageSlot>) {
    for op in &mut body.ops {
        let KernelOpKind::AsyncWait(wait) = &mut op.kind else {
            continue;
        };
        if let Some(slot) = slots.get(&wait.transaction.tag) {
            wait.transaction.stage = Some(*slot);
        }
    }
    for child in &mut body.child_bodies {
        assign_waits(child, slots);
    }
}

impl KernelOpKind {
    /// Issue a workgroup-collective global-to-shared transfer with no ring
    /// slot.
    ///
    /// This is the form semantic IR lowers to: a schedule that stages the
    /// transfer assigns the slot afterwards. A caller declaring a narrower
    /// visibility or a slot of its own builds the transaction directly.
    #[must_use]
    pub fn async_load(tag: Name) -> Self {
        Self::AsyncLoad(Box::new(AsyncTransaction::unstaged(
            tag,
            TransactionScope::Workgroup,
        )))
    }

    /// Issue a workgroup-collective shared-to-global transfer with no ring
    /// slot.
    #[must_use]
    pub fn async_store(tag: Name) -> Self {
        Self::AsyncStore(Box::new(AsyncTransaction::unstaged(
            tag,
            TransactionScope::Workgroup,
        )))
    }

    /// Wait for a workgroup-collective transfer behind the workgroup fence its
    /// visibility requires.
    #[must_use]
    pub fn async_wait(tag: Name) -> Self {
        Self::AsyncWait(Box::new(AsyncWaitSpec::new(AsyncTransaction::unstaged(
            tag,
            TransactionScope::Workgroup,
        ))))
    }
}

/// Reason a declared transaction cannot be carried to a target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AsyncTransactionError {
    /// The issue and its wait pair through the tag, so an empty one pairs
    /// nothing.
    EmptyTag,
    /// A staged transaction declared a ring with no slots.
    ZeroRing,
    /// The declared slot is outside its own ring.
    SlotOutOfRing {
        /// Declared slot.
        slot: u16,
        /// Declared ring depth.
        ring_slots: u16,
    },
    /// The wait fence is narrower than the visibility the transaction claims,
    /// so a reader inside that scope can observe the destination before the
    /// transfer's writes.
    FenceNarrowerThanVisibility {
        /// Declared fence.
        fence: MemoryProxyFence,
        /// Declared visibility.
        visibility: TransactionScope,
    },
}

impl TransactionScope {
    /// Narrowest fence that makes a completed transfer readable at this scope.
    ///
    /// Subgroup visibility resolves to the workgroup fence: the descriptor has
    /// no narrower fence, and a wider one is always sound.
    #[must_use]
    pub const fn minimum_fence(self) -> MemoryProxyFence {
        match self {
            Self::Invocation => MemoryProxyFence::None,
            Self::Subgroup | Self::Workgroup => MemoryProxyFence::Workgroup,
            Self::Device => MemoryProxyFence::Device,
        }
    }

    /// Ordering rank, widest last.
    const fn rank(self) -> u8 {
        match self {
            Self::Invocation => 0,
            Self::Subgroup => 1,
            Self::Workgroup => 2,
            Self::Device => 3,
        }
    }

    /// Whether this scope covers every invocation `other` covers.
    #[must_use]
    pub const fn covers(self, other: Self) -> bool {
        self.rank() >= other.rank()
    }
}

impl MemoryProxyFence {
    /// Ordering rank, widest last.
    const fn rank(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Workgroup => 1,
            Self::Device => 2,
        }
    }

    /// Whether this fence orders at least as widely as `required`.
    #[must_use]
    pub const fn covers(self, required: Self) -> bool {
        self.rank() >= required.rank()
    }
}

impl StageSlot {
    /// Declare slot `slot` of a `ring_slots`-deep ring.
    #[must_use]
    pub const fn new(slot: u16, ring_slots: u16) -> Self {
        Self { slot, ring_slots }
    }

    /// Slot this transaction occupies.
    #[must_use]
    pub const fn slot(&self) -> u16 {
        self.slot
    }

    /// Depth of the ring the slot belongs to.
    #[must_use]
    pub const fn ring_slots(&self) -> u16 {
        self.ring_slots
    }
}

impl AsyncTransaction {
    /// Declare a transfer that is not staged: a wait on it admits no other
    /// transfer of the same body still being in flight.
    #[must_use]
    pub fn unstaged(tag: Name, visibility: TransactionScope) -> Self {
        Self {
            tag,
            visibility,
            stage: None,
        }
    }

    /// Declare a transfer occupying one slot of a bounded stage ring.
    #[must_use]
    pub fn staged(tag: Name, visibility: TransactionScope, stage: StageSlot) -> Self {
        Self {
            tag,
            visibility,
            stage: Some(stage),
        }
    }

    /// Tag pairing this transaction with its wait.
    #[must_use]
    pub fn tag(&self) -> &Name {
        &self.tag
    }

    /// Widest set of invocations that observes the completed transfer.
    #[must_use]
    pub const fn visibility(&self) -> TransactionScope {
        self.visibility
    }

    /// Ring slot the transfer occupies, absent when the schedule did not stage
    /// it.
    #[must_use]
    pub const fn stage(&self) -> Option<StageSlot> {
        self.stage
    }

    /// Whether a schedule staged this transfer into a bounded ring.
    #[must_use]
    pub const fn is_staged(&self) -> bool {
        self.stage.is_some()
    }

    /// Transfers of the same ring that may still be incomplete when a wait on
    /// this transaction returns.
    ///
    /// A wait on slot `s` of a `d`-slot ring is the point where slot `s` is
    /// consumed; the remaining `d - 1` slots are what the ring buys, and a
    /// target that drains all of them has emitted a correct kernel with no
    /// overlap. An unstaged transfer admits nothing in flight.
    #[must_use]
    pub const fn in_flight_allowed(&self) -> u16 {
        match self.stage {
            Some(stage) => stage.ring_slots.saturating_sub(1),
            None => 0,
        }
    }

    /// Whether both transactions name the same ring slot of the same transfer.
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        self.tag == other.tag && self.stage == other.stage
    }

    /// Check the facts a target is allowed to rely on.
    ///
    /// # Errors
    ///
    /// Returns an error when the tag pairs nothing or the declared slot is not
    /// a slot of its own ring.
    pub fn validate(&self) -> Result<(), AsyncTransactionError> {
        if self.tag.is_empty() {
            return Err(AsyncTransactionError::EmptyTag);
        }
        if let Some(stage) = self.stage {
            if stage.ring_slots == 0 {
                return Err(AsyncTransactionError::ZeroRing);
            }
            if stage.slot >= stage.ring_slots {
                return Err(AsyncTransactionError::SlotOutOfRing {
                    slot: stage.slot,
                    ring_slots: stage.ring_slots,
                });
            }
        }
        Ok(())
    }
}

impl AsyncWaitSpec {
    /// Wait for `transaction` behind the narrowest fence its visibility needs.
    #[must_use]
    pub fn new(transaction: AsyncTransaction) -> Self {
        let fence = transaction.visibility().minimum_fence();
        Self { transaction, fence }
    }

    /// Wait for `transaction` behind an explicit fence.
    #[must_use]
    pub fn fenced(transaction: AsyncTransaction, fence: MemoryProxyFence) -> Self {
        Self { transaction, fence }
    }

    /// Transaction this wait completes.
    #[must_use]
    pub fn transaction(&self) -> &AsyncTransaction {
        &self.transaction
    }

    /// Fence a generic-proxy read needs after the transfer lands.
    #[must_use]
    pub const fn fence(&self) -> MemoryProxyFence {
        self.fence
    }

    /// Check the wait against the transaction it completes.
    ///
    /// # Errors
    ///
    /// Returns an error when the transaction does not check or the fence orders
    /// narrower than the visibility the transaction claims.
    pub fn validate(&self) -> Result<(), AsyncTransactionError> {
        self.transaction.validate()?;
        let required = self.transaction.visibility().minimum_fence();
        if !self.fence.covers(required) {
            return Err(AsyncTransactionError::FenceNarrowerThanVisibility {
                fence: self.fence,
                visibility: self.transaction.visibility(),
            });
        }
        Ok(())
    }
}

impl fmt::Display for TransactionScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Invocation => "invocation",
            Self::Subgroup => "subgroup",
            Self::Workgroup => "workgroup",
            Self::Device => "device",
        };
        f.write_str(text)
    }
}

impl fmt::Display for MemoryProxyFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::None => "no fence",
            Self::Workgroup => "workgroup fence",
            Self::Device => "device fence",
        };
        f.write_str(text)
    }
}

impl fmt::Display for AsyncTransactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTag => f.write_str("asynchronous transaction tag is empty"),
            Self::ZeroRing => f.write_str("staged transaction declares a ring with no slots"),
            Self::SlotOutOfRing { slot, ring_slots } => write!(
                f,
                "transaction declares slot {slot} of a {ring_slots}-slot ring"
            ),
            Self::FenceNarrowerThanVisibility { fence, visibility } => write!(
                f,
                "{fence} does not order a transfer observed at {visibility} scope"
            ),
        }
    }
}

impl std::error::Error for AsyncTransactionError {}
