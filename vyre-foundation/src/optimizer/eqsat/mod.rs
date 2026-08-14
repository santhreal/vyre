//! Equality-saturation engine  -  minimal `EGraph` substrate for vyre IR
//! algebraic rewrite families.
//!
//! Op id: `vyre-foundation::optimizer::eqsat`. Soundness: every equivalence
//! added to the `EGraph` must be a true semantic equality of the underlying
//! IR. Cost-direction: extraction phase picks the lowest-cost equivalent
//! representative under a caller-supplied cost function  -  guaranteed
//! cost-monotone-down by construction.
//!
//! ## Why
//!
//! Pass-by-pass rewriting commits to a single rewrite at every step. When
//! two passes both want to fire on the same expression, one wins
//! (whichever is scheduled first), even if the other would have unlocked
//! a much better optimization downstream. Equality saturation sidesteps
//! this by accumulating all known equivalences into one `EGraph`, running
//! every rewrite rule to a fixed point, and then extracting the
//! lowest-cost equivalent at the end.
//!
//! This module ships the substrate: a minimal but sound `EGraph` with
//! hashcons, union-find, rebuild, saturation, and a `Family` trait
//! that wraps a set of related rewrite rules.
//!
//! ## `ENode`
//!
//! `ENodes` are domain-specific: each family defines its own `ENode` enum.
//! The substrate is generic over `Lang: ENodeLang` which provides the
//! children-iteration API the `EGraph` needs to canonicalize and rebuild.
//!
//! ## Why not import egg
//!
//! This implementation is intentionally minimal so it lives entirely
//! within `vyre-foundation` with no external dep, no proc-macro, and
//! no per-rule code generation. The egg crate is more featureful but
//! adds a dependency tree that conflicts with vyre's "every dep is a
//! supply-chain risk" stance.
//!
//! [`EGraph`] behavior lives in `egraph`, its dense-index and reserve
//! bookkeeping in `class_index`, the rewrite loop in `saturation`, the
//! device-fact rule adapter in `device_aware_rule`, and cost extraction in
//! `extraction`. The public types stay declared here so their rendered
//! documentation paths do not move.

use std::error::Error as StdError;
use std::fmt;
use std::hash::Hash;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

mod class_index;
mod device_aware_rule;
mod egraph;
mod extraction;
mod saturation;

#[cfg(test)]
mod arith_fixture;

pub use extraction::{extract_best, try_extract_best, try_extract_best_with_budget};
pub use saturation::{
    saturate, saturate_per_family, saturate_per_family_detailed, saturate_with_report,
    try_saturate, try_saturate_named, try_saturate_per_family, try_saturate_per_family_detailed,
    try_saturate_with_report,
};

/// Default extraction fixed-point iteration budget.
pub const DEFAULT_EXTRACTION_ITER_BUDGET: usize = 1024;

/// Stack-backed child list used by `EGraph` node APIs. Most IR algebra nodes
/// have 0-3 children; keeping that path inline avoids allocator traffic during
/// saturation.
pub type EChildren = SmallVec<[EClassId; 4]>;

/// Identifier of an `EClass` in the `EGraph`. `EClasses` are dense u32-indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EClassId(pub u32);

/// Domain-specific `ENode` language. Implementations describe how to
/// iterate the children of a node (for canonicalization) and how to
/// rebuild a node with replacement child ids (for rebuild).
pub trait ENodeLang: Clone + Eq + Hash {
    /// Iterate the `EClass`-child ids referenced by this node, in order.
    fn children(&self) -> EChildren;

    /// Rebuild this node with replacement `EClass` children. The returned
    /// node has the same shape as `self` but with each child replaced by
    /// the corresponding entry in `children`. `children.len()` must equal
    /// `self.children().len()`.
    #[must_use]
    fn with_children(&self, children: &[EClassId]) -> Self;
}

/// One equivalence class  -  the set of all `ENodes` proven equal so far.
#[derive(Debug, Clone)]
pub struct EClass<L: ENodeLang> {
    /// Every `ENode` that lives in this class (canonicalized form).
    pub nodes: Vec<L>,
    /// `EClasses` that have THIS one as a child  -  used during rebuild to
    /// propagate canonicalization.
    pub parents: Vec<EClassId>,
}

/// The `EGraph`: a union-find of `EClasses` + a hashcons mapping
/// canonicalized `ENodes` to their `EClass`.
#[derive(Debug, Clone)]
pub struct EGraph<L: ENodeLang> {
    /// Class storage (dense). The class at index `i` is `EClass(i)`.
    classes: Vec<EClass<L>>,
    /// Hashcons: canonicalized `ENode` → `EClassId`. Maintained incrementally
    /// by `add()` and rebuilt after `union()` operations.
    hashcons: FxHashMap<L, EClassId>,
    /// Union-find parent pointers for path-compression find.
    parent: Vec<EClassId>,
    /// Set of `EClasses` that need rebuild after a union  -  drained by
    /// `rebuild()`.
    pending: Vec<EClassId>,
}

/// E-graph construction, indexing, and staging failure.
///
/// Equality saturation is optimizer infrastructure, so allocator pressure and
/// class-id overflow must be explicit errors on the fallible APIs rather than
/// latent panics or poisoned sentinel ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EGraphError {
    /// A fallible staging/allocation reservation failed.
    Capacity {
        /// Operation reserving memory.
        context: &'static str,
        /// Additional elements/slots requested.
        requested: usize,
        /// Allocator error rendered with platform-specific detail.
        source: String,
    },
    /// Dense class storage exceeded the public `u32` id space.
    ClassIdOverflow {
        /// Dense class index that could not be represented as [`EClassId`].
        index: usize,
    },
    /// A caller supplied an `EClassId` outside the current dense tables.
    ClassIdOutOfBounds {
        /// Operation resolving the id.
        context: &'static str,
        /// Invalid id.
        id: EClassId,
        /// Current table length.
        len: usize,
    },
}

impl fmt::Display for EGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity {
                context,
                requested,
                source,
            } => write!(
                f,
                "{context} could not reserve {requested} additional slots: {source}. Fix: lower the saturation batch size or split the optimizer workload."
            ),
            Self::ClassIdOverflow { index } => write!(
                f,
                "egraph class index {index} exceeds the u32 EClassId space. Fix: split the egraph or extract before adding more classes."
            ),
            Self::ClassIdOutOfBounds { context, id, len } => write!(
                f,
                "{context} referenced eclass id {} but only {len} class slots exist. Fix: pass ids returned by this EGraph instance.",
                id.0
            ),
        }
    }
}

impl StdError for EGraphError {}

pub(crate) fn log_egraph_compat_error(context: &'static str, error: &EGraphError) {
    tracing::error!(
        context,
        error = %error,
        "legacy infallible egraph API failed; use the matching try_* API to handle this condition explicitly"
    );
}

/// One equality-saturation rewrite rule. Returns a list of `(left, right)`
/// `EClass` pairs that should be unioned after the rule fires.
///
/// Implementations walk the `EGraph` (via `iter_nodes`), pattern-match on
/// shapes they recognize, and return the equivalences they want to add.
pub trait Rule<L: ENodeLang> {
    /// Human-readable rule name for telemetry + tests.
    fn name(&self) -> &'static str;

    /// Find every match of this rule's LHS pattern in `egraph` and return
    /// the (a, b) pairs that should be equated.
    fn matches(&self, egraph: &EGraph<L>) -> Vec<(EClassId, EClassId)>;
}

/// A family of related rewrite rules.
pub trait Family<L: ENodeLang> {
    /// Family name (e.g. "`commutative_arith`").
    fn name(&self) -> &'static str;

    /// Vec of rules in this family. Stored as boxed trait objects so a
    /// single family can mix rule shapes (literal-matching, pattern-
    /// matching, conditional rewrites).
    fn rules(&self) -> Vec<Box<dyn Rule<L>>>;
}

/// Reason an equality-saturation run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaturationStopReason {
    /// No rule set was supplied.
    EmptyRuleSet,
    /// The caller supplied a zero-iteration cap.
    ZeroBudget,
    /// A rule scan produced no more equivalences.
    FixedPoint,
    /// The run consumed the supplied iteration cap while matches were still
    /// being produced.
    IterationBudget,
}

/// Executable telemetry for one equality-saturation run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaturationReport {
    /// Rewrite family label for this run. Raw rule-slice calls use `global`;
    /// per-family calls use [`Family::name`].
    pub rewrite_family: &'static str,
    /// Number of rules scanned each iteration.
    pub rule_count: usize,
    /// Iterations actually executed.
    pub iters_used: usize,
    /// Caller-supplied iteration budget.
    pub budget: usize,
    /// Why the run stopped.
    pub stop_reason: SaturationStopReason,
    /// Dense class slots before rule application.
    pub class_count_before: usize,
    /// Dense class slots after rule application.
    pub class_count_after: usize,
    /// Equivalence pairs returned by rules and handed to union.
    pub applied_equivalences: usize,
    /// Extra unions discovered by rebuild propagation.
    pub rebuild_unions: usize,
}

/// Adapter that gates a base [`Rule`] on a device-fact predicate.
///
/// ROADMAP A9. The "should this rule fire on this hardware?" check
/// recurs across every device-aware Rule (FP16 only on `supports_f16`,
/// tensor-core fusion only on `supports_tensor_cores`, subgroup
/// shuffle only on `has_subgroup_shuffle`). Without a shared adapter,
/// every Rule re-implements the same `if !facts.feature { return
/// vec![] }` preamble. This wrapper centralises it.
///
/// `DeviceFacts` is a free-form caller-owned object so the foundation
/// crate does not pull `DeviceProfile` (which lives in `vyre-driver`)
/// into its dependency graph. Callers either pass a borrowed
/// `&DeviceProfile` directly via the `predicate` closure capture, or
/// thread a snapshot through their own type.
///
/// When `predicate` returns `false` the wrapped rule's [`matches`]
/// short-circuits to an empty vector  -  the saturation loop sees no
/// equivalences and the rule contributes nothing. When `true`, the
/// wrapped rule fires unchanged.
pub struct DeviceAwareRule<L: ENodeLang, F: Fn() -> bool> {
    inner: Box<dyn Rule<L>>,
    predicate: F,
}

/// One family's saturation result: how many iterations were spent in
/// that family's [`saturate`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilySaturationReport {
    /// Family name as returned by [`Family::name`].
    pub family: &'static str,
    /// Iterations the family actually used (≤ `budget`). 0 when the
    /// budget was 0 or when the rule set converged immediately.
    pub iters_used: usize,
    /// Budget the family was given. Echoed back so callers can compare
    /// against `iters_used` without re-querying the budget function.
    pub budget: usize,
}

/// Detailed per-family saturation telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilySaturationTelemetry {
    /// Family name as returned by [`Family::name`].
    pub family: &'static str,
    /// Full saturation report for this family run.
    pub saturation: SaturationReport,
}

/// Reason an extraction run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionStopReason {
    /// The extraction cost table reached a fixed point.
    FixedPoint,
    /// The extraction loop consumed the supplied iteration cap.
    IterationBudget,
    /// The root class remained uncosted, usually because the represented term
    /// is cyclic or depends on an uncosted child class.
    MissingCost,
}

/// Executable telemetry for one extraction run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionReport<L: ENodeLang> {
    /// Root class requested by the caller.
    pub class_id: EClassId,
    /// Best node and computed cost when extraction produced a candidate.
    pub best: Option<(L, u64)>,
    /// Iterations actually executed.
    pub iters_used: usize,
    /// Caller-supplied extraction iteration budget.
    pub budget: usize,
    /// Why extraction stopped.
    pub stop_reason: ExtractionStopReason,
    /// Dense class slots visible to extraction.
    pub class_count: usize,
}
