//! Declared contract for every pass this crate registers.
//!
//! One row per registered pass. The closure test compares this set against
//! `registered_pass_registrations()` in both directions, so a new pass is red
//! until its contract is recorded here and a removed pass is red until its row
//! goes with it.
//!
//! A `Structural` witness carries the argument that authorizes the rewrite. It
//! is stated here rather than only in the pass module, because the registry is
//! what candidate search and the closure test read.

use super::{
    BoundedExpansion::{self, NodeBudget, NodeFactor, NonGrowing},
    NumericalContract::{BitExact, FloatContraction, IntegerWrapping},
    ProfitabilityFact::{
        self, Canonicalizes, EnablesFusion, RaisesOccupancy, ReducesSynchronization,
        RemovesLaunches, RemovesNodes, RemovesTraffic, ShortensDependence, WidensVector,
    },
    RewriteContract,
    RewriteEffect::{
        self, Allocation, Atomic, BufferAbi, ControlFlow, Reads, Synchronization, Writes,
    },
    RewritePrecondition::{
        self, AbiPreserved, BoundedIndices, ConstantLoopBounds, DisjointBuffers, EffectFreeRegion,
        IntegerElements, LiteralOperands, SingleReachingDefinition, SynchronizationFreeRegion,
    },
    RewriteWitness::{Obligation, Structural},
};
use vyre_spec::IrLevel::{Logical, Schedule};

/// Obligation families discharged by the solver gate, matching the `family`
/// column of `algebraic_rules::arithmetic_rewrite_proof_contracts`.
/// `const_fold` fires the literal-folding ids and the integer identity
/// eliminations in `const_fold/binop_identities.rs`.
const CONST_FOLD_FAMILY: &[&str] = &["const_fold", "identity_elim"];
const CANONICALIZE_FAMILY: &[&str] = &["canonicalize"];
const STRENGTH_REDUCE_FAMILY: &[&str] = &["strength_reduce"];

const fn contract(
    pass: &'static str,
    level: vyre_spec::IrLevel,
    preconditions: &'static [RewritePrecondition],
    effects: &'static [RewriteEffect],
    numerical: super::NumericalContract,
    witness: super::RewriteWitness,
    profitability: &'static [ProfitabilityFact],
    expansion: BoundedExpansion,
) -> RewriteContract {
    RewriteContract {
        pass,
        level,
        preconditions,
        effects,
        numerical,
        witness,
        profitability,
        expansion,
    }
}

/// Every contract this crate declares.
pub(super) const SHIPPED_CONTRACTS: &[RewriteContract] = &[
    contract(
        "atomic_minimize",
        Schedule,
        &[SynchronizationFreeRegion],
        &[Atomic, Reads],
        BitExact,
        Structural(
            "an identity-op read-modify-write under relaxed ordering observes the same value a plain load does, and non-identity atomics are left alone",
        ),
        &[ReducesSynchronization],
        NonGrowing,
    ),
    contract(
        "canonicalize",
        Logical,
        &[],
        &[],
        BitExact,
        Obligation(CANONICALIZE_FAMILY),
        &[Canonicalizes, EnablesFusion],
        NonGrowing,
    ),
    contract(
        "const_fold",
        Logical,
        &[LiteralOperands],
        &[],
        BitExact,
        Obligation(CONST_FOLD_FAMILY),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "reaching_def_propagate",
        Logical,
        &[LiteralOperands, SingleReachingDefinition],
        &[],
        BitExact,
        Structural(
            "a let bound to a literal and never rebound has that literal as its only reaching definition, so every use reads the same value",
        ),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "normalize_atomics",
        Schedule,
        &[],
        &[Atomic, ControlFlow],
        BitExact,
        Structural(
            "binding an atomic to a let before the branch evaluates it exactly once, which is what an atomic in a branch predicate already means",
        ),
        &[Canonicalizes],
        NodeBudget(2),
    ),
    contract(
        "strength_reduce",
        Logical,
        &[IntegerElements],
        &[],
        FloatContraction,
        Obligation(STRENGTH_REDUCE_FAMILY),
        &[RemovesNodes, ShortensDependence],
        NonGrowing,
    ),
    contract(
        "branch_coalesce",
        Logical,
        &[],
        &[ControlFlow],
        BitExact,
        Structural(
            "an if whose body is exactly one guardless if runs its inner body under the conjunction of both conditions and nothing else",
        ),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "branch_value_hoist",
        Logical,
        &[EffectFreeRegion],
        &[ControlFlow],
        BitExact,
        Structural(
            "a prefix common to both arms runs on every path through the branch, so evaluating it once before the branch preserves order and effects",
        ),
        &[RemovesNodes, ShortensDependence],
        NonGrowing,
    ),
    contract(
        "buffer_decl_sort",
        Logical,
        &[AbiPreserved],
        &[BufferAbi],
        BitExact,
        Structural(
            "buffer references resolve by name, so declaration order is not observable and a canonical order makes structurally equal programs compare equal",
        ),
        &[Canonicalizes],
        NonGrowing,
    ),
    contract(
        "empty_block_collapse",
        Logical,
        &[],
        &[],
        BitExact,
        Structural("an empty block performs no effect and binds no name"),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "if_constant_branch_eliminate",
        Logical,
        &[LiteralOperands],
        &[ControlFlow],
        BitExact,
        Structural(
            "a constant condition selects one arm on every invocation, so inlining that arm keeps the only reachable path",
        ),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "noop_assign_eliminate",
        Logical,
        &[],
        &[],
        BitExact,
        Structural("assigning a variable to itself leaves the same value bound"),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "region_fusion_hint",
        Logical,
        &[DisjointBuffers],
        &[],
        BitExact,
        Structural(
            "the hint records that two adjacent regions match a fusion rule; it changes no statement and the fusion pass owns the rewrite",
        ),
        &[EnablesFusion],
        NonGrowing,
    ),
    contract(
        "region_inline",
        Logical,
        &[],
        &[],
        BitExact,
        Structural(
            "a region is a naming scope over its body, so splicing the body into the parent keeps statement order and every binding it declares",
        ),
        &[RemovesNodes, EnablesFusion],
        NonGrowing,
    ),
    contract(
        "region_promote_singleton_block",
        Logical,
        &[],
        &[],
        BitExact,
        Structural("a region whose body is one block has the block's statements as its body"),
        &[Canonicalizes],
        NonGrowing,
    ),
    contract(
        "rematerialize_cheap_let",
        Logical,
        &[SingleReachingDefinition],
        &[],
        BitExact,
        Structural(
            "a let bound to a leaf expression with no effect evaluates to the same value at every use, so inlining it and dropping the binding preserves values",
        ),
        &[RemovesNodes, RaisesOccupancy],
        NodeFactor(2),
    ),
    contract(
        "tail_duplication",
        Logical,
        &[EffectFreeRegion],
        &[ControlFlow],
        BitExact,
        Structural(
            "a tail common to both arms runs on every path out of the branch, so hoisting it after the branch keeps order and effects",
        ),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "cse",
        Logical,
        &[EffectFreeRegion, SingleReachingDefinition],
        &[],
        BitExact,
        Structural(
            "two structurally equal expressions with no intervening write to what they read evaluate to the same value, so one binding serves both",
        ),
        &[RemovesNodes, ShortensDependence],
        NodeBudget(1),
    ),
    contract(
        "dce",
        Logical,
        &[EffectFreeRegion],
        &[],
        BitExact,
        Structural(
            "a binding no reachable use reads, and a statement with no effect, contribute nothing observable",
        ),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "fusion",
        Logical,
        &[DisjointBuffers],
        &[Reads, Writes, Allocation],
        BitExact,
        Structural(
            "two regions whose written buffers are disjoint from what the other reads commute, so running them in one region preserves every read-write order",
        ),
        &[RemovesLaunches, RemovesTraffic],
        NodeFactor(2),
    ),
    contract(
        "loop_bound_tighten",
        Logical,
        &[BoundedIndices],
        &[ControlFlow],
        BitExact,
        Structural(
            "iterations the inner guard rejects perform no effect, so narrowing the bound to the guarded range drops only empty iterations",
        ),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "loop_fission",
        Logical,
        &[DisjointBuffers],
        &[Reads, Writes],
        BitExact,
        Structural(
            "when the two halves of a body touch disjoint buffers, no value crosses the split, so two loops over the same range compute the same state",
        ),
        &[EnablesFusion, WidensVector],
        NodeFactor(2),
    ),
    contract(
        "loop_fusion",
        Logical,
        &[DisjointBuffers],
        &[Reads, Writes],
        BitExact,
        Structural(
            "adjacent loops with equal bounds and disjoint buffer sets have no cross-iteration dependence, so one loop runs both bodies in order",
        ),
        &[RemovesNodes, RemovesTraffic],
        NonGrowing,
    ),
    contract(
        "loop_licm",
        Logical,
        &[EffectFreeRegion],
        &[],
        BitExact,
        Structural(
            "a binding whose value depends on no loop variable and no written buffer evaluates identically in every iteration",
        ),
        &[RemovesNodes, ShortensDependence],
        NodeBudget(2),
    ),
    contract(
        "loop_lower_bound_normalize",
        Logical,
        &[ConstantLoopBounds],
        &[],
        IntegerWrapping,
        Structural(
            "shifting a literal-bounded range to start at zero and adding the offset at every use enumerates the same values in the same order",
        ),
        &[Canonicalizes],
        NodeFactor(2),
    ),
    contract(
        "loop_peel",
        Logical,
        &[ConstantLoopBounds],
        &[ControlFlow],
        BitExact,
        Structural(
            "the first iteration is the only one the first-iteration guard admits, so running it before the loop and starting at the next value keeps every effect once",
        ),
        &[RemovesNodes],
        NodeFactor(2),
    ),
    contract(
        "loop_redundant_bound_check_elide",
        Logical,
        &[ConstantLoopBounds],
        &[ControlFlow],
        BitExact,
        Structural(
            "the enclosing loop already admits only values below its upper bound, so a guard re-checking that bound is true on every iteration",
        ),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "loop_software_pipeline",
        Schedule,
        &[ConstantLoopBounds, DisjointBuffers],
        &[Reads, Writes],
        BitExact,
        Structural(
            "the prologue issues the first load and the body stores iteration n while loading n+1, so every load still precedes the store that reads it",
        ),
        &[ShortensDependence, RaisesOccupancy],
        NodeFactor(3),
    ),
    contract(
        "loop_strip_mine",
        Schedule,
        &[ConstantLoopBounds],
        &[],
        BitExact,
        Structural(
            "the tiled pair enumerates the original range in the original order, with the residual tail kept as its own trip range",
        ),
        &[WidensVector, RaisesOccupancy],
        NodeFactor(3),
    ),
    contract(
        "loop_trip_zero_eliminate",
        Logical,
        &[ConstantLoopBounds],
        &[ControlFlow],
        BitExact,
        Structural("a loop whose literal bounds give no iteration performs no effect"),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "loop_unroll",
        Schedule,
        &[ConstantLoopBounds],
        &[],
        BitExact,
        Structural(
            "each copy substitutes one literal loop value, so the copies perform exactly the iterations the loop performed, in order",
        ),
        &[ShortensDependence, WidensVector],
        NodeFactor(8),
    ),
    contract(
        "loop_var_range_fold",
        Logical,
        &[ConstantLoopBounds],
        &[],
        IntegerWrapping,
        Structural(
            "inside a literal-bounded loop the induction variable is known to lie in that range, so a cast or bound check over that range folds to its constant answer",
        ),
        &[RemovesNodes],
        NonGrowing,
    ),
    contract(
        "dead_buffer_elim",
        Logical,
        &[AbiPreserved],
        &[BufferAbi, Allocation],
        BitExact,
        Structural(
            "a declared buffer no statement reads or writes, and which the entry ABI does not publish, carries no value out of the program",
        ),
        &[RemovesTraffic],
        NonGrowing,
    ),
    contract(
        "dead_store_elim",
        Logical,
        &[DisjointBuffers],
        &[Writes],
        BitExact,
        Structural(
            "a store overwritten by a later sibling store to the same buffer and index, with no effect between them that could read it, is never observed",
        ),
        &[RemovesTraffic],
        NonGrowing,
    ),
    contract(
        "decode_scan_fuse",
        Schedule,
        &[DisjointBuffers],
        &[Reads, Writes, Allocation],
        BitExact,
        Structural(
            "the scan reads exactly the buffer the decode wrote, so computing each element and consuming it in one region keeps the same values and drops the round trip",
        ),
        &[RemovesLaunches, RemovesTraffic],
        NodeFactor(2),
    ),
    contract(
        "read_only_load_hoist",
        Logical,
        &[DisjointBuffers],
        &[Reads],
        BitExact,
        Structural(
            "both arms load the same index of a buffer nothing in the branch writes, so one load before the branch yields the value both arms read",
        ),
        &[RemovesTraffic, ShortensDependence],
        NonGrowing,
    ),
    contract(
        "store_to_load_forward",
        Logical,
        &[DisjointBuffers, SingleReachingDefinition],
        &[Reads, Writes],
        BitExact,
        Structural(
            "a load from the same buffer and structurally equal index in the same block, with no intervening write, reads the value the store wrote",
        ),
        &[RemovesTraffic],
        NonGrowing,
    ),
    contract(
        "vectorization",
        Schedule,
        &[DisjointBuffers, BoundedIndices],
        &[Reads, Writes],
        BitExact,
        Structural(
            "contiguous element accesses within one invocation cover the same addresses as the wide access that replaces them",
        ),
        &[WidensVector, RemovesTraffic],
        NonGrowing,
    ),
    contract(
        "autotune",
        Schedule,
        &[],
        &[],
        BitExact,
        Structural(
            "the pass selects among schedule alternatives the target admits and changes no statement's value",
        ),
        &[RaisesOccupancy],
        NodeFactor(2),
    ),
    contract(
        "barrier_coalesce",
        Schedule,
        &[],
        &[Synchronization],
        BitExact,
        Structural(
            "consecutive barriers with no statement between them order the same accesses as the single barrier that joins their memory scopes",
        ),
        &[ReducesSynchronization],
        NonGrowing,
    ),
];
