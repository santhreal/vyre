//! IR transformation passes.
//!
//! Before a `Program` is lowered to backend code, it runs through a series
//! of target-independent optimizations and transformations: call inlining,
//! common-subexpression elimination, dead-code elimination, and visitor
//! utilities. These passes are the vyre analogue of LLVM's mid-level IR
//! passes.

use crate::ir::Program;

/// Call inlining transforms.
///
/// This pass expands `Expr::Call` nodes into the callee's IR body,
/// eliminating kernel-dispatch overhead for small compositional ops.
pub mod inline;

/// Constant propagation over a scope.
///
/// Substitutes let-bound literals into their uses and folds the resulting
/// integer arithmetic.
pub mod const_prop;

/// Dead-branch elimination.
///
/// Collapses an `If` whose condition folded to a literal, and drops a branch
/// whose body neither mutates memory nor calls an opaque extension.
pub mod dead_branch;

/// Loop-invariant code motion.
///
/// The resident pipeline's binding of the loop LICM pass, which owns the rule.
pub mod licm;

/// Shared-nothing parallel dispatch analysis.
pub mod parallelism;

/// The one structural `Node` rewrite.
///
/// Substitution, fusion alpha-renaming, cache-key canonicalization, and the
/// pass engine's encoded-order rewrite all drive this walk instead of carrying
/// their own per-variant match.
pub mod rewrite_walk;

/// Induction-variable substitution shared by the optimizer loop passes and the
/// autodiff loop arm. One complete `var -> expr` rewrite over the whole IR.
pub(crate) mod subst;

/// Reverse-mode automatic differentiation via IR transform (RFC 0002).
///
/// Given a forward `Program` + output/input buffer names, emits a backward
/// `Program` computing gradients via the chain rule.
pub mod autodiff;

/// Collective communication rewrites shared by reference and GPU backends.
pub mod collectives;

/// Cutting a whole-grid fence into sequential dispatch segments.
///
/// One owner for the `Node::Barrier { GridSync }` walk, hoist, and segmentation,
/// shared by the compile-time planner cut and the dispatch-time split.
pub mod grid_sync_split;

/// Contract classification for foundation transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum TransformContractClass {
    /// Required for backend execution and legal hardware representation.
    RequiredLegalization,
    /// Target-neutral algebraic and structural optimization.
    CanonicalOptimization,
    /// Explicitly requested by callers (e.g. autodiff).
    CallerRequestedTransform,
    /// Reusable structural traversal and AST walk mechanics.
    SharedStructuralWalk,
    /// Read-only inspection producing semantic facts without modifying IR.
    Analysis,
}

/// Descriptor documenting a foundation transformation's contract and ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransformDescriptor {
    /// Authoritative module or pass name.
    pub name: &'static str,
    /// Contract classification.
    pub class: TransformContractClass,
    /// Contract purpose and invariants.
    pub description: &'static str,
}

/// Authoritative classification catalog for all foundation transform modules.
pub const FOUNDATION_TRANSFORM_CLASSIFICATIONS: &[TransformDescriptor] = &[
    TransformDescriptor {
        name: "inline",
        class: TransformContractClass::RequiredLegalization,
        description: "Inlines compositional Expr::Call operations into callee bodies",
    },
    TransformDescriptor {
        name: "grid_sync_split",
        class: TransformContractClass::RequiredLegalization,
        description: "Segments whole-grid synchronization barriers into sequential dispatches",
    },
    TransformDescriptor {
        name: "const_prop",
        class: TransformContractClass::CanonicalOptimization,
        description: "Folds let-bound constants and propagates scalar literals",
    },
    TransformDescriptor {
        name: "dead_branch",
        class: TransformContractClass::CanonicalOptimization,
        description: "Eliminates unreachable If branches with constant conditions",
    },
    TransformDescriptor {
        name: "licm",
        class: TransformContractClass::CanonicalOptimization,
        description: "Hoists invariant bindings out of Loop bodies",
    },
    TransformDescriptor {
        name: "collectives",
        class: TransformContractClass::CanonicalOptimization,
        description: "Rewrites collective communication patterns for device execution",
    },
    TransformDescriptor {
        name: "autodiff",
        class: TransformContractClass::CallerRequestedTransform,
        description: "Reverse-mode automatic differentiation generating gradient programs",
    },
    TransformDescriptor {
        name: "rewrite_walk",
        class: TransformContractClass::SharedStructuralWalk,
        description: "Unified structural AST rewrite traversal",
    },
    TransformDescriptor {
        name: "subst",
        class: TransformContractClass::SharedStructuralWalk,
        description: "Induction-variable and let-binding substitution walk",
    },
    TransformDescriptor {
        name: "parallelism",
        class: TransformContractClass::Analysis,
        description: "Shared-nothing parallel dispatch and divergence analysis",
    },
];

/// One host-side IR rewrite the resident pipeline runs.
pub struct HostRewrite {
    /// The module that owns the rewrite, which is also how a trace line and a
    /// firing case name it.
    pub name: &'static str,
    /// Rewrite one program into an equivalent one. An identity return means the
    /// rewrite declined this program, not that it failed.
    pub apply: fn(&Program) -> Program,
}

/// Every host-side rewrite, in the order the resident pipeline applies them.
///
/// This table is the only place the set and the order are stated. The pipeline
/// walks it rather than naming three functions in sequence, so a rewrite that
/// is not here does not run, and `vyre-foundation/tests/transform_rewrites_still_fire.rs`
/// derives its coverage from it rather than from a scan of this directory.
pub const HOST_REWRITES: &[HostRewrite] = &[
    // For each `Loop`, hoist a binding whose value is invariant across the
    // iteration space to a sibling above the loop. A load leaves the loop only
    // when the buffer is declared read-only, and a name bound elsewhere in the
    // enclosing scope stays put so the hoist cannot duplicate a binding.
    HostRewrite {
        name: "licm",
        apply: licm::apply_licm,
    },
    // Constant folding may have turned `let v = 1 + 2` into a literal. Propagate
    // it into every use, which is what makes the cascading folds of fold plus
    // let-dedupe visible to the dead-code pass that follows.
    HostRewrite {
        name: "const_prop",
        apply: const_prop::apply_const_prop,
    },
    // Propagation may have turned a condition into a literal, collapsing an
    // `If` to one branch. Splices the surviving branch into the parent scope so
    // the dead-code pass sees a flatter program.
    HostRewrite {
        name: "dead_branch",
        apply: dead_branch::apply_dead_branch,
    },
];
