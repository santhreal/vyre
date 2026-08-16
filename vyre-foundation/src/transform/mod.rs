//! IR transformation passes.
//!
//! Before a `Program` is lowered to backend code, it runs through a series
//! of target-independent optimizations and transformations: call inlining,
//! common-subexpression elimination, dead-code elimination, and visitor
//! utilities. These passes are the vyre analogue of LLVM's mid-level IR
//! passes.

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
/// Hoists a let whose value does not depend on the loop index out of the loop
/// body, including a `Load` from a buffer the loop only reads.
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
