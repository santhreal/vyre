//! Exhaustive registry and applicability contracts for lowering rewrites.
//!
//! Every rewrite candidate considered across the lowering boundary is declared
//! in [`LoweringRewriteRule`] and classified in [`classify_rule`].
//!
//! Non-goals: Layer-1 semantic transformations belong in `vyre-foundation::optimizer`;
//! concrete instruction selection and target scheduling belong in `vyre-emit-*`.
//! Lowering owns only backend-neutral structural rewrites driven by analysis facts.

use serde::Serialize;

/// Universe of lowering and lowering-adjacent rewrite candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum LoweringRewriteRule {
    /// Bounded dependency ordering for pure same-body SSA operations.
    RepresentationCanonicalize,
    /// Promotion of eligible read-only global buffers to constant memory.
    ConstBufferPromote,
    /// Elimination of dead pure operations on `KernelDescriptor`.
    DeadOpElimination,
    /// Vector load and store fusion (vec2/vec4 packing).
    VectorLoadFusion,
    /// Promotion of global memory to workgroup shared memory with cooperative tiling.
    SharedMemPromote,
    /// Decoration and routing of 2D/3D spatial buffers to hardware texture units.
    TexturePromote,
    /// Transformation of compound Array-of-Structures layouts to Structure-of-Arrays.
    LayoutAosToSoa,
    /// Hoisting of loop-invariant operations and loads out of loop bodies.
    LoopLicmLoadHoist,
    /// Common subexpression elimination for pure expressions.
    CommonSubexpressionElim,
}

/// Owner responsible for executing or applying a rewrite rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum RewriteOwnership {
    /// Lowering crate (`vyre-lower`): backend-neutral structural rewrite on `KernelDescriptor`.
    LoweringOwned,
    /// Backend emitter crate (`vyre-emit-*`): target-specific instruction selection or decoration.
    EmitterOwned {
        /// Emitter role and target strategy.
        emitter_role: &'static str,
    },
    /// Optimizer pass (`vyre-foundation::optimizer`): Layer-1 semantic transformation on `Program`.
    FoundationOptimizerOwned {
        /// Foundation pass name.
        pass_name: &'static str,
    },
    /// Host runtime dispatch (`vyre-driver`): host buffer binding and dispatch signature transformation.
    HostDispatchOwned {
        /// Driver dispatch role.
        role: &'static str,
    },
}

/// Applicability contract defining ownership, semantics, and preconditions for a rewrite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RewriteApplicabilityContract {
    /// Rewrite rule variant.
    pub rule: LoweringRewriteRule,
    /// Stable string identifier.
    pub id: &'static str,
    /// Component owning implementation.
    pub ownership: RewriteOwnership,
    /// Whether the rewrite preserves exact `Program` semantics and values.
    pub preserves_program_semantics: bool,
    /// Upstream analysis consumed by the rewrite, if any.
    pub consumed_analysis: Option<&'static str>,
    /// Target IR or artifact modified by the owner.
    pub target_structure: &'static str,
    /// Rationale for ownership decision and lowering applicability.
    pub rationale: &'static str,
    /// Preconditions required before the rewrite may execute.
    pub preconditions: &'static [&'static str],
}

impl LoweringRewriteRule {
    /// Return the stable identifier for this rule.
    #[must_use]
    pub fn id(self) -> &'static str {
        self.contract().id
    }

    /// Return the component owning this rewrite.
    #[must_use]
    pub fn ownership(self) -> RewriteOwnership {
        self.contract().ownership
    }

    /// Return whether this rewrite is lowering-owned and executable on `KernelDescriptor`.
    #[must_use]
    pub fn is_lowering_owned(self) -> bool {
        matches!(self.ownership(), RewriteOwnership::LoweringOwned)
    }

    /// Return the formal applicability contract for this rule.
    #[must_use]
    pub fn contract(self) -> RewriteApplicabilityContract {
        classify_rule(self)
    }
}

/// Complete enumeration of all declared rewrite candidates.
pub const ALL_REWRITE_RULES: &[LoweringRewriteRule] = &[
    LoweringRewriteRule::RepresentationCanonicalize,
    LoweringRewriteRule::ConstBufferPromote,
    LoweringRewriteRule::DeadOpElimination,
    LoweringRewriteRule::VectorLoadFusion,
    LoweringRewriteRule::SharedMemPromote,
    LoweringRewriteRule::TexturePromote,
    LoweringRewriteRule::LayoutAosToSoa,
    LoweringRewriteRule::LoopLicmLoadHoist,
    LoweringRewriteRule::CommonSubexpressionElim,
];

/// Exhaustive classifier for every rewrite rule.
///
/// WHY: No wildcard pattern `_ => ...` is permitted. Adding a variant to
/// [`LoweringRewriteRule`] causes a compile error until its contract, ownership,
/// and applicability decisions are explicitly classified.
#[must_use]
pub fn classify_rule(rule: LoweringRewriteRule) -> RewriteApplicabilityContract {
    match rule {
        LoweringRewriteRule::RepresentationCanonicalize => RewriteApplicabilityContract {
            rule,
            id: "representation_canonicalize",
            ownership: RewriteOwnership::LoweringOwned,
            preserves_program_semantics: true,
            consumed_analysis: None,
            target_structure: "KernelDescriptor",
            rationale: "Linear emitters require pure SSA producers to precede same-body consumers. Lowering owns this backend-neutral topological ordering repair.",
            preconditions: &[
                "Valid input KernelDescriptor",
                "Pure same-body operations have no unresolvable cycles",
            ],
        },
        LoweringRewriteRule::ConstBufferPromote => RewriteApplicabilityContract {
            rule,
            id: "const_buffer_promote",
            ownership: RewriteOwnership::LoweringOwned,
            preserves_program_semantics: true,
            consumed_analysis: Some("analyses::const_buffer_promote"),
            target_structure: "KernelDescriptor",
            rationale: "Promotes qualified read-only global bindings (fixed size, within byte budget, multiple workgroup loads) to MemoryClass::Constant and rewrites LoadGlobal to LoadConstant without changing values or semantics.",
            preconditions: &[
                "Binding has Global memory class and ReadOnly visibility",
                "Fixed element count within const-buffer byte budget",
                "At least two loads target the binding in the kernel body",
            ],
        },
        LoweringRewriteRule::DeadOpElimination => RewriteApplicabilityContract {
            rule,
            id: "dead_op_elimination",
            ownership: RewriteOwnership::LoweringOwned,
            preserves_program_semantics: true,
            consumed_analysis: Some("analyses::dead_op"),
            target_structure: "KernelDescriptor",
            rationale: "Strips dead pure operations produced during lowering expansion (unused literal definitions, unread loop bounds, dead copies) while preserving all surviving SSA result IDs.",
            preconditions: &[
                "Op produces a result id",
                "Op kind is pure per kernel_op_kind_is_dce_pure",
                "Result id is not referenced in the body or any child body",
            ],
        },
        LoweringRewriteRule::VectorLoadFusion => RewriteApplicabilityContract {
            rule,
            id: "vector_load_fusion",
            ownership: RewriteOwnership::LoweringOwned,
            preserves_program_semantics: true,
            consumed_analysis: Some("analyses::vec_pack"),
            target_structure: "KernelDescriptor",
            rationale: "Fuses and canonicalizes verified unit-stride adjacent global memory load and store chains at vec2/vec4 widths when proven aligned, alias-free, side-effect-free, and within scheduling boundaries.",
            preconditions: &[
                "Adjacent scalar loads or stores on the same global buffer",
                "Unit-stride consecutive offsets with identical base expression",
                "Proven alignment to vector transaction width (vec2 or vec4)",
                "No alias uncertainty, intervening side effect, structured control boundary, or scheduling fence",
                "Supported vector width (vec2 or vec4)",
            ],
        },
        LoweringRewriteRule::SharedMemPromote => RewriteApplicabilityContract {
            rule,
            id: "shared_mem_promote",
            ownership: RewriteOwnership::FoundationOptimizerOwned {
                pass_name: "vyre_foundation::transform / megakernel cooperative tiling",
            },
            preserves_program_semantics: false,
            consumed_analysis: Some("analyses::shared_mem_promote"),
            target_structure: "vyre_foundation::ir::Program",
            rationale: "Shared memory promotion requires inserting cooperative tile-loading loops, workgroup barrier synchronization (Barrier), and index transformations. These are semantic IR transformations that belong in vyre-foundation or runtime megakernel planners.",
            preconditions: &[
                "Workgroup data reuse detected",
                "Fits in per-workgroup shared memory byte budget",
                "Cooperative tile loop bounds proved",
            ],
        },
        LoweringRewriteRule::TexturePromote => RewriteApplicabilityContract {
            rule,
            id: "texture_promote",
            ownership: RewriteOwnership::EmitterOwned {
                emitter_role: "Hardware texture binding decoration and sampler configuration",
            },
            preserves_program_semantics: true,
            consumed_analysis: Some("analyses::texture_promote"),
            target_structure: "Backend binding table / descriptor set",
            rationale: "Texture promotion is a backend-specific binding decoration and sampling strategy owned by concrete emitters and target drivers based on hardware filtering and surface capabilities.",
            preconditions: &[
                "Read-only global binding",
                "2D or 3D spatial access pattern detected",
            ],
        },
        LoweringRewriteRule::LayoutAosToSoa => RewriteApplicabilityContract {
            rule,
            id: "layout_aos_to_soa",
            ownership: RewriteOwnership::HostDispatchOwned {
                role: "vyre-driver host buffer binding and dispatch signature transformation",
            },
            preserves_program_semantics: false,
            consumed_analysis: Some("analyses::layout_aos_to_soa"),
            target_structure: "vyre_foundation::ir::Program / Host dispatch signature",
            rationale: "AoS to SoA transformation splits one compound buffer into multiple component buffers, altering the kernel binding signature and requiring host-side buffer dispatch migration. Owned by foundation optimizer and host runtime dispatchers.",
            preconditions: &[
                "Compound data type (Vec/Array/TensorShaped)",
                "Consecutive thread access to structure components",
            ],
        },
        LoweringRewriteRule::LoopLicmLoadHoist => RewriteApplicabilityContract {
            rule,
            id: "loop_licm_load_hoist",
            ownership: RewriteOwnership::FoundationOptimizerOwned {
                pass_name: "vyre_foundation::optimizer::passes::loops::loop_licm",
            },
            preserves_program_semantics: true,
            consumed_analysis: Some("vyre_foundation::transform::licm"),
            target_structure: "vyre_foundation::ir::Program",
            rationale: "Loop-invariant code motion and load hoisting require lexical scope and variable binding analysis at the Program IR layer. Hoisting across StructuredForLoop at the descriptor layer would violate descriptor structured loop scoping rules.",
            preconditions: &[
                "Expression does not depend on loop induction variable",
                "No intervening memory writes in loop body",
            ],
        },
        LoweringRewriteRule::CommonSubexpressionElim => RewriteApplicabilityContract {
            rule,
            id: "common_subexpression_elim",
            ownership: RewriteOwnership::FoundationOptimizerOwned {
                pass_name: "vyre_foundation::optimizer::passes::fusion_cse::cse",
            },
            preserves_program_semantics: true,
            consumed_analysis: Some("analyses::common_subexpr"),
            target_structure: "vyre_foundation::ir::Program",
            rationale: "Full semantic CSE requires complete data type, floating-point rounding mode, and scope information owned by vyre-foundation optimizer. Descriptor-level CSE analysis is diagnostic only.",
            preconditions: &[
                "Identical op kind and operands",
                "Pure operation without side effects",
                "No intervening side effects",
            ],
        },
    }
}

/// Return all registered applicability contracts.
#[must_use]
pub fn all_registered_contracts() -> Vec<RewriteApplicabilityContract> {
    ALL_REWRITE_RULES
        .iter()
        .map(|&rule| classify_rule(rule))
        .collect()
}

/// Return all lowering-owned rewrite rules.
#[must_use]
pub fn lowering_owned_rules() -> Vec<LoweringRewriteRule> {
    ALL_REWRITE_RULES
        .iter()
        .copied()
        .filter(|rule| rule.is_lowering_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_rules_have_non_empty_identifiers_and_rationales() {
        for rule in ALL_REWRITE_RULES {
            let contract = classify_rule(*rule);
            assert_eq!(contract.rule, *rule);
            assert!(!contract.id.trim().is_empty());
            assert!(!contract.rationale.trim().is_empty());
            assert!(!contract.target_structure.trim().is_empty());
            assert!(!contract.preconditions.is_empty());
        }
    }

    #[test]
    fn lowering_owned_rules_are_verified_and_preserve_semantics() {
        let owned = lowering_owned_rules();
        assert_eq!(
            owned,
            vec![
                LoweringRewriteRule::RepresentationCanonicalize,
                LoweringRewriteRule::ConstBufferPromote,
                LoweringRewriteRule::DeadOpElimination,
                LoweringRewriteRule::VectorLoadFusion,
            ]
        );
        for rule in owned {
            let contract = rule.contract();
            assert!(contract.preserves_program_semantics);
            assert_eq!(contract.target_structure, "KernelDescriptor");
            assert!(rule.is_lowering_owned());
        }
    }

    #[test]
    fn vector_load_fusion_is_lowering_owned() {
        let contract = LoweringRewriteRule::VectorLoadFusion.contract();
        assert_eq!(contract.ownership, RewriteOwnership::LoweringOwned);
        assert!(contract.rule.is_lowering_owned());
        assert!(contract.preserves_program_semantics);
        assert_eq!(contract.consumed_analysis, Some("analyses::vec_pack"));
    }
    #[test]
    fn loop_licm_and_cse_are_foundation_optimizer_owned() {
        let licm = LoweringRewriteRule::LoopLicmLoadHoist.contract();
        assert!(matches!(
            licm.ownership,
            RewriteOwnership::FoundationOptimizerOwned { .. }
        ));

        let cse = LoweringRewriteRule::CommonSubexpressionElim.contract();
        assert!(matches!(
            cse.ownership,
            RewriteOwnership::FoundationOptimizerOwned { .. }
        ));
    }
}
