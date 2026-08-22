//! Security / taint compositions for program-analysis pipelines.
//!
//! Each operation submits one `OperationRegistration` and exports a
//! `fn(...) -> Program`. Program-analysis lowerers use these stable paths
//! directly.
//!
//! All security ops compose GPU-parallel graph algorithms over the
//! vyre IR: forward / backward reachability, dominator walks, and
//! taint propagation with sanitizer masking.
//!
//! Every public operation is re-exported at this module root. Callers do not
//! depend on the internal file layout.
//!
//! `flow_composition` is `pub(crate)` because its helpers
//! (`fuse_security_flow`, `dataflow_hit_program`,
//! `sanitized_dataflow_hit_program`) are internal building blocks
//! the public primitives compose; consumers should reach them only
//! through a stable public op.

macro_rules! define_bitset_and_security_op {
    (
        $module:ident,
        $function:ident,
        $marker:ident,
        $op_id:literal,
        $left:ident,
        $right:ident,
        $doc:literal,
        tests { $($test_name:ident: ($lhs:expr, $rhs:expr) => $expected:expr;)+ }
    ) => {
        pub(crate) mod $module {
            use vyre_foundation::ir::Program;
            use crate::bitset::and::bitset_and;
            use crate::bitset::bitset_words;

            pub(crate) const OP_ID: &str = $op_id;

            /// Build the canonical security bitset-intersection program.
            #[must_use]
            pub fn $function(
                node_count: u32,
                $left: &str,
                $right: &str,
                out: &str,
            ) -> Program {
                let words = bitset_words(node_count);
                vyre_foundation::composition::tag_program(OP_ID, bitset_and($left, $right, out, words))
            }

            /// CPU oracle for this security bitset-intersection predicate.
            #[must_use]
            #[cfg(test)]
            pub(crate) fn cpu_ref($left: &[u32], $right: &[u32]) -> Vec<u32> {
                vyre_reference::composition_witness::bitset_and_witness($left, $right)
            }

            #[doc = concat!("Soundness marker for [`", stringify!($function), "`].")]
            pub struct $marker;

            impl vyre_spec::soundness::SoundnessTagged for $marker {
                fn soundness(&self) -> vyre_spec::soundness::Soundness {
                    vyre_spec::soundness::Soundness::Exact
                }
            }

            #[cfg(test)]
            mod tests {
                use super::*;

                $(
                    #[test]
                    fn $test_name() {
                        assert_eq!(cpu_ref($lhs, $rhs), $expected);
                    }
                )+
            }
        }

        #[doc = $doc]
        pub use $module::$function;

        #[doc = concat!("Soundness marker for [`", stringify!($function), "`].")]
        pub use $module::$marker;
    };
}

macro_rules! define_bitset_and_not_security_op {
    (
        $module:ident,
        $function:ident,
        $marker:ident,
        $op_id:literal,
        $left:ident,
        $right:ident,
        $doc:literal,
        tests { $($test_name:ident: ($lhs:expr, $rhs:expr) => $expected:expr;)+ }
    ) => {
        pub(crate) mod $module {
            use vyre_foundation::ir::Program;
            use crate::bitset::and_not::bitset_and_not;
            use crate::bitset::bitset_words;

            pub(crate) const OP_ID: &str = $op_id;

            /// Build the canonical security bitset-subtraction program.
            #[must_use]
            pub fn $function(
                node_count: u32,
                $left: &str,
                $right: &str,
                out: &str,
            ) -> Program {
                let words = bitset_words(node_count);
                vyre_foundation::composition::tag_program(OP_ID, bitset_and_not($left, $right, out, words))
            }

            /// CPU oracle for this security bitset-subtraction predicate.
            #[must_use]
            #[cfg(test)]
            pub(crate) fn cpu_ref($left: &[u32], $right: &[u32]) -> Vec<u32> {
                vyre_reference::composition_witness::bitset_and_not_witness($left, $right)
            }

            #[doc = concat!("Soundness marker for [`", stringify!($function), "`].")]
            pub struct $marker;

            impl vyre_spec::soundness::SoundnessTagged for $marker {
                fn soundness(&self) -> vyre_spec::soundness::Soundness {
                    vyre_spec::soundness::Soundness::Exact
                }
            }

            #[cfg(test)]
            mod tests {
                use super::*;

                $(
                    #[test]
                    fn $test_name() {
                        assert_eq!(cpu_ref($lhs, $rhs), $expected);
                    }
                )+
            }
        }

        #[doc = $doc]
        pub use $module::$function;

        #[doc = concat!("Soundness marker for [`", stringify!($function), "`].")]
        pub use $module::$marker;
    };
}

pub(crate) mod aliases_dataflow;
define_bitset_and_security_op!(
    auth_check_dominates,
    auth_check_dominates,
    AuthCheckDominates,
    "vyre-libs::security::auth_check_dominates",
    auth_doms,
    sensitive_op_set,
    "`auth_check_dominates` - authorization check dominates sensitive operation.",
    tests {
        protected_op_returns_set: (&[0b1100], &[0b0100]) => vec![0b0100];
        unprotected_op_returns_empty: (&[0b0001], &[0b1110]) => vec![0];
        no_sensitive_ops: (&[0xFFFF], &[0]) => vec![0];
        no_auth_checks: (&[0], &[0xFFFF]) => vec![0];
    }
);
pub(crate) mod bounded_by_comparison;
define_bitset_and_security_op!(
    buffer_size_check,
    buffer_size_check,
    BufferSizeCheck,
    "vyre-libs::security::buffer_size_check",
    size_compared,
    user_input_set,
    "`buffer_size_check` - buffer size is compared to user input.",
    tests {
        checked_size_returns_set: (&[0b1010], &[0b1100]) => vec![0b1000];
        unchecked_size_returns_empty: (&[0b0001], &[0b1110]) => vec![0];
        no_user_input_yields_empty: (&[0xFFFF], &[0]) => vec![0];
        full_overlap: (&[0xDEAD], &[0xDEAD]) => vec![0xDEAD];
    }
);
mod catalog;
pub(crate) mod dominance_predecessors;
#[cfg(test)]
pub(crate) mod facts;
/// Canonical `@family` name to tag-bit allocation shared by rule labels.
pub mod family_mask;
pub(crate) mod flow_composition;
pub(crate) mod flows_to;
pub(crate) mod flows_to_to_sink;
pub(crate) mod flows_to_with_sanitizer;
define_bitset_and_not_security_op!(
    format_string_check,
    format_string_check,
    FormatStringCheck,
    "vyre-libs::security::format_string_check",
    format_arg_pts,
    non_literal_set,
    "`format_string_check` - format argument is reachable only from literals.",
    tests {
        literal_only_returns_full: (&[0xFFFF], &[0]) => vec![0xFFFF];
        user_input_present_subtracts: (&[0xFFFF], &[0xFF00]) => vec![0x00FF];
        fully_user_input_returns_empty: (&[0xDEAD], &[0xFFFF]) => vec![0];
        distributes: (&[0xFFFF, 0x0F0F], &[0xFF00, 0x0000]) => vec![0x00FF, 0x0F0F];
    }
);
pub(crate) mod integer_overflow_arith;
pub(crate) mod label_by_family;
define_bitset_and_security_op!(
    lock_dominates,
    lock_dominates,
    LockDominates,
    "vyre-libs::security::lock_dominates",
    lock_doms,
    shared_access_set,
    "`lock_dominates` - lock acquisition dominates shared-state access.",
    tests {
        locked_access: (&[0b1110], &[0b0010]) => vec![0b0010];
        unlocked_access: (&[0b0001], &[0b0010]) => vec![0];
        no_accesses: (&[0xFFFF], &[0]) => vec![0];
        empty_lock_set: (&[0], &[0xFFFF]) => vec![0];
    }
);
define_bitset_and_security_op!(
    path_canonical,
    path_canonical,
    PathCanonical,
    "vyre-libs::security::path_canonical",
    canonicalizer_dominates,
    fs_op_set,
    "`path_canonical` - path string was canonicalized before a filesystem operation.",
    tests {
        canonicalized_op: (&[0b1110], &[0b0010]) => vec![0b0010];
        uncanonicalized_op: (&[0b0001], &[0b0010]) => vec![0];
        no_fs_ops: (&[0xFFFF], &[0]) => vec![0];
        distributes: (&[0xFF00, 0x00FF], &[0xFFFF, 0xFFFF]) => vec![0xFF00, 0x00FF];
    }
);
#[cfg(test)]
pub(crate) mod predicate_catalog;
#[cfg(test)]
pub(crate) mod relation_analyzer;
#[cfg(test)]
pub(crate) mod reporter;
pub(crate) mod sanitized_by;
define_bitset_and_security_op!(
    sanitizer_dominates,
    sanitizer_dominates,
    SanitizerDominates,
    "vyre-libs::security::sanitizer_dominates",
    sanitizer_doms,
    sink_set,
    "`sanitizer_dominates` - sanitizer dominates the queried sink.",
    tests {
        dominated_sink_returns_set: (&[0b1111], &[0b0010]) => vec![0b0010];
        non_dominated_sink_returns_empty: (&[0b0001], &[0b0010]) => vec![0];
        no_sinks_returns_empty: (&[0xFFFF], &[0]) => vec![0];
        distributes_per_word: (&[0xFF00, 0x00FF], &[0x0FF0, 0x0FF0]) => vec![0x0F00, 0x00F0];
    }
);
pub(crate) mod sink_intersection;
define_bitset_and_security_op!(
    sql_param_bound,
    sql_param_bound,
    SqlParamBound,
    "vyre-libs::security::sql_param_bound",
    param_binding_set,
    sql_query_set,
    "`sql_param_bound` - SQL query is built through parameter binding.",
    tests {
        parameterized_query: (&[0b1100], &[0b0100]) => vec![0b0100];
        raw_concat_query: (&[0b0001], &[0b0010]) => vec![0];
        no_queries: (&[0xFFFF], &[0]) => vec![0];
        distributes: (&[0xFF00, 0xF0F0], &[0x0FF0, 0x0F0F]) => vec![0x0F00, 0x0000];
    }
);
pub(crate) mod taint_flow;
pub(crate) mod taint_kill;
pub(crate) mod taint_pollution;
define_bitset_and_not_security_op!(
    unchecked_return,
    unchecked_return,
    UncheckedReturn,
    "vyre-libs::security::unchecked_return",
    use_set,
    check_dominates,
    "`unchecked_return` - sensitive return-value use lacks a dominating check.",
    tests {
        use_without_check_returns_set: (&[0b1100], &[0b0001]) => vec![0b1100];
        use_with_dominating_check_returns_empty: (&[0b0010], &[0b0010]) => vec![0];
        no_uses_returns_empty: (&[0], &[0xFFFF]) => vec![0];
        distributes: (&[0xFFFF, 0x0F0F], &[0x00FF, 0xF000]) => vec![0xFF00, 0x0F0F];
    }
);
define_bitset_and_security_op!(
    xss_escape,
    xss_escape,
    XssEscape,
    "vyre-libs::security::xss_escape",
    escape_dominates,
    render_set,
    "`xss_escape` - HTML output escaping dominates render sites.",
    tests {
        escaped_render: (&[0b1100], &[0b0100]) => vec![0b0100];
        unescaped_render: (&[0b0001], &[0b0010]) => vec![0];
        no_renders: (&[0xFFFF], &[0]) => vec![0];
        no_escape_dominators: (&[0], &[0xFFFF]) => vec![0];
    }
);

pub use aliases_dataflow::OP_ID;
pub use aliases_dataflow::{aliases_dataflow, try_aliases_dataflow};
pub use bounded_by_comparison::bounded_by_comparison;
pub use dominance_predecessors::dominance_predecessors;
#[cfg(test)]
pub use facts::{finding_from_sanitized_source_to_sink_query, SourceToSinkFindingRequest};
#[cfg(test)]
pub use facts::{
    AnalysisFact, AnalysisFactColumns, AnalysisFactError, AnalysisFactTable, AnalysisSourceSpan,
    FactId, FactKind, FindingProofBundle, FindingProofStep,
};
pub use flows_to::flows_to;
pub use flows_to::{flows_to_alias_only, ALIAS_PROPAGATION_MASK, FLOWS_TO_MASK};
pub use flows_to_to_sink::flows_to_to_sink;
pub use flows_to_with_sanitizer::flows_to_with_sanitizer;
#[cfg(test)]
pub use flows_to_with_sanitizer::sanitized_flow_final_finding_soundness;
pub use flows_to_with_sanitizer::{
    sanitized_flow_final_soundness_contract, sanitized_flow_soundness_contract,
    SanitizedFlowContractViolation, SanitizedFlowExecutionMode, SanitizedFlowSoundnessContract,
    FIXPOINT_OP_ID,
};
pub use integer_overflow_arith::integer_overflow_arith;
pub use integer_overflow_arith::IntegerOverflowArith;
pub use label_by_family::label_by_family;
#[cfg(test)]
pub use predicate_catalog::{
    security_predicate_row_by_op_id, security_predicate_rows, try_security_predicate_rows,
    SecurityPredicateOperation, SecurityPredicateRow,
};
#[cfg(test)]
pub use relation_analyzer::{
    generated_relation_finding_fact_ids, run_generated_security_relation_analyzer,
    GeneratedSecurityRelationAnalyzerEvidence, GeneratedSecurityRelationAnalyzerReport,
    GeneratedSecurityRelationAnalyzerRunStats, GeneratedSecurityRelationAnalyzerSpec,
    SecurityRelationAnalyzerError, SecurityRelationQueryFamily,
    SECURITY_RELATION_ANALYZER_SCHEMA_VERSION,
};
#[cfg(test)]
pub use reporter::{
    render_security_reporter_output, SecurityReporterError, SecurityReporterFinding,
    SecurityReporterOutputBytes, SecurityReporterPlannerPath, SecurityReporterSourceFile,
    SECURITY_REPORTER_SCHEMA_VERSION,
};
pub use sanitized_by::sanitized_by;
pub use sink_intersection::sink_intersection;
pub use sink_intersection::SinkIntersection;
pub use taint_flow::taint_flow;
pub use taint_kill::taint_kill;
pub use taint_kill::TaintKill;
pub use taint_pollution::taint_pollution;
pub use taint_pollution::TaintPollution;

/// Validate that a security composition's input shape + buffer names
/// are non-degenerate. Panics with a `Fix:` message on violation so
/// downstream substrate errors don't surface as cryptic OOB indices.
///
/// The contract is: every security op rejects degenerate input rather
/// than building a Program that traps inside the reference interpreter
/// (or worse, runs to completion and emits silently-wrong taint sets).
pub(crate) fn assert_security_inputs(op: &str, node_count: u32, buffers: &[(&str, &str)]) {
    assert!(
        node_count > 0,
        "Fix: {op} node_count must be positive; got 0. \
         A taint analysis over an empty program graph has no meaningful \
         result  -  callers must skip empty translation units before lowering."
    );
    for (role, name) in buffers {
        assert!(
            !name.is_empty(),
            "Fix: {op} requires non-empty buffer name for {role}. \
             Empty buffer names alias to the zero-length lookup key in the \
             validator and produce silent miscompiles. Pass a stable \
             non-empty buffer identifier."
        );
    }
}
