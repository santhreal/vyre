//! Public release macro descriptors and generated-case construction shared by Criterion
//! entrypoints and coverage tests.

use super::families::{release_macro_workloads, release_macro_workloads_for_family};
use super::metadata_condition::METADATA_RECORDS;
use super::run_assembly::encode_u32_words;
use super::synthetic_count::{SyntheticCountWorkload, SyntheticPattern};
use super::synthetic_oracle::{
    pattern_input_count, string_bitmap_scatter_expected_words, string_bitmap_scatter_inputs,
    synthetic_cpu_count, synthetic_inputs,
};
use super::synthetic_programs::build_synthetic_release_program;
use crate::api::metric::digest64_buffers;
use crate::api::resident::input_bytes_total;
use vyre::ir::Program;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReleaseMacroFamily {
    Scan,
    Flow,
    Graph,
    Parser,
    Egraph,
    Resident,
    Matrix,
    Condition,
}

/// Public release macro workload program descriptor for local benchmark entrypoints.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ReleaseMacroProgramSpec {
    /// Stable benchmark case id.
    pub id: &'static str,
    /// Human-readable benchmark name.
    pub name: &'static str,
    /// Logical records processed by the release workload.
    pub records: u32,
    /// Number of input buffers in the generated release workload.
    pub input_buffers: usize,
    /// Minimum CUDA speedup contract attached to this release workload.
    pub min_speedup_x: u32,
    /// Typed release workload family.
    pub family: ReleaseMacroFamily,
    /// Owner crate responsible for this workload.
    pub owner_crate: &'static str,
}

/// Generated release workload case with concrete inputs and CPU-oracle outputs.
#[derive(Clone)]
pub struct ReleaseMacroGeneratedCase {
    /// Public descriptor for the generated workload shape.
    pub spec: ReleaseMacroProgramSpec,
    /// IR program generated for this workload shape.
    pub program: Program,
    /// Concrete input byte buffers.
    pub inputs: Vec<Vec<u8>>,
    /// Expected output byte buffers from the CPU oracle.
    pub expected_outputs: Vec<Vec<u8>>,
    /// Total logical input bytes for the generated inputs.
    pub input_bytes_total: u64,
    /// Digest of expected output bytes.
    pub expected_output_digest: u64,
}

fn release_macro_workload(id: &str) -> Option<&'static SyntheticCountWorkload> {
    release_macro_workloads()
        .into_iter()
        .find(|workload| workload.id == id)
}

fn release_macro_program_spec(
    workload: &SyntheticCountWorkload,
    records: u32,
) -> ReleaseMacroProgramSpec {
    ReleaseMacroProgramSpec {
        id: workload.id,
        name: workload.name,
        records,
        input_buffers: pattern_input_count(workload.pattern),
        min_speedup_x: workload.min_speedup_x as u32,
        family: workload.family,
        owner_crate: workload.owner_crate,
    }
}

/// Return compiler-grade release macro workload descriptors used by Criterion
/// and generated coverage tests.
#[must_use]
pub fn release_macro_program_specs() -> Vec<ReleaseMacroProgramSpec> {
    release_macro_program_specs_for_records(METADATA_RECORDS)
}

/// Return compiler-grade release macro workload descriptors at a reduced or
/// stress-scale record count.
#[must_use]
pub fn release_macro_program_specs_for_records(records: u32) -> Vec<ReleaseMacroProgramSpec> {
    release_macro_workloads()
        .into_iter()
        .map(|workload| release_macro_program_spec(workload, records))
        .collect()
}

/// Return release macro descriptors for one typed workload family.
#[must_use]
pub fn release_macro_program_specs_for_family_and_records(
    family: ReleaseMacroFamily,
    records: u32,
) -> Vec<ReleaseMacroProgramSpec> {
    release_macro_workloads_for_family(family)
        .iter()
        .map(|workload| release_macro_program_spec(workload, records))
        .collect()
}

/// Return only release macro descriptors whose output is a single count word.
#[must_use]
pub fn release_count_macro_program_specs_for_records(records: u32) -> Vec<ReleaseMacroProgramSpec> {
    release_macro_workloads()
        .into_iter()
        .filter(|workload| is_count_output_pattern(workload.pattern))
        .map(|workload| release_macro_program_spec(workload, records))
        .collect()
}

/// Build the IR program for a compiler-grade release macro workload.
#[must_use]
pub fn build_release_macro_program(id: &str) -> Option<Program> {
    release_macro_workload(id)
        .map(|workload| build_synthetic_release_program(workload.pattern, workload.records))
}

/// Build the IR program for a compiler-grade release macro workload at a
/// caller-selected record count.
#[must_use]
pub fn build_release_macro_program_for_records(id: &str, records: u32) -> Option<Program> {
    release_macro_workload(id)
        .map(|workload| build_synthetic_release_program(workload.pattern, records))
}

/// Build a reduced or stress-scale release macro case with generated hostile
/// inputs and CPU-oracle outputs.
#[must_use]
pub fn build_release_macro_case_for_records(
    id: &str,
    records: u32,
) -> Option<ReleaseMacroGeneratedCase> {
    let workload = release_macro_workload(id)?;
    Some(build_release_macro_case_from_workload(workload, records))
}

/// Build all generated release macro cases for one typed workload family.
#[must_use]
pub fn build_release_macro_cases_for_family_and_records(
    family: ReleaseMacroFamily,
    records: u32,
) -> Vec<ReleaseMacroGeneratedCase> {
    release_macro_workloads_for_family(family)
        .iter()
        .map(|workload| build_release_macro_case_from_workload(workload, records))
        .collect()
}

fn build_release_macro_case_from_workload(
    workload: &SyntheticCountWorkload,
    records: u32,
) -> ReleaseMacroGeneratedCase {
    let spec = release_macro_program_spec(workload, records);
    let program = build_synthetic_release_program(workload.pattern, records);
    let (inputs, expected_outputs) = match workload.pattern {
        SyntheticPattern::StringBitmapScatter => {
            let generated = string_bitmap_scatter_inputs(records);
            let expected_words = string_bitmap_scatter_expected_words(
                &generated.pattern_bitmap,
                &generated.rule_bitmap,
                records,
            );
            (generated.inputs, vec![encode_u32_words(&expected_words)])
        }
        SyntheticPattern::ConditionEval
        | SyntheticPattern::OffsetCountAggregation
        | SyntheticPattern::EntropyWindow
        | SyntheticPattern::QuantifiedLoops
        | SyntheticPattern::AliasReachingDef
        | SyntheticPattern::IfdsWitness
        | SyntheticPattern::CAstTraversal
        | SyntheticPattern::MegakernelQueuedBatch
        | SyntheticPattern::EgraphSaturation => {
            let generated = synthetic_inputs(workload.pattern, records);
            let expected = synthetic_cpu_count(workload.pattern, records);
            assert_eq!(
                generated.expected, expected,
                "Fix: release macro generated input oracle diverged from CPU count oracle for {}",
                workload.id
            );
            (generated.inputs, vec![expected.to_le_bytes().to_vec()])
        }
    };

    ReleaseMacroGeneratedCase {
        input_bytes_total: input_bytes_total(&inputs),
        expected_output_digest: digest64_buffers(&expected_outputs),
        spec,
        program,
        inputs,
        expected_outputs,
    }
}

/// Build a reduced or stress-scale release macro case whose output is one
/// CPU-oracle count word.
#[must_use]
pub fn build_release_count_macro_case_for_records(
    id: &str,
    records: u32,
) -> Option<ReleaseMacroGeneratedCase> {
    let workload = release_macro_workload(id)?;
    if !is_count_output_pattern(workload.pattern) {
        return None;
    }
    build_release_macro_case_for_records(id, records)
}

fn is_count_output_pattern(pattern: SyntheticPattern) -> bool {
    !matches!(pattern, SyntheticPattern::StringBitmapScatter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_macro_family_registry_preserves_exact_ids_owners_and_families() {
        let specs = release_macro_program_specs_for_records(33);
        let observed = specs
            .iter()
            .map(|spec| (spec.id, spec.owner_crate, spec.family, spec.input_buffers))
            .collect::<Vec<_>>();

        assert_eq!(
            observed,
            vec![
                (
                    "release.condition_eval.1m",
                    "vyre",
                    ReleaseMacroFamily::Condition,
                    3,
                ),
                (
                    "release.string_bitmap_scatter.1m",
                    "vyre-libs",
                    ReleaseMacroFamily::Scan,
                    2,
                ),
                (
                    "release.offset_count_aggregation.1m",
                    "vyre-libs",
                    ReleaseMacroFamily::Scan,
                    3,
                ),
                (
                    "release.entropy_window.1m",
                    "vyre-libs",
                    ReleaseMacroFamily::Scan,
                    3,
                ),
                (
                    "release.quantified_condition_loops.1m",
                    "vyre",
                    ReleaseMacroFamily::Condition,
                    3,
                ),
                (
                    "release.alias_reaching_def.1m",
                    "vyre-bench",
                    ReleaseMacroFamily::Flow,
                    3,
                ),
                (
                    "release.ifds_witness.1m",
                    "vyre-bench",
                    ReleaseMacroFamily::Flow,
                    3,
                ),
                (
                    "release.c_ast_traversal.1m",
                    "vyre-libs",
                    ReleaseMacroFamily::Parser,
                    3,
                ),
                (
                    "release.megakernel_queue.1m",
                    "vyre-runtime",
                    ReleaseMacroFamily::Resident,
                    3,
                ),
                (
                    "release.egraph_saturation.1m",
                    "vyre-lower",
                    ReleaseMacroFamily::Egraph,
                    3,
                ),
            ],
            "Fix: typed release family registry must preserve the exact release macro case surface."
        );
    }

    #[test]
    fn release_macro_typed_family_builders_preserve_external_flow_and_empty_families() {
        let flow_specs =
            release_macro_program_specs_for_family_and_records(ReleaseMacroFamily::Flow, 33);
        assert_eq!(
            flow_specs
                .iter()
                .map(|spec| (spec.id, spec.owner_crate, spec.family, spec.input_buffers))
                .collect::<Vec<_>>(),
            vec![
                (
                    "release.alias_reaching_def.1m",
                    "vyre-bench",
                    ReleaseMacroFamily::Flow,
                    3,
                ),
                (
                    "release.ifds_witness.1m",
                    "vyre-bench",
                    ReleaseMacroFamily::Flow,
                    3,
                ),
            ],
            "Fix: flow release workloads must stay attached to their benchmark product owner."
        );

        let flow_cases =
            build_release_macro_cases_for_family_and_records(ReleaseMacroFamily::Flow, 33);
        assert_eq!(flow_cases.len(), flow_specs.len());
        for case in flow_cases {
            assert_eq!(case.input_bytes_total, 396);
            assert_eq!(
                case.expected_output_digest,
                digest64_buffers(&case.expected_outputs)
            );
            assert_ne!(
                case.expected_output_digest, 0,
                "Fix: flow release case {} must carry a nonzero CPU-oracle digest.",
                case.spec.id
            );
        }

        for workload in release_macro_workloads_for_family(ReleaseMacroFamily::Flow) {
            assert!(
                workload.tags.contains(&"external-facts"),
                "Fix: flow workload {} must advertise the generic external-facts boundary.",
                workload.id
            );
            assert!(
                !workload.primitive.trim().is_empty(),
                "Fix: flow workload {} must name its benchmarked primitive.",
                workload.id
            );
        }

        assert!(
            release_macro_program_specs_for_family_and_records(ReleaseMacroFamily::Graph, 33)
                .is_empty()
        );
        assert!(
            build_release_macro_cases_for_family_and_records(ReleaseMacroFamily::Matrix, 33)
                .is_empty()
        );
    }

    #[test]
    fn release_macro_generated_cases_record_input_bytes_and_expected_output_digest() {
        for spec in release_macro_program_specs_for_records(33) {
            let case = build_release_macro_case_for_records(spec.id, spec.records)
                .expect("Fix: every release macro spec must build a generated case.");
            let expected_input_bytes = match spec.id {
                "release.string_bitmap_scatter.1m" => 272,
                _ => 396,
            };

            assert_eq!(case.spec.owner_crate, spec.owner_crate);
            assert_eq!(case.spec.family, spec.family);
            assert_eq!(case.input_bytes_total, expected_input_bytes);
            assert_eq!(
                case.expected_output_digest,
                digest64_buffers(&case.expected_outputs)
            );
            assert_ne!(
                case.expected_output_digest, 0,
                "Fix: generated release macro case {} must expose a nonzero expected output digest.",
                spec.id
            );
        }
    }
}
