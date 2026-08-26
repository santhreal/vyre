//! Unit tests for host oracle elimination gate (Part 1).

use super::host_oracle_elimination_test_fixtures::{incrementing_oracle_body, oracle_body};
use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::gate::Finding;

use super::host_oracle_elimination_eval::analyze_parsed;
use super::host_oracle_elimination_scanners::{
    derive_canonical_dispatcher_methods, derive_canonical_execution_fns,
    derive_registration_expected_output_indices,
};

pub(super) fn analyze_files(files: &[(&str, &str)]) -> Vec<Finding> {
    let canonical_source = include_str!("../../../vyre-megakernel/src/execution.rs");
    let canonical_parsed =
        syn::parse_file(canonical_source).expect("canonical trait must parse as Rust");
    let mut canonical_trait_methods = BTreeSet::new();
    let mut canonical_input_binding_methods = BTreeSet::new();
    derive_canonical_dispatcher_methods(
        &canonical_parsed,
        &mut canonical_trait_methods,
        &mut canonical_input_binding_methods,
    );
    assert!(
        !canonical_trait_methods.is_empty(),
        "canonical SemanticExecutor must yield non-empty execution methods"
    );
    assert!(
        !canonical_input_binding_methods.is_empty(),
        "canonical SemanticExecutionRequest must yield non-empty input-binding methods"
    );
    let canonical_execution_fns =
        derive_canonical_execution_fns(&canonical_parsed, &canonical_trait_methods);
    assert!(
        !canonical_execution_fns.is_empty(),
        "canonical seam must publish at least one free execution helper"
    );

    let parsed_sources: Vec<(PathBuf, syn::File, bool)> = files
        .iter()
        .map(|&(path, code)| {
            let parsed = syn::parse_file(code).expect("test code must parse as Rust");
            (PathBuf::from(path), parsed, false)
        })
        .collect();

    let registration_source = include_str!("../../../vyre-foundation/src/operation/mod.rs");
    let registration_parsed =
        syn::parse_file(registration_source).expect("registration source must parse as Rust");
    let registration_expected_output_indices =
        derive_registration_expected_output_indices(&registration_parsed);
    assert!(
        !registration_expected_output_indices.is_empty(),
        "OperationRegistration must publish a constructor taking `expected_output`"
    );

    analyze_parsed(
        &parsed_sources,
        &canonical_trait_methods,
        &canonical_input_binding_methods,
        &canonical_execution_fns,
        &registration_expected_output_indices,
    )
}

#[test]
fn canonical_semantic_executor_exact_path_derives_execution_methods() {
    let canonical_source = include_str!("../../../vyre-megakernel/src/execution.rs");
    let canonical_parsed =
        syn::parse_file(canonical_source).expect("canonical trait must parse as Rust");
    let mut canonical_trait_methods = BTreeSet::new();
    let mut canonical_input_binding_methods = BTreeSet::new();
    derive_canonical_dispatcher_methods(
        &canonical_parsed,
        &mut canonical_trait_methods,
        &mut canonical_input_binding_methods,
    );
    assert!(
        canonical_trait_methods.contains("execute"),
        "must contain the semantic execution method"
    );
    assert!(
        !canonical_trait_methods.contains("writable_graph_values"),
        "graph inspection helpers must not be execution methods"
    );
    assert!(
        canonical_input_binding_methods.contains("new"),
        "must derive input-binding methods from immutable byte payload parameters"
    );

    let canonical_execution_fns =
        derive_canonical_execution_fns(&canonical_parsed, &canonical_trait_methods);
    assert!(
        canonical_execution_fns.contains("execute_single_program"),
        "must derive the free helper that submits through the executor"
    );
    assert!(
        !canonical_execution_fns.contains("writable_graph_value_buffers"),
        "a helper that never submits through the executor must not read as execution"
    );
}

#[test]
fn clean_production_code_produces_no_findings() {
    let code = r#"
use vyre_foundation::ir::Program;

pub fn add_u32(input: &str, out: &str, n: u32) -> Result<Program, String> {
    Ok(Program::new())
}

const EXPECTED_OUTPUT: [u8; 4] = [0, 1, 2, 3];

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![EXPECTED_OUTPUT.to_vec()]]
}
"#;
    let findings = analyze_files(&[("vyre-primitives/src/add.rs", code)]);
    assert!(
        findings.is_empty(),
        "expected clean production code to pass without findings, got: {findings:?}"
    );
}

#[test]
fn expected_output_with_wire_pack_and_literal_helper_is_convicted() {
    let code = r#"
use vyre_foundation::ir::Program;

const EXPECTED: [u32; 2] = [42, 99];

fn expected_bytes() -> Vec<u8> {
    vec![1, 2, 3, 4]
}

pub fn add_u32(input: &str, out: &str, n: u32) -> Result<Program, String> {
    Ok(Program::new())
}

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![
        crate::wire::pack_u32_slice(&EXPECTED),
        expected_bytes(),
    ]]
}
"#;
    let findings = analyze_files(&[("vyre-primitives/src/add.rs", code)]);
    assert!(
        !findings.is_empty(),
        "expected_output with wire pack and literal helper must be convicted"
    );
}

#[test]
fn gpu_dispatcher_boundary_and_validation_functions_are_permitted() {
    let code = r#"
use vyre_megakernel::SemanticExecutor;

pub fn validate_circuit(n: u32) -> Result<(), String> {
    if n == 0 {
        return Err("n must be non-zero".to_string());
    }
    Ok(())
}

pub fn predict_runtime_fixed_via(
    dispatcher: &impl SemanticExecutor,
    weights: &[u32],
) -> Result<(u32, u32), String> {
    validate_circuit(weights.len() as u32)?;
    let _ = dispatcher.execute(1, 2);
    Ok((10, 20))
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/analysis/cost_model.rs", code)]);
    assert!(
        findings.is_empty(),
        "gpu dispatcher boundary and validation functions must be permitted, got: {findings:?}"
    );
}

#[test]
fn compiler_planner_reachable_from_builder_is_permitted() {
    let code = r#"
use vyre_foundation::ir::Program;

pub struct DiagnosticAggregationPlan {
    pub items: u32,
}

pub fn plan_compact_diagnostic_readback(n: u32) -> Result<DiagnosticAggregationPlan, String> {
    let _table = binary_byte_lut();
    Ok(DiagnosticAggregationPlan { items: n })
}

pub fn compile_pipeline(n: u32) -> Program {
    let _plan = plan_compact_diagnostic_readback(n).unwrap();
    Program::new()
}

fn binary_byte_lut() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0..256 {
        table[i] = (i as u32).wrapping_mul(3);
    }
    table
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/analysis/planner.rs", code)]);
    assert!(
        findings.is_empty(),
        "compiler planner called by IR builder must be permitted, got: {findings:?}"
    );
}

#[test]
fn cfg_test_cpu_ref_is_permitted_and_not_flagged() {
    let code = r#"
use vyre_foundation::ir::Program;

pub fn fma_f32(a: &str, b: &str, c: &str, out: &str, n: u32) -> Program {
    Program::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cpu_ref(a: &[f32], b: &[f32], c: &[f32]) -> Vec<u8> {
        let sim = vyre_reference::subgroup::SubgroupSimulator::default();
        vec![]
    }

    #[test]
    fn test_fma_correctness() {
        let out = test_cpu_ref(&[], &[], &[]);
        assert_eq!(out, vec![]);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/math/fma.rs", code)]);
    assert!(
        findings.is_empty(),
        "expected test-scoped cpu_ref helper to be permitted, got: {findings:?}"
    );
}

#[test]
fn cfg_test_item_level_function_body_vyre_reference_is_not_misclassified() {
    let code = r#"
#[test]
fn test_sim_in_standalone_function() {
    let _sim = vyre_reference::subgroup::SubgroupSimulator::default();
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/test_item.rs", code)]);
    assert!(
            findings.is_empty(),
            "item-level test functions must not report production simulator findings, got: {findings:?}"
        );
}

#[test]
fn cfg_test_impl_block_is_scoped_as_test() {
    let code = format!(
        r#"
pub struct TestHelper;

#[cfg(test)]
impl TestHelper {{
    pub fn compute_oracle(&self, input: &[u32]) -> Vec<u8> {{
{body}
    }}
}}
"#,
        body = oracle_body("wrapping_mul(3)")
    );
    let findings = analyze_files(&[("vyre-libs/src/scoped_impl.rs", &code)]);
    assert!(
            findings.is_empty(),
            "#[cfg(test)] impl methods must be scoped as test-only and produce zero findings, got: {findings:?}"
        );
}

#[test]
fn cfg_test_trait_block_is_scoped_as_test() {
    let code = format!(
        r#"
#[cfg(test)]
pub trait TestOracleTrait {{
    fn default_sim(&self, input: &[u32]) -> Vec<u8> {{
{body}
    }}
}}
"#,
        body = incrementing_oracle_body()
    );
    let findings = analyze_files(&[("vyre-libs/src/scoped_trait.rs", &code)]);
    assert!(
        findings.is_empty(),
        "#[cfg(test)] trait with default body must be scoped as test-only, got: {findings:?}"
    );
}

#[test]
fn production_trait_default_method_uncalled_oracle_is_flagged() {
    let code = format!(
        r#"
pub trait ProductionTrait {{
    fn default_sim(&self, input: &[u32]) -> Vec<u8> {{
{body}
    }}
}}
"#,
        body = incrementing_oracle_body()
    );
    let findings = analyze_files(&[("vyre-libs/src/trait_oracle.rs", &code)]);
    assert_eq!(
        findings.len(),
        1,
        "production trait with unreached default oracle body must be flagged"
    );
    assert!(findings[0].message.contains("`default_sim`"));
}

#[test]
fn mutation_oracle_detection_catches_production_vyre_reference_usage() {
    let code = r#"
pub fn simulate_runtime() {
    let _sim = vyre_reference::subgroup::SubgroupSimulator::default();
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/runtime.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "expected finding for production vyre_reference simulator usage"
    );
    assert_eq!(findings[0].line, Some(3));
}

#[test]
fn mutation_catches_local_dummy_program_masquerade() {
    let code = r#"
pub struct Program;

pub fn fake_builder(x: f32) -> Program {
    let _ = x * x;
    Program
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/fake_program.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "locally declared Program masquerade must be flagged"
    );
    assert!(findings[0].message.contains("`fake_builder`"));
    assert_eq!(findings[0].line, Some(4));
}

#[test]
fn mutation_catches_crate_bogus_program_masquerade() {
    let code = r#"
use crate::bogus::Program;

pub fn fake_builder(x: f32) -> Program {
    let _ = x * x;
    Program
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/bogus_program.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "crate::bogus::Program masquerade must be flagged"
    );
    assert!(findings[0].message.contains("`fake_builder`"));
    assert_eq!(findings[0].line, Some(4));
}

#[test]
fn mutation_catches_glob_import_program_masquerade() {
    let code = r#"
mod fake {
    pub struct Program;
}
use fake::*;

pub fn fake_builder(x: f32) -> Program {
    let _ = x * x;
    Program
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/glob_program.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "glob imported Program masquerade must be flagged"
    );
    assert!(findings[0].message.contains("`fake_builder`"));
    assert_eq!(findings[0].line, Some(7));
}

#[test]
fn mutation_catches_sibling_imported_fake_program_masquerade() {
    let code = r#"
mod sibling {
    pub struct Program;
}
use sibling::Program;

pub fn fake_builder(x: f32) -> Program {
    let _ = x * x;
    Program
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/sibling_program.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "sibling imported Program masquerade must be flagged"
    );
    assert!(findings[0].message.contains("`fake_builder`"));
    assert_eq!(findings[0].line, Some(7));
}

#[test]
fn mutation_catches_sibling_imported_fake_dispatcher_trait_masquerade() {
    let code = r#"
mod sibling {
    pub trait SemanticExecutor {
        fn dispatch(&self, a: u32, b: u32);
    }
}
use sibling::SemanticExecutor;

pub fn fake_dispatch(d: &impl SemanticExecutor, x: f32) -> f32 {
    d.execute(1, 2);
    x + 1.0
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/sibling_dispatcher.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "sibling imported SemanticExecutor masquerade must be flagged"
    );
    assert!(findings[0].message.contains("`fake_dispatch`"));
    assert_eq!(findings[0].line, Some(9));
}

#[test]
fn mutation_catches_fake_dispatch_error_without_canonical_dispatcher() {
    let code = r#"
pub struct FakeSemanticExecutionError;

pub struct LocalDevice;
impl LocalDevice {
    pub fn dispatch(&self, _a: u32, _b: u32) {}
}

pub fn fake_dispatch(x: f32) -> Result<f32, FakeSemanticExecutionError> {
    let obj = LocalDevice;
    obj.execute(1, 2);
    Ok(x + 1.0)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/fake_device.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "dispatch call without canonical dispatcher parameter must be flagged"
    );
    assert!(findings[0].message.contains("`fake_dispatch`"));
    assert_eq!(findings[0].line, Some(9));
}

#[test]
fn mutation_catches_dispatch_error_and_resident_read_range_param_masquerade() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionRequest};

pub struct LocalDevice;
impl LocalDevice {
    pub fn dispatch(&self, _a: u32, _b: u32) {}
}

pub fn fake_dispatch_with_error_param(
    _err: &SemanticExecutionError,
    _range: &SemanticExecutionRequest,
    x: f32,
) -> f32 {
    let obj = LocalDevice;
    obj.execute(1, 2);
    x + 1.0
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/fake_dispatch_params.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "SemanticExecutionError/SemanticExecutionRequest parameters must not establish dispatch roots"
    );
    assert!(findings[0]
        .message
        .contains("`fake_dispatch_with_error_param`"));
    assert_eq!(findings[0].line, Some(9));
}

#[test]
fn mutation_catches_mixed_tuple_data_type_masquerade() {
    let code = r#"
use vyre_foundation::ir::DataType;

pub fn oracle_with_mixed_tuple(data: &[u32]) -> (Vec<u32>, DataType) {
    let mut out = Vec::new();
    for &x in data {
        out.push(x * 2);
    }
    (out, DataType::U32)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/mixed_tuple.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "mixed tuple with DataType metadata must not establish an IR builder root"
    );
    assert!(findings[0].message.contains("`oracle_with_mixed_tuple`"));
    assert_eq!(findings[0].line, Some(4));
}

#[test]
fn mutation_catches_result_vec_with_fusion_error_masquerade() {
    let code = r#"
use vyre_foundation::execution_plan::fusion::FusionError;

pub fn fake_fusion_oracle(data: &[u32]) -> Result<Vec<u32>, FusionError> {
    let mut out = Vec::new();
    for &x in data {
        out.push(x.wrapping_add(1));
    }
    Ok(out)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/fake_fusion.rs", code)]);
    assert_eq!(findings.len(), 1, "Result<Vec<u32>, FusionError> where success type is data must not establish an IR builder root");
    assert!(findings[0].message.contains("`fake_fusion_oracle`"));
    assert_eq!(findings[0].line, Some(4));
}

#[test]
fn mutation_operation_registration_allows_test_inputs_generator_and_catches_expected_output_oracle()
{
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program(a: &str, b: &str, out: &str, n: u32) -> Program {
    Program::new()
}

pub fn generate_deterministic_inputs() -> Vec<Vec<Vec<u8>>> {
    let mut words = Vec::new();
    for i in 0..10 {
        words.push(i.wrapping_mul(7));
    }
    vec![vec![crate::wire::pack_u32_slice(&words)]]
}

pub fn dynamic_math_oracle(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &w in words {
        out.extend_from_slice(&w.wrapping_mul(17).to_le_bytes());
    }
    out
}

inventory::submit! {
    OperationRegistration::library_unconstrained(
        "test::op",
        || add_program("a", "b", "out", 2),
        Some(generate_deterministic_inputs),
        Some(|| {
            let fixture = dynamic_math_oracle(&[1, 2]);
            vec![vec![fixture]]
        }),
    )
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/op_test_inputs.rs", code)]);
    assert!(
        !findings.is_empty(),
        "test_inputs must be permitted while dynamic expected_output oracle is convicted"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`dynamic_math_oracle`")
            || f.message.contains("dynamic_math_oracle")));
}

#[test]
fn mutation_operation_registration_struct_literal_catches_expected_output_oracle() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::{OperationRegistration, OperationTier};

pub fn add_program() -> Program {
    Program::new()
}

pub fn struct_literal_oracle(words: &[u32]) -> Vec<u8> {
    words.iter().map(|w| (w.wrapping_mul(3)) as u8).collect()
}

pub static REG: OperationRegistration = OperationRegistration {
    id: "test::literal",
    semantic_version: 1,
    signature: None,
    tier: OperationTier::Library,
    category: None,
    build: Some(add_program),
    test_inputs: None,
    expected_output: Some(|| vec![vec![struct_literal_oracle(&[1, 2])]]),
    laws: &[],
    tolerance: vyre_foundation::operation::TolerancePolicy::EXACT,
    geometry_requirements: vyre_foundation::GeometryRequirements::agnostic(),
    source_file: "test.rs",
    explicit_effects: None,
    explicit_capabilities: None,
};
"#;
    let findings = analyze_files(&[("vyre-libs/src/op_struct_literal.rs", code)]);
    assert!(
        !findings.is_empty(),
        "struct literal expected_output oracle must be caught"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`struct_literal_oracle`")
            || f.message.contains("struct_literal_oracle")));
}

#[test]
fn mutation_operation_registration_aliased_as_or_catches_expected_output_oracle() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration as OR;

pub fn add_program() -> Program {
    Program::new()
}

pub fn aliased_oracle(words: &[u32]) -> Vec<u8> {
    words.iter().map(|w| (w.wrapping_mul(2)) as u8).collect()
}

inventory::submit! {
    OR::library(
        "test::aliased",
        add_program,
        None,
        Some(|| vec![vec![aliased_oracle(&[1, 2])]]),
    )
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/op_aliased.rs", code)]);
    assert!(
        !findings.is_empty(),
        "aliased OR::library expected_output oracle must be caught"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`aliased_oracle`") || f.message.contains("aliased_oracle")));
}

#[test]
fn control_bogus_local_operation_registration_does_not_false_positive() {
    let code = r#"
pub struct BogusOperationRegistration {
    pub expected_output: fn() -> Vec<u8>,
}

impl BogusOperationRegistration {
    pub fn library(_id: &str, _build: fn(), _inputs: Option<fn()>, _expected: Option<fn()>) {}
}

pub fn setup_mock() {
    BogusOperationRegistration::library_unconstrained("mock", || {}, None, None);
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/op_bogus_control.rs", code)]);
    assert!(
        findings.is_empty(),
        "bogus local OperationRegistration struct must not false positive, got: {findings:?}"
    );
}

#[test]
fn mutation_operation_registration_catches_inline_closure_loop_and_math() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program() -> Program {
    Program::new()
}

inventory::submit! {
    OperationRegistration::library_unconstrained(
        "test::inline_math",
        add_program,
        None,
        Some(|| {
            let mut out = Vec::new();
            for i in 0..10 {
                out.push((i * 2) as u8);
            }
            vec![vec![out]]
        }),
    )
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/op_inline_math.rs", code)]);
    assert!(
        !findings.is_empty(),
        "inline closure loop and math must be convicted"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("inline expected_output closure")
            || f.message.contains("expected_output")));
}

#[test]
fn clean_operation_registration_allows_literal_and_const_byte_array() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program() -> Program {
    Program::new()
}

const EXPECTED_BYTES: [u8; 4] = [100, 0, 200, 0];

inventory::submit! {
    OperationRegistration::library_unconstrained(
        "test::clean_literal",
        add_program,
        None,
        Some(|| {
            vec![vec![EXPECTED_BYTES.to_vec()]]
        }),
    )
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/op_clean_literal.rs", code)]);
    assert!(
        findings.is_empty(),
        "clean literal array and to_vec must produce zero findings, got: {findings:?}"
    );
}

#[test]
fn mutation_operation_registration_catches_wire_pack_in_expected_output() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program() -> Program {
    Program::new()
}

const EXPECTED_BYTES: [u32; 2] = [100, 200];

inventory::submit! {
    OperationRegistration::library_unconstrained(
        "test::clean_pack",
        add_program,
        None,
        Some(|| {
            vec![vec![crate::wire::pack_u32_slice(&EXPECTED_BYTES)]]
        }),
    )
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/op_clean_pack.rs", code)]);
    assert!(
        !findings.is_empty(),
        "wire pack in expected_output must be convicted"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("expected_output")));
}

#[test]
fn mutation_operation_registration_catches_helper_function_in_expected_output() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program() -> Program {
    Program::new()
}

fn expected_bytes() -> Vec<u8> {
    vec![1, 2, 3, 4]
}

inventory::submit! {
    OperationRegistration::library_unconstrained(
        "test::helper_expected",
        add_program,
        None,
        Some(|| {
            vec![vec![expected_bytes()]]
        }),
    )
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/op_helper_expected.rs", code)]);
    assert!(
        !findings.is_empty(),
        "helper function call in expected_output must be convicted"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("expected_output")));
}

/// Every `OperationRegistration` constructor convicts a host oracle sitting in
/// its `expected_output` argument.
///
/// The constructor roster was once hardcoded in the visitor. The constructors
/// were later renamed with an `_unconstrained` suffix, the stale roster matched
/// nothing, and the entire registered-fixture class stopped being analyzed: an
/// oracle inside `expected_output` went unconvicted, and every registered
/// callback read as unreachable host data processing. Deriving the roster and
/// each argument position from the registration source closes that class, so a
/// constructor added or renamed there enrolls itself here.
#[test]
fn every_registration_constructor_convicts_a_host_oracle_in_expected_output() {
    let registration_source = include_str!("../../../vyre-foundation/src/operation/mod.rs");
    let registration_parsed =
        syn::parse_file(registration_source).expect("registration source must parse as Rust");
    let roster = derive_registration_expected_output_indices(&registration_parsed);
    assert!(
        roster.len() >= 2,
        "expected several registration constructors taking `expected_output`, derived {roster:?}"
    );

    for (constructor, expected_idx) in &roster {
        let mut args: Vec<String> = (0..*expected_idx).map(|idx| format!("arg_{idx}")).collect();
        args.push("Some(|| vec![vec![expected_bytes()]])".to_string());
        let args = args.join(", ");
        let code = format!(
            r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn add_program() -> Program {{
    Program::new()
}}

fn expected_bytes() -> Vec<u8> {{
    vec![1, 2, 3, 4]
}}

inventory::submit! {{
    OperationRegistration::{constructor}({args})
}}
"#
        );
        let findings = analyze_files(&[("vyre-libs/src/op_roster_probe.rs", &code)]);
        let messages: Vec<String> = findings.iter().map(|f| f.message.clone()).collect();
        assert!(
            messages.iter().any(|message| message.contains("expected_output")),
            "`{constructor}` (expected_output at argument {expected_idx}) left a host oracle unconvicted: {messages:?}"
        );
    }
}

#[test]
fn mutation_operation_registration_catches_local_closure_alias_in_expected_output() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn reduce_count(bitset: &str, out: &str, words: u32) -> Program {
    Program::default()
}

inventory::submit! {
    OperationRegistration::library_unconstrained(
        "vyre-libs::reduce::count",
        || reduce_count("bitset", "out", 2),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0b1111, 0xFFFF_FFFF]), to_bytes(&[0])]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[36])]]
        }),
    )
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/reduce/count.rs", code)]);
    assert!(
        !findings.is_empty(),
        "expected_output local closure/codec execution must be convicted"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("expected_output")));
}

#[test]
fn mutation_catches_dispatcher_err_fallback_branch_oracle() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn host_oracle_fallback(words: &[u32]) -> u32 {
    let mut sum = 0u32;
    for &w in words {
        sum = sum.wrapping_add(w);
    }
    sum
}

pub fn dispatch_with_cpu_fallback(
    dispatcher: &impl SemanticExecutor,
    words: &[u32],
) -> Result<u32, SemanticExecutionError> {
    match dispatcher.execute(1, 2) {
        Ok(_) => Ok(42),
        Err(_) => Ok(host_oracle_fallback(words)),
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/dispatch_fallback.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "dispatcher fallback branch oracle must be convicted"
    );
    assert!(findings[0].message.contains("host CPU reference fallback"));
    assert!(findings[0].message.contains("`host_oracle_fallback`"));
    assert_eq!(findings[0].line, Some(4));
}

#[test]
fn mutation_catches_dispatcher_unwrap_or_else_fallback_oracle() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn host_oracle_fallback(words: &[u32]) -> u32 {
    let mut sum = 0u32;
    for &w in words {
        sum = sum.wrapping_add(w);
    }
    sum
}

pub fn dispatch_with_unwrap_fallback(
    dispatcher: &impl SemanticExecutor,
    words: &[u32],
) -> Result<u32, SemanticExecutionError> {
    dispatcher
        .execute(1, 2)
        .map(|_| 42)
        .unwrap_or_else(|_| Ok(host_oracle_fallback(words)))
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/dispatch_unwrap_fallback.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "dispatcher unwrap_or_else fallback oracle must be convicted"
    );
    assert!(findings[0].message.contains("host CPU reference fallback"));
    assert!(findings[0].message.contains("`host_oracle_fallback`"));
    assert_eq!(findings[0].line, Some(4));
}

#[test]
fn mutation_catches_dispatcher_post_dispatch_iter_any_reduction() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn motif_matches_via(
    dispatcher: &impl SemanticExecutor,
    words: &[u32],
) -> Result<bool, SemanticExecutionError> {
    let _ = dispatcher.execute(1, 2)?;
    let output = [1u32, 0, 0];
    let any_match = output.iter().any(|&x| x == 1);
    Ok(any_match)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/post_dispatch_any.rs", code)]);
    assert!(
        !findings.is_empty(),
        "post-dispatch .iter().any reduction must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host reduction/aggregation `.any`")));
}

#[test]
fn mutation_catches_dispatcher_match_ok_post_dispatch_iter_any() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn motif_matches_via(
    dispatcher: &impl SemanticExecutor,
    words: &[u32],
) -> Result<bool, SemanticExecutionError> {
    match dispatcher.execute(1, 2) {
        Ok(output) => {
            let any_match = output.iter().any(|&x| x == 1);
            Ok(any_match)
        }
        Err(e) => Err(e),
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/post_dispatch_match_any.rs", code)]);
    assert!(
        !findings.is_empty(),
        "match Ok arm post-dispatch .iter().any reduction must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host reduction/aggregation `.any`")));
}

#[test]
fn mutation_catches_dispatcher_map_closure_post_dispatch_reduction() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn motif_count_via(
    dispatcher: &impl SemanticExecutor,
    words: &[u32],
) -> Result<usize, SemanticExecutionError> {
    dispatcher
        .execute(1, 2)
        .map(|output| output.iter().count())
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/post_dispatch_map_count.rs", code)]);
    assert!(
        !findings.is_empty(),
        "chained .map closure post-dispatch count reduction must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host reduction/aggregation `.count`")));
}

#[test]
fn mutation_catches_dispatcher_post_dispatch_loop_accumulation() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn motif_participation_count_via(
    dispatcher: &impl SemanticExecutor,
    words: &[u32],
) -> Result<u32, SemanticExecutionError> {
    let _ = dispatcher.execute(1, 2)?;
    let output = [1u32, 2, 0];
    let mut count = 0u32;
    for &x in &output {
        if x != 0 {
            count += 1;
        }
    }
    Ok(count)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/post_dispatch_loop.rs", code)]);
    assert!(
        !findings.is_empty(),
        "post-dispatch loop accumulation must be convicted"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("post-dispatch host loop/accumulation")));
}

#[test]
fn mutation_catches_dispatcher_post_dispatch_filter_count_reduction() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn count_matches_via(
    dispatcher: &impl SemanticExecutor,
    words: &[u32],
) -> Result<usize, SemanticExecutionError> {
    let _ = dispatcher.execute(1, 2)?;
    let output = [1u32, 2, 3, 0];
    let count = output.iter().filter(|&&x| x > 0).count();
    Ok(count)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/post_dispatch_count.rs", code)]);
    assert!(
        !findings.is_empty(),
        "post-dispatch filter count reduction must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host reduction/aggregation `.count`")));
}
#[test]
fn clean_dispatcher_allows_inter_dispatch_staging_loop() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn multi_stage_dispatch_via(dispatcher: &dyn SemanticExecutor, input: &[u32]) -> Result<Vec<u8>, SemanticExecutionError> {
    let prog1 = Program::default();
    let out1 = dispatcher.execute(&prog1, &[vec![]], None)?;
    let mut staged = vec![0u32; input.len()];
    for i in 0..input.len() {
        staged[i] = input[i] ^ (out1[0][0] as u32);
    }
    let prog2 = Program::default();
    let out2 = dispatcher.execute(&prog2, &[staged], None)?;
    Ok(out2[0].clone())
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/staging_loop.rs", code)]);
    assert!(
        findings.is_empty(),
        "inter-dispatch staging loop preceding subsequent dispatch must be permitted: {findings:?}"
    );
}
#[test]
fn mutation_catches_escaped_cache_invalidation_post_dispatch_host_projection() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn impacted_entries_into(
    dispatcher: &dyn SemanticExecutor,
    lineage_cells: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut impact_raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut impact_raw)?;

    let mut closure_raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut closure_raw)?;

    let mut impact_mask = Vec::new();
    for chunk in impact_raw.chunks_exact(4) {
        impact_mask.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }

    let mut closure = Vec::new();
    for chunk in closure_raw.chunks_exact(4) {
        closure.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }

    let n_us = n as usize;
    for &cell in lineage_cells {
        let v = (cell % n) as usize;
        let mut impacted = 0u32;
        for r in 0..n_us {
            if impact_mask[r] != 0 && closure[r * n_us + v] != 0 {
                impacted = 1;
                break;
            }
        }
        out.push(impacted);
    }
    Ok(())
}
"#;
    let findings = analyze_files(&[("vyre-driver/src/cache_invalidation.rs", code)]);
    assert!(
            !findings.is_empty(),
            "post-dispatch host projection loop combining comparisons and constructing final output must be flagged"
        );
    assert!(
        findings.iter().any(|f| f
            .message
            .contains("contains post-dispatch host loop/accumulation")
            || f.message
                .contains("executes post-dispatch host arithmetic / semantic derivation")),
        "finding message must identify post-dispatch host derivation, got: {findings:?}"
    );
}

#[test]
fn clean_dispatcher_allows_strict_fixed_width_decoder_only_return() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_raw_outputs(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut decoded = Vec::new();
    for chunk in raw.chunks_exact(4) {
        decoded.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(decoded)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/decoder_only.rs", code)]);
    assert!(
            findings.is_empty(),
            "strict fixed-width little-endian decoder loop must be allowed with zero findings, got: {findings:?}"
        );
}
#[test]
fn clean_dispatcher_allows_optional_local_and_capacity_reserve_decoder_loop() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_with_local_and_reserve(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut out = Vec::new();
    for chunk in raw.chunks_exact(4) {
        out.reserve(1);
        let word = u32::from_le_bytes(chunk.try_into().unwrap());
        out.push(word);
    }
    Ok(out)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/clean_local_decoder.rs", code)]);
    assert!(
            findings.is_empty(),
            "decoder loop with optional local binding and capacity reservation must be permitted with zero findings, got: {findings:?}"
        );
}

#[test]
fn mutation_catches_decoder_loop_independent_constant_append() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_constant(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut out = Vec::new();
    for _chunk in raw.chunks_exact(4) {
        out.reserve(1);
        out.push(7);
    }
    Ok(out)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/constant_append.rs", code)]);
    assert!(
        !findings.is_empty(),
        "decoder loop pushing independent constant must be convicted: {findings:?}"
    );
}

#[test]
fn mutation_catches_decoder_loop_subsliced_width_mismatch() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_subsliced(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u16>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut out = Vec::new();
    for chunk in raw.chunks_exact(4) {
        out.push(u16::from_le_bytes(chunk[..2].try_into().unwrap()));
    }
    Ok(out)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/subsliced_decoder.rs", code)]);
    assert!(
        !findings.is_empty(),
        "decoder loop with sub-slicing and width mismatch must be convicted: {findings:?}"
    );
}

#[test]
fn mutation_catches_decoder_loop_bitshift_transform() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_shifted(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut out = Vec::new();
    for chunk in raw.chunks_exact(4) {
        let word = u32::from_le_bytes(chunk.try_into().unwrap());
        out.push(word << 1);
    }
    Ok(out)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/shifted_decoder.rs", code)]);
    assert!(
        !findings.is_empty(),
        "decoder loop performing bitshift semantic transformation must be convicted: {findings:?}"
    );
}
#[test]
fn mutation_catches_unqualified_or_custom_from_le_bytes_helper() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

fn from_le_bytes(chunk: &[u8]) -> u32 {
    (chunk[0] as u32) + 42
}

pub fn decode_with_custom_helper(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut out = Vec::new();
    for chunk in raw.chunks_exact(4) {
        let word = from_le_bytes(chunk);
        out.push(word);
    }
    Ok(out)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/custom_helper.rs", code)]);
    assert!(
        !findings.is_empty(),
        "unqualified or custom from_le_bytes helper must be convicted: {findings:?}"
    );
}
