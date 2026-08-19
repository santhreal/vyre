//! Unit tests for host oracle elimination gate (Part 1).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use syn::visit::Visit;

use crate::gate::Finding;

use super::host_oracle_elimination_ast::AstAnalysisVisitor;
use super::host_oracle_elimination_eval::evaluate_rules;
use super::host_oracle_elimination_scanners::{
    compute_known_dispatch_exec_fns_multi, derive_canonical_dispatcher_methods,
};

pub(super) fn analyze_files(files: &[(&str, &str)]) -> Vec<Finding> {
    let canonical_source = include_str!("../../../vyre-foundation/src/program_dispatch/mod.rs");
    let canonical_parsed =
        syn::parse_file(canonical_source).expect("canonical trait must parse as Rust");
    let mut canonical_trait_methods = BTreeSet::new();
    let mut canonical_resident_upload_methods = BTreeSet::new();
    derive_canonical_dispatcher_methods(
        &canonical_parsed,
        &mut canonical_trait_methods,
        &mut canonical_resident_upload_methods,
    );
    assert!(
        !canonical_trait_methods.is_empty(),
        "canonical ProgramDispatcher must yield non-empty execution methods"
    );
    assert!(
        !canonical_resident_upload_methods.is_empty(),
        "canonical ProgramDispatcher must yield non-empty resident upload methods"
    );

    let mut parsed_files = Vec::new();
    for &(path, code) in files {
        let parsed = syn::parse_file(code).expect("test code must parse as Rust");
        parsed_files.push((path, parsed));
    }

    let file_asts: Vec<(&Path, &syn::File)> = parsed_files
        .iter()
        .map(|(p, ast)| (Path::new(*p), ast))
        .collect();
    let global_known_dispatch_exec_fns = compute_known_dispatch_exec_fns_multi(
        &file_asts,
        &canonical_trait_methods,
        &canonical_resident_upload_methods,
    );

    let mut all_functions = Vec::new();
    let mut all_calls = Vec::new();
    let mut all_static_consts = Vec::new();
    let mut all_findings = Vec::new();
    let mut all_types_with_public_fields = BTreeSet::new();
    for (path, parsed) in &parsed_files {
        let fn_offset = all_functions.len();
        let mut visitor = AstAnalysisVisitor::new(
            PathBuf::from(*path),
            false,
            fn_offset,
            canonical_trait_methods.clone(),
            canonical_resident_upload_methods.clone(),
        );
        visitor.known_dispatch_exec_fns = global_known_dispatch_exec_fns.clone();

        for item in &parsed.items {
            match item {
                syn::Item::Struct(s) => {
                    visitor.local_declared_types.insert(s.ident.to_string());
                }
                syn::Item::Enum(e) => {
                    visitor.local_declared_types.insert(e.ident.to_string());
                }
                syn::Item::Type(t) => {
                    visitor.local_declared_types.insert(t.ident.to_string());
                }
                syn::Item::Trait(tr) => {
                    visitor.local_declared_types.insert(tr.ident.to_string());
                }
                syn::Item::Union(u) => {
                    visitor.local_declared_types.insert(u.ident.to_string());
                }
                _ => {}
            }
        }

        visitor.visit_file(parsed);
        all_functions.extend(visitor.functions);
        all_calls.extend(visitor.calls);
        all_static_consts.extend(visitor.static_consts);
        all_findings.extend(visitor.direct_findings);
        all_types_with_public_fields.extend(visitor.types_with_public_fields);
    }

    let evaluated = evaluate_rules(
        &all_functions,
        &all_calls,
        &all_static_consts,
        &all_types_with_public_fields,
    );
    all_findings.extend(evaluated);

    let mut deduped_findings = Vec::new();
    let mut seen_findings = BTreeSet::new();
    for finding in all_findings {
        let key = (finding.file.clone(), finding.line, finding.message.clone());
        if seen_findings.insert(key) {
            deduped_findings.push(finding);
        }
    }
    deduped_findings
}

#[test]
fn canonical_program_dispatcher_exact_path_derives_execution_methods() {
    let canonical_source = include_str!("../../../vyre-foundation/src/program_dispatch/mod.rs");
    let canonical_parsed =
        syn::parse_file(canonical_source).expect("canonical trait must parse as Rust");
    let mut canonical_trait_methods = BTreeSet::new();
    let mut canonical_resident_upload_methods = BTreeSet::new();
    derive_canonical_dispatcher_methods(
        &canonical_parsed,
        &mut canonical_trait_methods,
        &mut canonical_resident_upload_methods,
    );
    assert!(
        canonical_trait_methods.contains("dispatch"),
        "must contain direct dispatch method"
    );
    assert!(
        canonical_trait_methods.contains("dispatch_resident"),
        "must contain dispatch_resident"
    );
    assert!(
        canonical_trait_methods.contains("dispatch_resident_sequence"),
        "must contain dispatch_resident_sequence"
    );
    assert!(
        canonical_trait_methods.contains("dispatch_resident_sequence_read_many"),
        "must contain dispatch_resident_sequence_read_many"
    );
    assert!(
        canonical_trait_methods.contains("dispatch_resident_sequence_read_ranges"),
        "must contain dispatch_resident_sequence_read_ranges"
    );
    assert!(
        !canonical_trait_methods.contains("supports_persistent"),
        "metadata methods must not be execution methods"
    );
    assert!(
        !canonical_trait_methods.contains("alloc_resident"),
        "allocation methods must not be execution methods"
    );
    assert!(
        canonical_resident_upload_methods.contains("upload_resident"),
        "must derive resident upload methods from immutable byte payload parameters"
    );
    assert!(
        !canonical_resident_upload_methods.contains("read_resident_ranges_into"),
        "mutable readback outputs must not masquerade as upload methods"
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
use vyre_foundation::program_dispatch::ProgramDispatcher;

pub fn validate_circuit(n: u32) -> Result<(), String> {
    if n == 0 {
        return Err("n must be non-zero".to_string());
    }
    Ok(())
}

pub fn predict_runtime_fixed_via(
    dispatcher: &impl ProgramDispatcher,
    weights: &[u32],
) -> Result<(u32, u32), String> {
    validate_circuit(weights.len() as u32)?;
    let _ = dispatcher.dispatch(1, 2);
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
    let code = r#"
pub struct TestHelper;

#[cfg(test)]
impl TestHelper {
    pub fn compute_oracle(&self, input: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &x in input {
            out.extend_from_slice(&x.wrapping_mul(3).to_le_bytes());
        }
        out
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/scoped_impl.rs", code)]);
    assert!(
            findings.is_empty(),
            "#[cfg(test)] impl methods must be scoped as test-only and produce zero findings, got: {findings:?}"
        );
}

#[test]
fn cfg_test_trait_block_is_scoped_as_test() {
    let code = r#"
#[cfg(test)]
pub trait TestOracleTrait {
    fn default_sim(&self, input: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &x in input {
            out.extend_from_slice(&x.wrapping_add(1).to_le_bytes());
        }
        out
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/scoped_trait.rs", code)]);
    assert!(
        findings.is_empty(),
        "#[cfg(test)] trait with default body must be scoped as test-only, got: {findings:?}"
    );
}

#[test]
fn production_trait_default_method_uncalled_oracle_is_flagged() {
    let code = r#"
pub trait ProductionTrait {
    fn default_sim(&self, input: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &x in input {
            out.extend_from_slice(&x.wrapping_add(1).to_le_bytes());
        }
        out
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/trait_oracle.rs", code)]);
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
    pub trait ProgramDispatcher {
        fn dispatch(&self, a: u32, b: u32);
    }
}
use sibling::ProgramDispatcher;

pub fn fake_dispatch(d: &impl ProgramDispatcher, x: f32) -> f32 {
    d.dispatch(1, 2);
    x + 1.0
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/sibling_dispatcher.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "sibling imported ProgramDispatcher masquerade must be flagged"
    );
    assert!(findings[0].message.contains("`fake_dispatch`"));
    assert_eq!(findings[0].line, Some(9));
}

#[test]
fn mutation_catches_fake_dispatch_error_without_canonical_dispatcher() {
    let code = r#"
pub struct FakeDispatchError;

pub struct LocalDevice;
impl LocalDevice {
    pub fn dispatch(&self, _a: u32, _b: u32) {}
}

pub fn fake_dispatch(x: f32) -> Result<f32, FakeDispatchError> {
    let obj = LocalDevice;
    obj.dispatch(1, 2);
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
use vyre_foundation::program_dispatch::{DispatchError, ResidentReadRange};

pub struct LocalDevice;
impl LocalDevice {
    pub fn dispatch(&self, _a: u32, _b: u32) {}
}

pub fn fake_dispatch_with_error_param(
    _err: &DispatchError,
    _range: &ResidentReadRange,
    x: f32,
) -> f32 {
    let obj = LocalDevice;
    obj.dispatch(1, 2);
    x + 1.0
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/fake_dispatch_params.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "DispatchError/ResidentReadRange parameters must not establish dispatch roots"
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
    OperationRegistration::library(
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
    geometry_requirements: None,
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
    BogusOperationRegistration::library("mock", || {}, None, None);
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
    OperationRegistration::library(
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
    OperationRegistration::library(
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
    OperationRegistration::library(
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
    OperationRegistration::library(
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

#[test]
fn mutation_operation_registration_catches_local_closure_alias_in_expected_output() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationRegistration;

pub fn reduce_count(bitset: &str, out: &str, words: u32) -> Program {
    Program::default()
}

inventory::submit! {
    OperationRegistration::library(
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn host_oracle_fallback(words: &[u32]) -> u32 {
    let mut sum = 0u32;
    for &w in words {
        sum = sum.wrapping_add(w);
    }
    sum
}

pub fn dispatch_with_cpu_fallback(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<u32, DispatchError> {
    match dispatcher.dispatch(1, 2) {
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn host_oracle_fallback(words: &[u32]) -> u32 {
    let mut sum = 0u32;
    for &w in words {
        sum = sum.wrapping_add(w);
    }
    sum
}

pub fn dispatch_with_unwrap_fallback(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<u32, DispatchError> {
    dispatcher
        .dispatch(1, 2)
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn motif_matches_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<bool, DispatchError> {
    let _ = dispatcher.dispatch(1, 2)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn motif_matches_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<bool, DispatchError> {
    match dispatcher.dispatch(1, 2) {
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn motif_count_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<usize, DispatchError> {
    dispatcher
        .dispatch(1, 2)
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn motif_participation_count_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<u32, DispatchError> {
    let _ = dispatcher.dispatch(1, 2)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn count_matches_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<usize, DispatchError> {
    let _ = dispatcher.dispatch(1, 2)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn multi_stage_dispatch_via(dispatcher: &dyn ProgramDispatcher, input: &[u32]) -> Result<Vec<u8>, DispatchError> {
    let prog1 = Program::default();
    let out1 = dispatcher.dispatch(&prog1, &[vec![]], None)?;
    let mut staged = vec![0u32; input.len()];
    for i in 0..input.len() {
        staged[i] = input[i] ^ (out1[0][0] as u32);
    }
    let prog2 = Program::default();
    let out2 = dispatcher.dispatch(&prog2, &[staged], None)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn impacted_entries_into(
    dispatcher: &dyn ProgramDispatcher,
    lineage_cells: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut impact_raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut impact_raw)?;

    let mut closure_raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut closure_raw)?;

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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_raw_outputs(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_with_local_and_reserve(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_constant(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_subsliced(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u16>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_shifted(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

fn from_le_bytes(chunk: &[u8]) -> u32 {
    (chunk[0] as u32) + 42
}

pub fn decode_with_custom_helper(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
