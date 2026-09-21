//! Unit tests for host oracle elimination gate (Part 2).

use super::host_oracle_elimination_test_fixtures::incrementing_oracle_body;
use super::host_oracle_elimination_tests_part1::analyze_files;
use std::path::Path;

#[test]
fn mutation_catches_spoofed_module_from_le_bytes_helper() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

mod evil {
    pub struct u32;
    impl u32 {
        pub fn from_le_bytes(chunk: &[u8]) -> u32 {
            7
        }
    }
}

pub fn decode_with_spoofed_path(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut out = Vec::new();
    for chunk in raw.chunks_exact(4) {
        let word = evil::u32::from_le_bytes(chunk);
        out.push(word);
    }
    Ok(out)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/spoofed_helper.rs", code)]);
    assert!(
        !findings.is_empty(),
        "spoofed evil::u32::from_le_bytes helper must be convicted: {findings:?}"
    );
}
#[test]
fn mutation_catches_big_endian_decoder_loop() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_be_outputs(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut decoded = Vec::new();
    for chunk in raw.chunks_exact(4) {
        decoded.push(u32::from_be_bytes(chunk.try_into().unwrap()));
    }
    Ok(decoded)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/be_decoder.rs", code)]);
    assert!(
        !findings.is_empty(),
        "big-endian decoder loop must be convicted as non-canonical host oracle: {findings:?}"
    );
}

#[test]
fn mutation_catches_native_endian_decoder_loop() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_ne_outputs(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut decoded = Vec::new();
    for chunk in raw.chunks_exact(4) {
        decoded.push(u32::from_ne_bytes(chunk.try_into().unwrap()));
    }
    Ok(decoded)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/ne_decoder.rs", code)]);
    assert!(
        !findings.is_empty(),
        "native-endian decoder loop must be convicted as non-canonical host oracle: {findings:?}"
    );
}

#[test]
fn mutation_catches_dynamic_chunk_width_decoder_loop() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_dynamic_width(
    dispatcher: &dyn SemanticExecutor,
    chunk_size: usize,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut decoded = Vec::new();
    for chunk in raw.chunks_exact(chunk_size) {
        decoded.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(decoded)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/dynamic_decoder.rs", code)]);
    assert!(
        !findings.is_empty(),
        "dynamic chunk width loop must be convicted as non-canonical host oracle: {findings:?}"
    );
}

#[test]
fn mutation_catches_decoder_unwrap_or_fallback() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_with_fallback(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut decoded = Vec::new();
    for chunk in raw.chunks_exact(4) {
        let fallback = [0u8; 4];
        let bytes = chunk.try_into().unwrap_or(fallback);
        decoded.push(u32::from_le_bytes(bytes));
    }
    Ok(decoded)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/unwrap_or_decoder.rs", code)]);
    assert!(
        !findings.is_empty(),
        "decoder loop with unwrap_or fallback must be convicted: {findings:?}"
    );
}

#[test]
fn mutation_catches_decoder_unwrap_or_default_fallback() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_with_default_fallback(
    dispatcher: &dyn SemanticExecutor,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut decoded = Vec::new();
    for chunk in raw.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().unwrap_or_default();
        decoded.push(u32::from_le_bytes(bytes));
    }
    Ok(decoded)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/unwrap_or_default_decoder.rs", code)]);
    assert!(
        !findings.is_empty(),
        "decoder loop with unwrap_or_default fallback must be convicted: {findings:?}"
    );
}

#[test]
fn mutation_catches_non_codec_indexing_in_decoder_loop() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn decode_with_table_lookup(
    dispatcher: &dyn SemanticExecutor,
    lookup_table: &[u32],
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut decoded = Vec::new();
    for chunk in raw.chunks_exact(4) {
        let word = u32::from_le_bytes(chunk.try_into().unwrap());
        let mapped = lookup_table[word as usize];
        decoded.push(mapped);
    }
    Ok(decoded)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/non_codec_indexing.rs", code)]);
    assert!(
        !findings.is_empty(),
        "decoder loop with non-codec indexing lookup must be convicted: {findings:?}"
    );
}

#[test]
fn unrelated_same_basename_helper_without_dispatch_does_not_create_fake_dispatch_root() {
    let module_a_code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn predict_impact(dispatcher: &dyn SemanticExecutor, out: &mut [u8]) -> Result<(), SemanticExecutionError> {
    dispatcher.execute(&[], out)
}
"#;
    let module_b_code = r#"
use vyre_megakernel::SemanticExecutor;

pub fn predict_impact(dispatcher: &dyn SemanticExecutor) -> usize {
    // Unrelated helper in module_b with the same base name, but doing NO GPU dispatch
    0
}

pub fn validate_before_run(
    dispatcher: &dyn SemanticExecutor,
) -> bool {
    let n = predict_impact(dispatcher);
    n == 0
}
"#;
    let module_c_code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};
use crate::module_a::predict_impact;

pub fn execute_and_filter(
    dispatcher: &dyn SemanticExecutor,
    out: &mut [u8],
) -> Result<bool, SemanticExecutionError> {
    predict_impact(dispatcher, out)?;
    Ok(out[0] != 0)
}
"#;
    let findings = analyze_files(&[
        ("vyre-libs/src/module_a.rs", module_a_code),
        ("vyre-driver/src/module_b.rs", module_b_code),
        ("vyre-driver/src/module_c.rs", module_c_code),
    ]);
    assert!(
            findings.iter().all(|f| f.file.as_deref() != Some(Path::new("vyre-driver/src/module_b.rs"))),
            "unrelated same-basename helper without dispatch must NOT create a fake dispatch root in module_b: {findings:?}"
        );
    assert!(
            findings.iter().any(|f| f.file.as_deref() == Some(Path::new("vyre-driver/src/module_c.rs"))),
            "module_c importing module_a::predict_impact with post-dispatch check must be convicted: {findings:?}"
        );
}

#[test]
fn mutation_catches_post_dispatch_computation_on_borrowed_references_and_slices() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn project_with_borrows(
    dispatcher: &dyn SemanticExecutor,
    cell: u32,
) -> Result<u32, SemanticExecutionError> {
    let mut raw = vec![0u8; 16];
    dispatcher.execute(&[], &mut raw)?;
    let mut decoded = Vec::new();
    for chunk in raw.chunks_exact(4) {
        decoded.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }
    let slice = &decoded[..];
    let mask = &decoded;
    if slice[0] != 0 && mask[1] > cell {
        Ok(1)
    } else {
        Ok(0)
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/borrowed_dispatch.rs", code)]);
    assert!(
            !findings.is_empty(),
            "post-dispatch arithmetic/comparisons on borrowed references and slices must be convicted as host oracle"
        );
}

#[test]
fn mutation_catches_cross_file_dispatch_helper_post_dispatch_projection() {
    let helper_code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn execute_gpu_step(dispatcher: &dyn SemanticExecutor, out: &mut [u8]) -> Result<(), SemanticExecutionError> {
    dispatcher.execute(&[], out)
}
"#;
    let driver_code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};
use crate::helper::execute_gpu_step;

pub fn run_pipeline_with_host_filter(
    dispatcher: &dyn SemanticExecutor,
    target: u32,
) -> Result<bool, SemanticExecutionError> {
    let mut buffer = vec![0u8; 8];
    execute_gpu_step(dispatcher, &mut buffer)?;
    let val0 = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
    let val1 = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
    Ok(val0 + val1 == target)
}
"#;
    let findings = analyze_files(&[
        ("vyre-libs/src/helper.rs", helper_code),
        ("vyre-driver/src/driver.rs", driver_code),
    ]);
    assert!(
        !findings.is_empty(),
        "cross-file dispatch helper post-dispatch arithmetic and comparison must be convicted"
    );
}

#[test]
fn clean_dispatcher_with_pre_validation_and_typed_unpacking_passes() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub struct CircuitSummary {
    pub a: u32,
    pub b: u32,
}

pub fn predict_summary_via(
    dispatcher: &impl SemanticExecutor,
    weights: &[u32],
) -> Result<CircuitSummary, SemanticExecutionError> {
    if weights.is_empty() {
        return Err(SemanticExecutionError::BadInputs("empty weights".to_string()));
    }
    let raw = dispatcher.execute(1, 2)?;
    Ok(CircuitSummary {
        a: raw.get(0).copied().unwrap_or(0),
        b: raw.get(1).copied().unwrap_or(0),
    })
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/clean_dispatch.rs", code)]);
    assert!(
        findings.is_empty(),
        "clean pre-validation and post-dispatch struct unpacking must pass, got: {findings:?}"
    );
}

#[test]
fn clean_dispatcher_with_dispatch_map_unpack_passes() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub struct CircuitSummary {
    pub a: u32,
    pub b: u32,
}

fn unpack_only(raw: Vec<u32>) -> CircuitSummary {
    CircuitSummary {
        a: raw.get(0).copied().unwrap_or(0),
        b: raw.get(1).copied().unwrap_or(0),
    }
}

pub fn predict_summary_via(
    dispatcher: &impl SemanticExecutor,
    weights: &[u32],
) -> Result<CircuitSummary, SemanticExecutionError> {
    dispatcher
        .execute(1, 2)
        .map(unpack_only)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/clean_dispatch_map.rs", code)]);
    assert!(
        findings.is_empty(),
        "clean dispatch.map(unpack_only) must pass with zero findings, got: {findings:?}"
    );
}

#[test]
fn clean_dispatcher_with_gpu_reduction_chain_passes() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn reduce_any_via(
    dispatcher: &impl SemanticExecutor,
    data: &[u32],
) -> Result<bool, SemanticExecutionError> {
    let _ = dispatcher.execute(1, 2)?;
    Ok(true)
}

pub fn motif_matches_via(
    dispatcher: &impl SemanticExecutor,
    words: &[u32],
) -> Result<bool, SemanticExecutionError> {
    let _ = dispatcher.execute(1, 2)?;
    let witness = [1u32, 2];
    reduce_any_via(dispatcher, &witness)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/clean_gpu_reduction.rs", code)]);
    assert!(
        findings.is_empty(),
        "clean GPU reduction dispatch chain must pass, got: {findings:?}"
    );
}

#[test]
fn mutation_catches_unnamed_computed_const_referenced_by_expected_output() {
    let code = r#"
const ORACLE_SCALAR: u32 = 7 * 9;

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![vec![ORACLE_SCALAR as u8]]]
}
"#;
    let findings = analyze_files(&[("vyre-primitives/src/unnamed_const.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "computed const without EXPECTED/OUTPUT name token must be flagged via path resolution"
    );
    assert!(findings[0].message.contains("`ORACLE_SCALAR`"));
    assert_eq!(findings[0].line, Some(2));
}

#[test]
fn mutation_catches_static_facade_referenced_by_expected_output() {
    let code = r#"
static COMPUTED_DATA: [u32; 2] = [10 + 2, 20 * 3];

fn expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![crate::wire::pack_u32_slice(&COMPUTED_DATA)]]
}
"#;
    let findings = analyze_files(&[("vyre-primitives/src/static_facade.rs", code)]);
    assert!(
        !findings.is_empty(),
        "computed static referenced by expected_output must be flagged"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`COMPUTED_DATA`") || f.message.contains("expected_output")));
}

#[test]
fn mutation_oracle_detection_catches_dynamic_expected_output_oracle_invocation() {
    let code = format!(
        r#"
pub fn compute_twin_fixture(input: &[u32]) -> Vec<u8> {{
{body}
}}

fn expected_output() -> Vec<Vec<Vec<u8>>> {{
    vec![vec![compute_twin_fixture(&[1, 2, 3, 4])]]
}}
"#,
        body = incrementing_oracle_body()
    );
    let findings = analyze_files(&[("vyre-libs/src/op.rs", &code)]);
    assert!(
        !findings.is_empty(),
        "expected finding for expected_output dynamic oracle invocation"
    );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("compute_twin_fixture")
                || f.message.contains("expected_output")),
        "finding message should report dynamic execution in expected_output: {findings:?}"
    );
}

#[test]
fn production_compiler_analysis_with_callers_is_classified_as_reachable() {
    let code = r#"
use vyre_foundation::ir::Program;

pub fn analyze_cost_graph(nodes: &[u32]) -> u32 {
    let mut total = 0u32;
    for &n in nodes {
        total = total.wrapping_add(n);
    }
    total
}

pub fn compile_pipeline(nodes: &[u32]) -> Program {
    let _cost = analyze_cost_graph(nodes);
    Program::new()
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/analysis/cost_model.rs", code)]);
    assert!(
        findings.is_empty(),
        "compiler analysis reachable from IR builder root must not be flagged, got: {findings:?}"
    );
}

#[test]
fn mutation_catches_expected_output_target_also_called_by_builder() {
    let code = format!(
        r#"
use vyre_foundation::ir::Program;

pub fn compute_twin_fixture(input: &[u32]) -> Vec<u8> {{
{body}
}}

pub fn compile_pipeline() -> Program {{
    let _ = compute_twin_fixture(&[1, 2, 3]);
    Program::new()
}}

fn expected_output() -> Vec<Vec<Vec<u8>>> {{
    vec![vec![compute_twin_fixture(&[1, 2, 3, 4])]]
}}
"#,
        body = incrementing_oracle_body()
    );
    let findings = analyze_files(&[("vyre-libs/src/op.rs", &code)]);
    assert!(
        !findings.is_empty(),
        "dynamic expected_output call must be flagged even if target is called by a builder"
    );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("compute_twin_fixture")
                || f.message.contains("expected_output")),
        "finding message should report dynamic execution in expected_output: {findings:?}"
    );
}

#[test]
fn mutation_permits_side_effect_telemetry_unit_function() {
    let code = r#"
pub fn record_fixpoint_telemetry(step: usize, active_nodes: usize) {
    if step > 0 {
        let _ = active_nodes + step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_telemetry() {
        record_fixpoint_telemetry(1, 10);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/telemetry.rs", code)]);
    assert!(
        findings.is_empty(),
        "side-effect telemetry returning () must be permitted: {findings:?}"
    );
}

#[test]
fn mutation_permits_pure_validator_returning_result_unit() {
    let code = r#"
pub fn check_signature_invariants(inputs: usize, outputs: usize) -> Result<(), String> {
    if inputs == 0 || outputs == 0 {
        return Err("invalid shape".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_validator() {
        assert!(check_signature_invariants(2, 2).is_ok());
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/validator.rs", code)]);
    assert!(
        findings.is_empty(),
        "pure validator returning Result<(), E> must be permitted: {findings:?}"
    );
}

#[test]
fn mutation_permits_display_debug_error_impl_formatters() {
    let code = r#"
use std::fmt;

pub enum ErrorReason {
    InvalidInput(u32),
}

impl fmt::Display for ErrorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(code) => write!(f, "error: {code}"),
        }
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/display.rs", code)]);
    assert!(
        findings.is_empty(),
        "Display/Debug formatters must be permitted: {findings:?}"
    );
}

#[test]
fn mutation_permits_wire_byte_codec_without_arithmetic() {
    let code = r#"
pub fn encode_wire_header(dst: &mut [u8], magic: u32, version: u16) -> Result<usize, String> {
    if dst.len() < 6 {
        return Err("buffer too short".to_string());
    }
    dst[0..4].copy_from_slice(&magic.to_le_bytes());
    dst[4..6].copy_from_slice(&version.to_le_bytes());
    Ok(6)
}

pub fn decode_wire_header(src: &[u8]) -> Result<(u32, u16), String> {
    if src.len() < 6 {
        return Err("buffer too short".to_string());
    }
    let magic = u32::from_le_bytes(src[0..4].try_into().map_err(|_| "slice error")?);
    let version = u16::from_le_bytes(src[4..6].try_into().map_err(|_| "slice error")?);
    Ok((magic, version))
}
"#;
    let findings = analyze_files(&[("vyre-primitives/src/wire.rs", code)]);
    assert!(
        findings.is_empty(),
        "wire byte codecs without arithmetic transforms must be permitted: {findings:?}"
    );
}

#[test]
fn mutation_catches_post_dispatch_float_arithmetic_derivation() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn total_set_bits_via(dispatcher: &dyn SemanticExecutor, _input: &[u32]) -> Result<Vec<u8>, SemanticExecutionError> {
    let prog = Program::default();
    let out = dispatcher.execute(&prog, &[vec![]], None)?;
    Ok(out[0].clone())
}

pub fn saturation_ratio_via(dispatcher: &dyn SemanticExecutor, input: &[u32]) -> Result<f64, SemanticExecutionError> {
    if input.is_empty() {
        return Ok(0.0);
    }
    let capacity = (input.len() as u64) * 32;
    let set_bytes = total_set_bits_via(dispatcher, input)?;
    let set = u64::from(set_bytes[0]);
    Ok((set as f64) / (capacity as f64))
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/bitset_summary.rs", code)]);
    assert!(
        !findings.is_empty(),
        "post-dispatch float division metric must be convicted: {findings:?}"
    );
    assert!(
            findings.iter().any(|f| f.message.contains("post-dispatch host arithmetic / semantic derivation")),
            "post-dispatch float division must generate specific arithmetic derivation finding: {findings:?}"
        );
}

#[test]
fn mutation_catches_imported_alias_transitive_post_dispatch_derivation() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor as CustomDispatcher};

pub fn leaf_dispatch(d: &dyn CustomDispatcher, _input: &[u32]) -> Result<Vec<u8>, SemanticExecutionError> {
    let prog = Program::default();
    let out = d.execute(&prog, &[vec![]], None)?;
    Ok(out[0].clone())
}

pub fn caller_with_arithmetic(d: &dyn CustomDispatcher, input: &[u32]) -> Result<u64, SemanticExecutionError> {
    let bytes = leaf_dispatch(d, input)?;
    let val = u64::from(bytes[0]);
    Ok(val * 42)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/custom_alias.rs", code)]);
    assert!(
        !findings.is_empty(),
        "transitive post-dispatch arithmetic with imported alias must be convicted: {findings:?}"
    );
    assert!(
            findings.iter().any(|f| f.message.contains("post-dispatch host arithmetic / semantic derivation")),
            "imported alias transitive caller must generate specific arithmetic derivation finding: {findings:?}"
        );
}

#[test]
fn mutation_permits_post_dispatch_byte_unpacking_and_indexing() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn popcount_via(dispatcher: &dyn SemanticExecutor, _input: &[u32]) -> Result<Vec<u32>, SemanticExecutionError> {
    let prog = Program::default();
    let out = dispatcher.execute(&prog, &[vec![]], None)?;
    let raw_bytes = &out[0];
    unpack_words(raw_bytes)
}

fn unpack_words(raw_bytes: &[u8]) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut words = Vec::with_capacity(raw_bytes.len() / 4);
    for i in 0..(raw_bytes.len() / 4) {
        let word = u32::from_le_bytes(raw_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        words.push(word);
    }
    Ok(words)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/popcount_clean.rs", code)]);
    assert!(
        findings.is_empty(),
        "post-dispatch byte slice indexing and unpacking must be permitted: {findings:?}"
    );
}

#[test]
fn mutation_permits_ir_inspector_returning_bool_or_analysis_plan() {
    let code = r#"
use vyre_foundation::ir::Program;

pub fn is_bitset_equal_program(program: &Program) -> bool {
    program.is_valid()
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/bitset/equal.rs", code)]);
    assert!(
        findings.is_empty(),
        "IR inspector returning bool must be permitted: {findings:?}"
    );
}

#[test]
fn mutation_catches_generic_semantic_executor_bound_host_reduction() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn pass_conflicts_via<D: SemanticExecutor>(dispatcher: &D, _input: &[u32]) -> Result<bool, SemanticExecutionError> {
    let prog = Program::default();
    let out = dispatcher.execute(&prog, &[vec![]], None)?;
    Ok(out[0].iter().any(|&b| b != 0))
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/generic_dispatch.rs", code)]);
    assert!(
        !findings.is_empty(),
        "generic SemanticExecutor bounded function with post-dispatch reduction must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host reduction/aggregation `.any`")));
}
#[test]
fn mutation_catches_post_dispatch_integer_addition_derivation() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn popcount_plus_one_via(dispatcher: &dyn SemanticExecutor, _input: &[u32]) -> Result<u64, SemanticExecutionError> {
    let prog = Program::default();
    let out = dispatcher.execute(&prog, &[vec![]], None)?;
    let decoded_u32 = out[0][0] as u64;
    Ok(decoded_u32 + 1)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/plus_one.rs", code)]);
    assert!(
        !findings.is_empty(),
        "post-dispatch integer addition derivation must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host arithmetic / semantic derivation")));
}

#[test]
fn mutation_catches_post_dispatch_count_ones_reduction() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn popcount_count_ones_via(dispatcher: &dyn SemanticExecutor, _input: &[u32]) -> Result<u32, SemanticExecutionError> {
    let prog = Program::default();
    let out = dispatcher.execute(&prog, &[vec![]], None)?;
    let word = u32::from_le_bytes(out[0][0..4].try_into().unwrap());
    Ok(word.count_ones())
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/count_ones.rs", code)]);
    assert!(
        !findings.is_empty(),
        "post-dispatch count_ones reduction must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host reduction/aggregation `.count_ones`")));
}
