//! Unit tests for host oracle elimination gate (Part 2).

use super::host_oracle_elimination_test_fixtures::{incrementing_oracle_body, oracle_body};
use super::host_oracle_elimination_tests_part1::analyze_files;
use std::path::{Path, PathBuf};

#[test]
fn mutation_catches_spoofed_module_from_le_bytes_helper() {
    let code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

mod evil {
    pub struct u32;
    impl u32 {
        pub fn from_le_bytes(chunk: &[u8]) -> u32 {
            7
        }
    }
}

pub fn decode_with_spoofed_path(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_be_outputs(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_ne_outputs(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_dynamic_width(
    dispatcher: &dyn ProgramDispatcher,
    chunk_size: usize,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_with_fallback(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_with_default_fallback(
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn decode_with_table_lookup(
    dispatcher: &dyn ProgramDispatcher,
    lookup_table: &[u32],
) -> Result<Vec<u32>, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn predict_impact(dispatcher: &dyn ProgramDispatcher, out: &mut [u8]) -> Result<(), DispatchError> {
    dispatcher.dispatch_resident(&[], out)
}
"#;
    let module_b_code = r#"
use vyre_foundation::program_dispatch::ProgramDispatcher;

pub fn predict_impact(dispatcher: &dyn ProgramDispatcher) -> usize {
    // Unrelated helper in module_b with the same base name, but doing NO GPU dispatch
    0
}

pub fn validate_before_run(
    dispatcher: &dyn ProgramDispatcher,
) -> bool {
    let n = predict_impact(dispatcher);
    n == 0
}
"#;
    let module_c_code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};
use crate::module_a::predict_impact;

pub fn execute_and_filter(
    dispatcher: &dyn ProgramDispatcher,
    out: &mut [u8],
) -> Result<bool, DispatchError> {
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn project_with_borrows(
    dispatcher: &dyn ProgramDispatcher,
    cell: u32,
) -> Result<u32, DispatchError> {
    let mut raw = vec![0u8; 16];
    dispatcher.dispatch_resident(&[], &mut raw)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn execute_gpu_step(dispatcher: &dyn ProgramDispatcher, out: &mut [u8]) -> Result<(), DispatchError> {
    dispatcher.dispatch_resident(&[], out)
}
"#;
    let driver_code = r#"
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};
use crate::helper::execute_gpu_step;

pub fn run_pipeline_with_host_filter(
    dispatcher: &dyn ProgramDispatcher,
    target: u32,
) -> Result<bool, DispatchError> {
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub struct CircuitSummary {
    pub a: u32,
    pub b: u32,
}

pub fn predict_summary_via(
    dispatcher: &impl ProgramDispatcher,
    weights: &[u32],
) -> Result<CircuitSummary, DispatchError> {
    if weights.is_empty() {
        return Err(DispatchError::BadInputs("empty weights".to_string()));
    }
    let raw = dispatcher.dispatch(1, 2)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

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
    dispatcher: &impl ProgramDispatcher,
    weights: &[u32],
) -> Result<CircuitSummary, DispatchError> {
    dispatcher
        .dispatch(1, 2)
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn reduce_any_via(
    dispatcher: &impl ProgramDispatcher,
    data: &[u32],
) -> Result<bool, DispatchError> {
    let _ = dispatcher.dispatch(1, 2)?;
    Ok(true)
}

pub fn motif_matches_via(
    dispatcher: &impl ProgramDispatcher,
    words: &[u32],
) -> Result<bool, DispatchError> {
    let _ = dispatcher.dispatch(1, 2)?;
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
fn mutation_catches_local_dummy_dispatcher_masquerade() {
    let code = r#"
pub struct Dispatcher;

pub fn fake_dispatch(d: &Dispatcher, x: f32) -> f32 {
    x + 1.0
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/fake_dispatcher.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "locally declared Dispatcher masquerade must be flagged"
    );
    assert!(findings[0].message.contains("`fake_dispatch`"));
    assert_eq!(findings[0].line, Some(4));
}

#[test]
fn mutation_oracle_detection_catches_scalar_square_semantic_twin() {
    let code = r#"
pub fn square(x: f32) -> f32 {
    x * x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq() {
        assert_eq!(square(3.0), 9.0);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/math/scalar.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "scalar square semantic twin must be caught"
    );
    assert!(findings[0].message.contains("`square`"));
    assert_eq!(findings[0].line, Some(2));
}

#[test]
fn mutation_oracle_detection_catches_branch_classifier_returning_custom_enum() {
    let code = r#"
pub enum RegionClass {
    Small,
    Medium,
    Large,
}

pub fn classify_region(size: usize) -> RegionClass {
    if size < 10 {
        RegionClass::Small
    } else if size < 100 {
        RegionClass::Medium
    } else {
        RegionClass::Large
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class() {
        let _ = classify_region(5);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/pattern/classifier.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "branch classifier returning custom enum must be caught"
    );
    assert!(findings[0].message.contains("`classify_region`"));
    assert_eq!(findings[0].line, Some(8));
}

#[test]
fn mutation_oracle_detection_catches_scalar_bitwise_transform() {
    let code = r#"
pub fn compute_mask(tag: u32, shift: u32) -> u32 {
    (tag ^ 0xAA) << shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask() {
        assert_eq!(compute_mask(1, 2), 680);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/bitset/mask.rs", code)]);
    assert_eq!(findings.len(), 1, "scalar bitwise transform must be caught");
    assert!(findings[0].message.contains("`compute_mask`"));
    assert_eq!(findings[0].line, Some(2));
}

#[test]
fn mutation_oracle_detection_catches_numeric_methods_clamp_abs() {
    let code = r#"
pub fn normalize_weight(w: f32) -> f32 {
    w.abs().clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm() {
        assert_eq!(normalize_weight(-0.5), 0.5);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/math/norm.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "numeric method semantic twin must be caught"
    );
    assert!(findings[0].message.contains("`normalize_weight`"));
    assert_eq!(findings[0].line, Some(2));
}

#[test]
fn mutation_oracle_detection_catches_generic_named_host_twin_encode_payload() {
    let code = r#"
pub fn encode_payload(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &w in words {
        out.extend_from_slice(&w.wrapping_mul(31).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_payload() {
        let bytes = encode_payload(&[1, 2, 3]);
        assert_eq!(bytes.len(), 12);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/bitset/stochastic_compute.rs", code)]);
    assert!(
        !findings.is_empty(),
        "encode_payload unreached from production roots must be flagged"
    );
    assert!(
        findings.iter().any(|f| f
            .message
            .contains("unisolated host data-processing semantic twin `encode_payload`")),
        "finding message should name the unisolated semantic twin: {findings:?}"
    );
}

#[test]
fn mutation_catches_renamed_oracle_in_wire_path() {
    let code = r#"
pub fn pack_oracle(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &w in words {
        out.extend_from_slice(&w.wrapping_mul(31).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire() {
        let _ = pack_oracle(&[1, 2]);
    }
}
"#;
    let findings = analyze_files(&[("vyre-primitives/src/wire/adversarial.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "wire-named unreached oracle must be flagged"
    );
    assert!(findings[0].message.contains("`pack_oracle`"));
    assert_eq!(findings[0].line, Some(2));
}

#[test]
fn mutation_catches_renamed_oracle_with_parse_witness_report_name() {
    let code = r#"
pub fn parse_witness_report(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &w in words {
        out.extend_from_slice(&w.wrapping_add(1).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let _ = parse_witness_report(&[1, 2]);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/security/witness.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "parse/witness/report named unreached oracle must be flagged"
    );
    assert!(findings[0].message.contains("`parse_witness_report`"));
    assert_eq!(findings[0].line, Some(2));
}

#[test]
fn mutation_catches_validator_like_result_bool_oracle() {
    let code = r#"
pub fn check_valid_witness(words: &[u32]) -> Result<bool, String> {
    let mut hash = 0u32;
    for &w in words {
        hash = hash.wrapping_mul(37).wrapping_add(w);
    }
    Ok(hash == 42)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check() {
        let _ = check_valid_witness(&[1, 2, 3]);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/validation/adversarial.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "Result<bool> computational oracle must be flagged"
    );
    assert!(findings[0].message.contains("`check_valid_witness`"));
    assert_eq!(findings[0].line, Some(2));
}

#[test]
fn mutation_catches_table_like_name_oracle() {
    let code = r#"
pub fn generate_table_oracle(words: &[u32]) -> Vec<u32> {
    words.iter().map(|w| w.wrapping_add(1)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen() {
        let _ = generate_table_oracle(&[1, 2]);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/tables/adversarial.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "table-named uncalled oracle must be flagged"
    );
    assert!(findings[0].message.contains("`generate_table_oracle`"));
    assert_eq!(findings[0].line, Some(2));
}

#[test]
fn mutation_catches_two_function_cycle_evasion() {
    let code = r#"
pub fn encode_step_a(words: &[u32]) -> Vec<u8> {
    encode_step_b(words)
}

pub fn encode_step_b(words: &[u32]) -> Vec<u8> {
    if words.is_empty() {
        return Vec::new();
    }
    encode_step_a(&words[1..])
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/cycle.rs", code)]);
    assert_eq!(
        findings.len(),
        2,
        "mutual recursion cycle without production roots must flag both functions"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`encode_step_a`")));
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`encode_step_b`")));
}

#[test]
fn mutation_catches_ungated_tests_module() {
    let code = format!(
        r#"
mod tests {{
    pub fn host_oracle_helper(input: &[u32]) -> Vec<u8> {{
{body}
    }}
}}
"#,
        body = incrementing_oracle_body()
    );
    let findings = analyze_files(&[("vyre-libs/src/ungated.rs", &code)]);
    assert_eq!(
        findings.len(),
        1,
        "ungated mod tests must be judged as production code and flag uncalled oracle"
    );
    assert!(findings[0].message.contains("`host_oracle_helper`"));
}

#[test]
fn mutation_catches_oracle_under_cfg_any_test_and_cpu_parity() {
    let code = r#"
#[cfg(any(test, feature = "cpu-parity"))]
pub fn stochastic_decode(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/bitset/stochastic_compute.rs", code)]);
    assert!(
        !findings.is_empty(),
        "cfg(any(test, feature = ...)) is not test-only and must be judged as production code"
    );
    assert!(
        findings.iter().any(|f| f
            .message
            .contains("unisolated host data-processing semantic twin `stochastic_decode`")),
        "finding message should report unisolated semantic twin under cfg(any): {findings:?}"
    );
}

#[test]
fn mutation_catches_name_collision_uncalled_oracle() {
    let code_a = format!(
        r#"
use vyre_foundation::ir::Program;

pub fn process_stream(input: &[u32]) -> Vec<u8> {{
{body}
}}

pub fn compile_domain_a() -> Program {{
    let _ = process_stream(&[]);
    Program::new()
}}
"#,
        body = incrementing_oracle_body()
    );
    let code_b = format!(
        r#"
pub fn process_stream(input: &[u32]) -> Vec<u8> {{
{body}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_b() {{
        let _ = process_stream(&[1, 2]);
    }}
}}
"#,
        body = oracle_body("wrapping_mul(7)")
    );
    let findings = analyze_files(&[
        ("vyre-libs/src/domain_a/mod.rs", &code_a),
        ("vyre-libs/src/domain_b/mod.rs", &code_b),
    ]);
    assert_eq!(
        findings.len(),
        1,
        "name collision must not hide uncalled oracle in domain_b"
    );
    assert_eq!(
        findings[0].file,
        Some(PathBuf::from("vyre-libs/src/domain_b/mod.rs"))
    );
    assert_eq!(findings[0].line, Some(2));
}

#[test]
fn mutation_distinguishes_same_named_methods_in_different_impl_blocks() {
    let code = r#"
use vyre_foundation::ir::Program;

pub struct ProductionPipeline;
pub struct UnusedOracleType;

impl ProductionPipeline {
    pub fn process(&self) -> Program {
        Program::new()
    }
}

impl UnusedOracleType {
    pub fn process(&self, words: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &w in words {
            out.extend_from_slice(&w.wrapping_mul(13).to_le_bytes());
        }
        out
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/impl_collision.rs", code)]);
    assert_eq!(
        findings.len(),
        1,
        "UnusedOracleType::process must be flagged as unreached semantic twin"
    );
    assert!(findings[0].message.contains("`process`"));
    assert_eq!(findings[0].line, Some(14));
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
fn mutation_catches_zero_arg_oracle_behind_facade() {
    let code = r#"
pub fn zero_arg_oracle() -> u32 {
    let mut total = 0u32;
    for i in 0..10 {
        total = total.wrapping_add(i);
    }
    total
}

pub fn facade() -> u32 {
    zero_arg_oracle()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facade() {
        assert_eq!(facade(), 45);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/facade.rs", code)]);
    assert!(
        !findings.is_empty(),
        "zero-arg oracle behind facade must be flagged"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`zero_arg_oracle`") || f.message.contains("`facade`")));
}

#[test]
fn mutation_catches_owned_vec_input_oracle() {
    let code = r#"
pub fn process_owned_words(words: Vec<u32>) -> Vec<u8> {
    let mut out = Vec::new();
    for w in words {
        out.extend_from_slice(&w.wrapping_mul(17).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owned() {
        let _ = process_owned_words(vec![1, 2, 3]);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/owned.rs", code)]);
    assert!(
        !findings.is_empty(),
        "owned Vec input arithmetic candidate must be flagged"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`process_owned_words`")));
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
fn mutation_catches_unreached_binary_search_or_partition_oracle() {
    let code = r#"
pub fn region_lookup(pos: u32, boundaries: &[u32]) -> usize {
    match boundaries.binary_search(&pos) {
        Ok(idx) => idx,
        Err(idx) => idx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup() {
        assert_eq!(region_lookup(5, &[0, 10, 20]), 1);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/pattern/lookup.rs", code)]);
    assert!(
        !findings.is_empty(),
        "unreached binary_search candidate must be flagged"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`region_lookup`")));
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
fn mutation_catches_unreached_semantic_popcount_reduction() {
    let code = r#"
pub fn checked_frontier_popcount(frontier: &[u64]) -> usize {
    let mut count = 0;
    for &word in frontier {
        count += word.count_ones() as usize;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_popcount() {
        assert_eq!(checked_frontier_popcount(&[0b111]), 3);
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/bitset/frontier.rs", code)]);
    assert!(
        !findings.is_empty(),
        "unreached host bitset popcount reduction must be convicted"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`checked_frontier_popcount`")));
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
fn mutation_catches_wire_codec_with_semantic_data_math() {
    let code = r#"
pub fn encode_transformed_payload(dst: &mut [u8], data: &[u32]) -> Result<usize, String> {
    let mut offset = 0;
    for &elem in data {
        let transformed = elem.wrapping_mul(31).wrapping_add(7);
        dst[offset..offset + 4].copy_from_slice(&transformed.to_le_bytes());
        offset += 4;
    }
    Ok(offset)
}
"#;
    let findings = analyze_files(&[("vyre-primitives/src/wire_math.rs", code)]);
    assert!(
        !findings.is_empty(),
        "codecs performing semantic data math must be convicted"
    );
    assert!(findings
        .iter()
        .any(|f| f.message.contains("`encode_transformed_payload`")));
}

#[test]
fn mutation_catches_post_dispatch_float_arithmetic_derivation() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn total_set_bits_via(dispatcher: &dyn ProgramDispatcher, _input: &[u32]) -> Result<Vec<u8>, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    Ok(out[0].clone())
}

pub fn saturation_ratio_via(dispatcher: &dyn ProgramDispatcher, input: &[u32]) -> Result<f64, DispatchError> {
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher as CustomDispatcher};

pub fn leaf_dispatch(d: &dyn CustomDispatcher, _input: &[u32]) -> Result<Vec<u8>, DispatchError> {
    let prog = Program::default();
    let out = d.dispatch(&prog, &[vec![]], None)?;
    Ok(out[0].clone())
}

pub fn caller_with_arithmetic(d: &dyn CustomDispatcher, input: &[u32]) -> Result<u64, DispatchError> {
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn popcount_via(dispatcher: &dyn ProgramDispatcher, _input: &[u32]) -> Result<Vec<u32>, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    let raw_bytes = &out[0];
    unpack_words(raw_bytes)
}

fn unpack_words(raw_bytes: &[u8]) -> Result<Vec<u32>, DispatchError> {
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
fn mutation_catches_generic_program_dispatcher_bound_host_reduction() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn pass_conflicts_via<D: ProgramDispatcher>(dispatcher: &D, _input: &[u32]) -> Result<bool, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
    Ok(out[0].iter().any(|&b| b != 0))
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/generic_dispatch.rs", code)]);
    assert!(
        !findings.is_empty(),
        "generic ProgramDispatcher bounded function with post-dispatch reduction must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host reduction/aggregation `.any`")));
}
#[test]
fn mutation_catches_post_dispatch_integer_addition_derivation() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn popcount_plus_one_via(dispatcher: &dyn ProgramDispatcher, _input: &[u32]) -> Result<u64, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
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
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub fn popcount_count_ones_via(dispatcher: &dyn ProgramDispatcher, _input: &[u32]) -> Result<u32, DispatchError> {
    let prog = Program::default();
    let out = dispatcher.dispatch(&prog, &[vec![]], None)?;
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
