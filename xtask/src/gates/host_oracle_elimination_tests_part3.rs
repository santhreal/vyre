//! Unit tests for host oracle elimination gate (Part 3).

use std::path::PathBuf;

use crate::gates::scan::{attribute_is_test_only, test_module_files, Tree};

use super::host_oracle_elimination_eval::analyze_sources;
use super::host_oracle_elimination_records::TARGET_ROOTS;
use super::host_oracle_elimination_test_fixtures::{
    canonical_dispatch_fn, resident_staging_source, self_binding_staging_source,
    CANONICAL_DISPATCH_IMPORTS, CANONICAL_REQUEST_ARGUMENTS,
};
use super::host_oracle_elimination_tests_part1::analyze_files;

#[test]
fn mutation_catches_post_dispatch_integer_division_derivation() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn average_via(dispatcher: &dyn SemanticExecutor, input: &[u32]) -> Result<u64, SemanticExecutionError> {
    let prog = Program::default();
    let out = dispatcher.execute(&prog, &[vec![]], None)?;
    let total = u64::from(out[0][0]);
    let count = input.len() as u64;
    Ok(total / count)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/average.rs", code)]);
    assert!(
        !findings.is_empty(),
        "post-dispatch integer division derivation must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host arithmetic / semantic derivation")));
}
#[test]
fn mutation_catches_metadata_only_dispatcher_call_not_establishing_execution() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn fake_oracle_capabilities_only(dispatcher: &dyn SemanticExecutor, input: &[u32]) -> u32 {
    let _caps = dispatcher.capabilities();
    let mut sum = 0u32;
    for &x in input {
        sum = sum.wrapping_add(x);
    }
    sum
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/fake_oracle.rs", code)]);
    assert!(
        !findings.is_empty(),
        "metadata-only dispatcher caller must be convicted as unisolated host algorithm"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("unisolated host data-processing semantic twin")));
}

#[test]
fn mutation_catches_unrelated_field_receiver_masquerading_as_dispatch() {
    let code = r#"
use vyre_megakernel::SemanticExecutor;

struct LocalDevice;
impl LocalDevice {
    fn dispatch(&self, _left: u32, _right: u32) {}
}

struct LocalContext {
    device: LocalDevice,
}

pub fn fake_field_dispatch(
    _dispatcher: &dyn SemanticExecutor,
    input: &[u32],
) -> u32 {
    let context = LocalContext { device: LocalDevice };
    context.device.execute(1, 2);
    input.iter().copied().sum()
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/fake_field_dispatch.rs", code)]);
    assert!(
            findings.iter().any(|finding| {
                finding
                    .message
                    .contains("unisolated host data-processing semantic twin")
            }),
            "a field receiver unrelated to the canonical dispatcher parameter must not establish a GPU execution root: {findings:?}"
        );
}

#[test]
fn mutation_catches_non_dispatching_helper_not_establishing_execution() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

fn record_dispatcher(_dispatcher: &dyn SemanticExecutor) {
    // telemetry/record only, does not dispatch
}

pub fn fake_oracle_with_record(dispatcher: &dyn SemanticExecutor, input: &[u32]) -> u32 {
    record_dispatcher(dispatcher);
    let mut sum = 0u32;
    for &x in input {
        sum = sum.wrapping_add(x);
    }
    sum
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/fake_record.rs", code)]);
    assert!(
        !findings.is_empty(),
        "non-dispatching helper caller must be convicted as unisolated host algorithm"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("unisolated host data-processing semantic twin")));
}

#[test]
fn mutation_permits_transitive_dispatch_helper_execution() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

fn helper_dispatch(dispatcher: &dyn SemanticExecutor, _input: &[u32]) -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
    let prog = Program::default();
    dispatcher.execute(&prog, &[vec![]], None)
}

pub fn wrapper_dispatch_via(dispatcher: &dyn SemanticExecutor, input: &[u32]) -> Result<Vec<u8>, SemanticExecutionError> {
    let out = helper_dispatch(dispatcher, input)?;
    Ok(out[0].clone())
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/wrapper_dispatch.rs", code)]);
    assert!(
        findings.is_empty(),
        "transitive dispatch helper must be recognized as valid GPU dispatch root: {findings:?}"
    );
}
#[test]
fn mutation_catches_generic_masquerade_with_similar_ident() {
    let code = r#"
pub struct NotD;

pub fn fake_oracle_with_not_d<D: vyre_megakernel::SemanticExecutor>(
    not_d: &NotD,
    input: &[u32],
) -> u32 {
    let mut sum = 0u32;
    for &x in input {
        sum = sum.wrapping_add(x);
    }
    sum
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/encoding/not_d.rs", code)]);
    assert!(
            !findings.is_empty(),
            "function taking NotD where D is bounded by SemanticExecutor must be convicted as unisolated host algorithm"
        );
    assert!(findings.iter().any(|f| f
        .message
        .contains("unisolated host data-processing semantic twin")));
}
#[test]
fn mutation_permits_legitimate_transpose_input_staging() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn forward_backward_via(
    dispatcher: &dyn SemanticExecutor,
    adj: &[u32],
    n: usize,
) -> Result<(Vec<u8>, Vec<u8>), SemanticExecutionError> {
    let prog1 = Program::default();
    let fwd = dispatcher.execute(&prog1, &[vec![]], None)?;
    let mut transpose = vec![0u32; n * n];
    for i in 0..n {
        for j in 0..n {
            transpose[j * n + i] = adj[i * n + j];
        }
    }
    let prog2 = Program::default();
    let bwd = dispatcher.execute(&prog2, &[transpose], None)?;
    Ok((fwd[0].clone(), bwd[0].clone()))
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/staging/transpose.rs", code)]);
    assert!(
            findings.is_empty(),
            "legitimate inter-dispatch input matrix transpose staging must be permitted with zero findings: {findings:?}"
        );
}

#[test]
fn mutation_catches_unrelated_sum_between_dispatches_returned_afterward() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn evasion_unrelated_sum_via(
    dispatcher: &dyn SemanticExecutor,
    _input: &[u32],
) -> Result<(u32, Vec<u8>), SemanticExecutionError> {
    let prog1 = Program::default();
    let out1 = dispatcher.execute(&prog1, &[vec![]], None)?;
    let host_sum = out1[0].iter().map(|&x| x as u32).sum::<u32>();
    let prog2 = Program::default();
    let out2 = dispatcher.execute(&prog2, &[vec![]], None)?;
    Ok((host_sum, out2[0].clone()))
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/evasion/unrelated_sum.rs", code)]);
    assert!(
        !findings.is_empty(),
        "unrelated sum between dispatches returned afterward must be convicted"
    );
    assert!(
        findings.iter().any(|f| f
            .message
            .contains("post-dispatch host reduction/aggregation `.sum`")),
        "must convict with reduction finding: {findings:?}"
    );
}

#[test]
fn mutation_catches_unrelated_semantic_side_effect_between_dispatches() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn evasion_side_effect_via(
    dispatcher: &dyn SemanticExecutor,
    acc: &mut Vec<u32>,
) -> Result<Vec<u8>, SemanticExecutionError> {
    let prog1 = Program::default();
    let out1 = dispatcher.execute(&prog1, &[vec![]], None)?;
    for &b in &out1[0] {
        acc.push((b as u32) * 2);
    }
    let prog2 = Program::default();
    let out2 = dispatcher.execute(&prog2, &[vec![]], None)?;
    Ok(out2[0].clone())
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/evasion/side_effect.rs", code)]);
    assert!(
        !findings.is_empty(),
        "unrelated semantic side effect between dispatches must be convicted"
    );
    assert!(
        findings.iter().any(
            |f| f.message.contains("post-dispatch host loop/accumulation")
                || f.message
                    .contains("post-dispatch host arithmetic / semantic derivation")
        ),
        "must convict with loop/arithmetic finding: {findings:?}"
    );
}

#[test]
fn mutation_catches_terminal_post_dispatch_math() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn evasion_terminal_math_via(
    dispatcher: &dyn SemanticExecutor,
    _input: &[u32],
) -> Result<u64, SemanticExecutionError> {
    let prog1 = Program::default();
    let _out1 = dispatcher.execute(&prog1, &[vec![]], None)?;
    let prog2 = Program::default();
    let out2 = dispatcher.execute(&prog2, &[vec![]], None)?;
    let total = (out2[0][0] as u64) + 100;
    Ok(total)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/evasion/terminal_math.rs", code)]);
    assert!(
        !findings.is_empty(),
        "terminal post-dispatch math must be convicted"
    );
    assert!(
        findings.iter().any(|f| f
            .message
            .contains("post-dispatch host arithmetic / semantic derivation on GPU results")),
        "must convict with arithmetic finding: {findings:?}"
    );
}
#[test]
fn mutation_permits_arbitrary_index_names_in_inter_dispatch_staging() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn staged_arbitrary_names_into(
    dispatcher: &dyn SemanticExecutor,
    matrix: &[u32],
    dim_rows: usize,
    dim_cols: usize,
) -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
    let mut staged_transpose = vec![0u8; dim_rows * dim_cols * 4];
    for row_arbitrary_alpha in 0..dim_rows {
        for col_arbitrary_beta in 0..dim_cols {
            let src_k = row_arbitrary_alpha * dim_cols + col_arbitrary_beta;
            let dst_m = col_arbitrary_beta * dim_rows + row_arbitrary_alpha;
            let val = matrix[src_k];
            staged_transpose[dst_m * 4] = (val & 0xFF) as u8;
        }
    }
    let prog = Program::default();
    dispatcher.execute(&prog, &[staged_transpose], None)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/clean_staging_names.rs", code)]);
    assert!(
        findings.is_empty(),
        "arbitrarily named index staging feeding dispatch must be permitted: {findings:?}"
    );
}

#[test]
fn mutation_catches_semantic_scalar_named_index_or_idx() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn evasion_semantic_scalar_named_idx(
    dispatcher: &dyn SemanticExecutor,
) -> Result<u64, SemanticExecutionError> {
    let prog = Program::default();
    let out = dispatcher.execute(&prog, &[vec![]], None)?;
    let idx = (out[0][0] as u64) + 10;
    let index = idx * 2;
    Ok(index)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/evasion_named_idx.rs", code)]);
    assert!(
        !findings.is_empty(),
        "semantic derivation named index/idx must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host arithmetic / semantic derivation")));
}

#[test]
fn mutation_catches_post_dispatch_decoded_value_comparison() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn evasion_post_dispatch_comparison(
    dispatcher: &dyn SemanticExecutor,
    threshold: u8,
) -> Result<bool, SemanticExecutionError> {
    let prog = Program::default();
    let out = dispatcher.execute(&prog, &[vec![]], None)?;
    let decoded_byte = out[0][0];
    if decoded_byte > threshold {
        Ok(true)
    } else {
        Ok(false)
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/evasion_comparison.rs", code)]);
    assert!(
        !findings.is_empty(),
        "post-dispatch decoded value comparison must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host arithmetic / semantic derivation")));
}

#[test]
fn mutation_catches_post_dispatch_reconstruction_loop_with_scalar_math() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn evasion_reconstruction_with_math_into(
    dispatcher: &dyn SemanticExecutor,
    outputs: &mut Vec<Vec<u8>>,
) -> Result<(), SemanticExecutionError> {
    let readbacks = dispatcher.execute(&vec![], &[vec![]], None)?;
    for (output, readback) in outputs.iter_mut().zip(&readbacks) {
        output.clear();
        output.extend_from_slice(readback);
        output.push(readback[0] * 2);
    }
    Ok(())
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/evasion_copy_math.rs", code)]);
    assert!(
        !findings.is_empty(),
        "reconstruction loop with scalar arithmetic must be convicted"
    );
    assert!(findings.iter().any(
        |f| f.message.contains("post-dispatch host loop/accumulation")
            || f.message.contains("post-dispatch host arithmetic")
    ));
}

#[test]
fn mutation_catches_post_dispatch_decoder_loop_with_accumulation() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

fn decode_u32_output_exact(
    _readback: &[u8],
    _expected_words: usize,
    _context: &str,
    _out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    Ok(())
}

pub fn evasion_decode_batch_into(
    dispatcher: &dyn SemanticExecutor,
    mut outs: Vec<(usize, &'static str, &mut Vec<u32>)>,
) -> Result<u32, SemanticExecutionError> {
    let readbacks = dispatcher.execute(&vec![], &[vec![]], None)?;
    let mut total_words = 0u32;
    for (index, (expected_words, context, out)) in outs.into_iter().enumerate() {
        decode_u32_output_exact(&readbacks[index], expected_words, context, out)?;
        total_words += expected_words as u32;
    }
    Ok(total_words)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/evasion_decoder_accum.rs", code)]);
    assert!(
        !findings.is_empty(),
        "decoder loop with host accumulation must be convicted"
    );
    assert!(findings.iter().any(
        |f| f.message.contains("post-dispatch host loop/accumulation")
            || f.message.contains("post-dispatch host arithmetic")
    ));
}

#[test]
fn mutation_catches_post_dispatch_output_base_arithmetic_derivation() {
    let code = r#"
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn evasion_output_base_scalar_math_via(
    dispatcher: &dyn SemanticExecutor,
) -> Result<u64, SemanticExecutionError> {
    let outputs = dispatcher.execute(&vec![], &[vec![]], None)?;
    let output_base = outputs[0][0] as u64;
    let computed = output_base + 42;
    Ok(computed)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/evasion_output_base_math.rs", code)]);
    assert!(
        !findings.is_empty(),
        "arithmetic on output_base derived from output byte must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("post-dispatch host arithmetic / semantic derivation")));
}
#[test]
fn mutation_permits_turbofish_generic_calls_and_cfg_alternative_definitions() {
    let wire_code = r#"
#[cfg(target_endian = "little")]
pub fn fill_custom_words_into<T: Copy>(src: &[u8], count: usize, out: &mut Vec<T>) {
    let _ = (src, count, out);
}

#[cfg(target_endian = "big")]
pub fn fill_custom_words_into<T: Copy>(src: &[u8], count: usize, out: &mut Vec<T>) {
    let _ = (src, count, out);
}

pub fn unpack_custom_u32_slice_into(src: &[u8], count: usize, out: &mut Vec<u32>) {
    fill_custom_words_into::<u32>(src, count, out);
}
"#;
    let caller_code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn resident_caller_using_unpack(
    dispatcher: &dyn SemanticExecutor,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let readbacks = dispatcher.execute(&vec![], &[vec![]], None)?;
    crate::wire::unpack_custom_u32_slice_into(&readbacks[0], 10, out);
    Ok(())
}
"#;
    let findings = analyze_files(&[
        ("vyre-primitives/src/wire.rs", wire_code),
        ("vyre-libs/src/caller.rs", caller_code),
    ]);
    assert!(
        findings.is_empty(),
        "turbofish generic caller reaching CFG-alternative definitions must be clean: {findings:?}"
    );
}

#[test]
fn mutation_catches_unreachable_generic_cfg_alternative_definition() {
    let wire_code = r#"
#[cfg(target_endian = "little")]
pub fn uncalled_twin_words_into<T: Copy>(src: &[u8], count: usize, out: &mut Vec<T>) {
    let mut sum = 0usize;
    for &b in src {
        sum += b as usize;
    }
    let _ = (sum, count, out);
}

#[cfg(target_endian = "big")]
pub fn uncalled_twin_words_into<T: Copy>(src: &[u8], count: usize, out: &mut Vec<T>) {
    let mut sum = 0usize;
    for &b in src {
        sum += b as usize;
    }
    let _ = (sum, count, out);
}
"#;
    let findings = analyze_files(&[("vyre-primitives/src/wire.rs", wire_code)]);
    assert!(
        !findings.is_empty(),
        "uncalled CFG alternative twin must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("unisolated host data-processing semantic twin")));
}

#[test]
fn mutation_permits_operation_metadata_iterator_with_arbitrary_name() {
    let code = r#"
use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};

pub fn arbitrary_catalog_query_into() -> impl Iterator<Item = SemanticOperation> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Library)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/catalog.rs", code)]);
    assert!(
        findings.is_empty(),
        "operation metadata iterator must be permitted: {findings:?}"
    );
}

#[test]
fn mutation_catches_adversarial_numeric_iterator_twin_with_same_shape() {
    let code = r#"
pub fn arbitrary_catalog_query_into(data: &[u32]) -> impl Iterator<Item = u32> + '_ {
    data.iter().map(|&x| x + 42)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/catalog_adversarial.rs", code)]);
    assert!(
        !findings.is_empty(),
        "unisolated numeric iterator twin must be convicted"
    );
    assert!(findings.iter().any(|f| f
        .message
        .contains("unisolated host data-processing semantic twin")));
}
#[test]
fn mutation_catches_local_type_masquerading_as_operation_metadata() {
    let code = r#"
pub struct SemanticOperation(u32);

pub fn arbitrary_catalog_query_into(
    data: &[u32],
) -> impl Iterator<Item = SemanticOperation> + '_ {
    data.iter().map(|&value| SemanticOperation(value + 42))
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/catalog_masquerade.rs", code)]);
    assert!(
        findings.iter().any(|finding| finding
            .message
            .contains("unisolated host data-processing semantic twin")),
        "a local same-named type must not receive canonical metadata treatment: {findings:?}"
    );
}

#[test]
fn mutation_permits_genuine_resident_staging_consumed_by_canonical_dispatch() {
    let staging_code = resident_staging_source(
        "ResidentDemoGraph",
        "pub ",
        &[("upload_resident_demo_graph", "0x5A5A_5A5A")],
    );
    let dispatch_code = format!(
        "{CANONICAL_DISPATCH_IMPORTS}use crate::staging::{{upload_resident_demo_graph, ResidentDemoGraph}};
{dispatch}
pub fn run_demo_traversal(
    dispatcher: &impl SemanticExecutor,
    logical: &LogicalProgramGraph<'_>,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    edges: &[u32],
) -> Result<(), SemanticExecutionError> {{
    let graph = upload_resident_demo_graph(node_count, edges)?;
    execute_demo_traversal(dispatcher, logical, policy, &graph)
}}
",
        dispatch = canonical_dispatch_fn("pub ", "execute_demo_traversal", "ResidentDemoGraph", "", "")
    );
    let findings = analyze_files(&[
        ("vyre-libs/src/staging.rs", staging_code.as_str()),
        ("vyre-libs/src/dispatch.rs", dispatch_code.as_str()),
    ]);
    assert!(
        findings.is_empty(),
        "resident staging consumed by genuine canonical dispatch must not be convicted: {findings:?}"
    );
}

#[test]
fn mutation_catches_same_basename_in_different_modules_rejected() {
    let staging_a = resident_staging_source(
        "ResidentDemoGraph",
        "pub ",
        &[("upload_resident_demo_graph", "0x5A5A_5A5A")],
    );
    let dispatch_b = format!(
        "{CANONICAL_DISPATCH_IMPORTS}
pub struct ResidentDemoGraph {{
    pub packed: Vec<u8>,
}}
{dispatch}
pub fn run_b(
    dispatcher: &impl SemanticExecutor,
    logical: &LogicalProgramGraph<'_>,
    policy: &SemanticExecutionPolicy,
    graph: &ResidentDemoGraph,
) -> Result<(), SemanticExecutionError> {{
    execute_demo_traversal(dispatcher, logical, policy, graph)
}}
",
        dispatch = canonical_dispatch_fn(
            "pub ",
            "execute_demo_traversal",
            "ResidentDemoGraph",
            "",
            ""
        )
    );
    let findings = analyze_files(&[
        ("vyre-libs/src/staging_a.rs", staging_a.as_str()),
        ("vyre-libs/src/dispatch_b.rs", dispatch_b.as_str()),
    ]);
    assert!(
        !findings.is_empty(),
        "same basename in different modules must be rejected and convicted"
    );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("upload_resident_demo_graph")),
        "upload_resident_demo_graph in staging_a must be convicted: {findings:?}"
    );
}

#[test]
fn mutation_catches_unused_alternative_producer_rejected() {
    let staging_code = resident_staging_source(
        "ResidentDemoGraph",
        "pub ",
        &[
            ("upload_resident_demo_graph", "0x5A5A_5A5A"),
            ("upload_unused_alt_graph", "0x1234_5678"),
        ],
    );
    let dispatch_code = format!(
        "{CANONICAL_DISPATCH_IMPORTS}use crate::staging::{{upload_resident_demo_graph, ResidentDemoGraph}};
{dispatch}
pub fn run_demo_traversal(
    dispatcher: &impl SemanticExecutor,
    logical: &LogicalProgramGraph<'_>,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    edges: &[u32],
) -> Result<(), SemanticExecutionError> {{
    let graph = upload_resident_demo_graph(node_count, edges)?;
    execute_demo_traversal(dispatcher, logical, policy, &graph)
}}
",
        dispatch = canonical_dispatch_fn("pub ", "execute_demo_traversal", "ResidentDemoGraph", "", "")
    );
    let findings = analyze_files(&[
        ("vyre-libs/src/staging.rs", staging_code.as_str()),
        ("vyre-libs/src/dispatch.rs", dispatch_code.as_str()),
    ]);
    assert!(
        !findings.is_empty(),
        "unused alternative producer returning same nominal type must be convicted"
    );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("upload_unused_alt_graph")),
        "upload_unused_alt_graph must be convicted: {findings:?}"
    );
    assert!(
        !findings
            .iter()
            .any(|f| f.message.contains("upload_resident_demo_graph")),
        "used producer upload_resident_demo_graph must NOT be convicted: {findings:?}"
    );
}

#[test]
fn mutation_catches_unrelated_upload_only_host_math_not_consumed_by_dispatch() {
    let upload_only_code = r#"
use vyre_megakernel::SemanticExecutionError;

pub struct UnusedResidentGraph {
    pub packed: Vec<u8>,
}

pub fn upload_unused_graph_with_math(
    edges: &[u32],
) -> Result<UnusedResidentGraph, SemanticExecutionError> {
    let mut packed = Vec::new();
    for &e in edges {
        packed.extend_from_slice(&(e ^ 0x1234_5678).to_le_bytes());
    }
    Ok(UnusedResidentGraph { packed })
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/upload_only.rs", upload_only_code)]);
    assert!(
        !findings.is_empty(),
        "upload-only host math with no downstream dispatch root must be convicted"
    );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("upload_unused_graph_with_math")),
        "upload_unused_graph_with_math must be flagged: {findings:?}"
    );
}

#[test]
fn mutation_catches_resident_staging_values_not_uploaded_before_dispatch() {
    let code = format!(
        "{CANONICAL_DISPATCH_IMPORTS}
struct ResidentGraph {{
    packed: Vec<u8>,
}}

fn prepare_graph_without_binding(edges: &[u32]) -> ResidentGraph {{
    let mut packed = Vec::with_capacity(edges.len() * 4);
    for &edge in edges {{
        packed.extend_from_slice(&edge.wrapping_mul(3).to_le_bytes());
    }}
    ResidentGraph {{ packed }}
}}

pub fn run_graph(
    dispatcher: &impl SemanticExecutor,
    logical: &LogicalProgramGraph<'_>,
    policy: &SemanticExecutionPolicy,
    edges: &[u32],
) -> Result<(), SemanticExecutionError> {{
    let graph = prepare_graph_without_binding(edges);
    let _ = graph;
    let inputs = BTreeMap::new();
    let request = SemanticExecutionRequest::new(
{CANONICAL_REQUEST_ARGUMENTS}
    )?;
    dispatcher.execute(&request)?;
    Ok(())
}}
"
    );
    let findings = analyze_files(&[("vyre-libs/src/unconsumed_staging.rs", code.as_str())]);
    assert!(
        findings
            .iter()
            .any(|finding| finding.message.contains("prepare_graph_without_binding")),
        "host staging whose bytes never reach the seam must be convicted: {findings:?}"
    );
}

#[test]
fn mutation_catches_staging_consumed_only_by_fake_local_dispatcher_masquerade() {
    let fake_code = r#"
pub struct FakeResidentGraph {
    pub handle: u64,
}

pub trait SemanticExecutor {
    fn alloc_resident(&self, bytes: usize) -> Result<u64, String>;
    fn dispatch(&self, prog: u32, handles: &[u64]) -> Result<(), String>;
}

pub fn upload_fake_graph(
    dispatcher: &impl SemanticExecutor,
    edges: &[u32],
) -> Result<FakeResidentGraph, String> {
    let mut sum = 0u32;
    for &e in edges {
        sum = sum.wrapping_add(e ^ 0xA5A5);
    }
    let handle = dispatcher.alloc_resident(sum as usize)?;
    Ok(FakeResidentGraph { handle })
}

pub fn fake_dispatch(
    dispatcher: &impl SemanticExecutor,
    graph: &FakeResidentGraph,
) -> Result<(), String> {
    dispatcher.execute(1, &[graph.handle])
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/fake_staging.rs", fake_code)]);
    assert!(
        !findings.is_empty(),
        "staging with fake local dispatcher masquerade must be convicted"
    );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("upload_fake_graph")),
        "upload_fake_graph must be flagged: {findings:?}"
    );
}

#[test]
fn mutation_catches_parse_ambiguity_fails_closed() {
    let code = r#"
mod fake_ambiguous {
    pub struct AmbiguousGraph;
}

pub fn upload_ambiguous_graph(
    edges: &[u32],
) -> Result<fake_ambiguous::AmbiguousGraph, String> {
    let mut sum = 0u32;
    for &e in edges {
        sum = sum.wrapping_add(e ^ 0x3333);
    }
    let _ = sum;
    Ok(fake_ambiguous::AmbiguousGraph)
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/ambiguous.rs", code)]);
    assert!(
        !findings.is_empty(),
        "parse ambiguity or unresolvable type path must fail closed and convict host math"
    );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("upload_ambiguous_graph")),
        "upload_ambiguous_graph must be flagged: {findings:?}"
    );
}

#[test]
fn mutation_permits_genuine_resident_staging_separate_apis_unique_producer() {
    let staging_code = resident_staging_source(
        "ResidentDemoGraph",
        "pub(crate) ",
        &[("upload_resident_demo_graph", "0x5A5A_5A5A")],
    );
    let dispatch_code = format!(
        "{CANONICAL_DISPATCH_IMPORTS}use crate::staging::ResidentDemoGraph;
{dispatch}",
        dispatch = canonical_dispatch_fn(
            "pub ",
            "execute_demo_traversal",
            "ResidentDemoGraph",
            "",
            ""
        )
    );
    let findings = analyze_files(&[
        ("vyre-libs/src/staging.rs", staging_code.as_str()),
        ("vyre-libs/src/dispatch.rs", dispatch_code.as_str()),
    ]);
    assert!(
        findings.is_empty(),
        "genuine staging with separate upload and dispatch APIs and unique producer must not be convicted: {findings:?}"
    );
}

#[test]
fn mutation_catches_staging_type_with_pub_fields_without_call_path() {
    let staging_code = resident_staging_source(
        "ResidentDemoGraphWithPubFields",
        "pub ",
        &[("upload_resident_demo_graph", "0x5A5A_5A5A")],
    );
    let dispatch_code = format!(
        "{CANONICAL_DISPATCH_IMPORTS}use crate::staging::ResidentDemoGraphWithPubFields;
{dispatch}",
        dispatch = canonical_dispatch_fn(
            "pub ",
            "execute_demo_traversal",
            "ResidentDemoGraphWithPubFields",
            "",
            ""
        )
    );
    let findings = analyze_files(&[
        ("vyre-libs/src/staging.rs", staging_code.as_str()),
        ("vyre-libs/src/dispatch.rs", dispatch_code.as_str()),
    ]);
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("upload_resident_demo_graph")),
        "a carrier a caller can build field by field does not make its producer reachable: {findings:?}"
    );
}

#[test]
fn mutation_catches_staging_with_ignored_metadata_parameter_rejected() {
    let staging_code = format!(
        "{staging}
pub struct UnusedConfig {{
    pub(crate) threshold: u32,
}}

pub fn upload_unused_config_with_math(
    edges: &[u32],
) -> Result<UnusedConfig, SemanticExecutionError> {{
    let mut sum = 0u32;
    for &e in edges {{
        sum = sum.wrapping_add(e ^ 0x7777);
    }}
    Ok(UnusedConfig {{ threshold: sum }})
}}
",
        staging = resident_staging_source(
            "ResidentDemoGraph",
            "pub(crate) ",
            &[("upload_resident_demo_graph", "0x5A5A_5A5A")]
        )
    );
    let dispatch_code = format!(
        "{CANONICAL_DISPATCH_IMPORTS}use crate::staging::{{ResidentDemoGraph, UnusedConfig}};
{dispatch}",
        dispatch = canonical_dispatch_fn(
            "pub ",
            "execute_demo_traversal_with_config",
            "ResidentDemoGraph",
            "\n    config: &UnusedConfig,",
            "\n    let _ = config;"
        )
    );
    let findings = analyze_files(&[
        ("vyre-libs/src/staging.rs", staging_code.as_str()),
        ("vyre-libs/src/dispatch.rs", dispatch_code.as_str()),
    ]);
    assert!(
        !findings.is_empty(),
        "unused config parameter with host-math producer must be convicted"
    );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("upload_unused_config_with_math")),
        "upload_unused_config_with_math must be flagged: {findings:?}"
    );
    assert!(
        !findings
            .iter()
            .any(|f| f.message.contains("upload_resident_demo_graph")),
        "the producer whose bytes reach the seam must NOT be flagged: {findings:?}"
    );
}

#[test]
fn mutation_permits_genuine_resident_staging_transitive_helper_dispatch() {
    let staging_code = resident_staging_source(
        "ResidentDemoGraph",
        "pub(crate) ",
        &[("upload_resident_demo_graph", "0x5A5A_5A5A")],
    );
    let dispatch_code = format!(
        "{CANONICAL_DISPATCH_IMPORTS}use crate::staging::ResidentDemoGraph;
{helper}
pub fn execute_demo_traversal(
    dispatcher: &impl SemanticExecutor,
    logical: &LogicalProgramGraph<'_>,
    policy: &SemanticExecutionPolicy,
    graph: &ResidentDemoGraph,
) -> Result<(), SemanticExecutionError> {{
    helper_dispatch(dispatcher, logical, policy, graph)
}}
",
        helper = canonical_dispatch_fn("", "helper_dispatch", "ResidentDemoGraph", "", "")
    );
    let findings = analyze_files(&[
        ("vyre-libs/src/staging.rs", staging_code.as_str()),
        ("vyre-libs/src/dispatch.rs", dispatch_code.as_str()),
    ]);
    assert!(
        findings.is_empty(),
        "genuine staging with transitive helper dispatch flow must not be convicted: {findings:?}"
    );
}

#[test]
fn mutation_catches_pre_dispatch_host_math_helper_in_gpu_dispatch_fn() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn pre_calc_weights(input: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for &x in input {
        out.push(x * 2.5 + 1.0);
    }
    out
}

pub fn execute_with_pre_calc(
    dispatcher: &impl SemanticExecutor,
    input: &[f32],
) -> Result<(), SemanticExecutionError> {
    let weights = pre_calc_weights(input);
    let prog = Program::default();
    dispatcher.execute(&prog, &[&weights], &mut [])
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/pre_calc.rs", code)]);
    assert!(
            !findings.is_empty(),
            "GPU dispatch function invoking pre-dispatch host math helper must be convicted: {findings:?}"
        );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("pre_calc_weights")
                || f.message.contains("execute_with_pre_calc")),
        "expected pre-calc helper conviction, got: {findings:?}"
    );
}

#[test]
fn mutation_recognizes_dispatcher_in_wrapper_struct_and_catches_post_dispatch_math() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub struct DispatchContext<'a, D: SemanticExecutor> {
    pub dispatcher: &'a D,
}

impl<'a, D: SemanticExecutor> DispatchContext<'a, D> {
    pub fn run_pipeline(&self, prog: &Program, out: &mut [u8]) -> Result<u32, SemanticExecutionError> {
        self.dispatcher.execute(prog, &[], out)?;
        let mut sum = 0u32;
        for &b in out.iter() {
            sum = sum.wrapping_add(b as u32);
        }
        Ok(sum)
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/wrapper.rs", code)]);
    assert!(
            !findings.is_empty(),
            "dispatcher in wrapper struct with post-dispatch host reduction must be convicted: {findings:?}"
        );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("run_pipeline") || f.message.contains("wrapping_add")),
        "expected post-dispatch reduction conviction in wrapper struct method, got: {findings:?}"
    );
}

#[test]
fn mutation_catches_struct_literal_operation_registration_dynamic_expected_output() {
    let code = r#"
use vyre_foundation::operation::OperationRegistration;

pub fn dynamic_oracle_fixture(input: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &x in input {
        out.extend_from_slice(&x.wrapping_mul(3).to_le_bytes());
    }
    out
}

pub fn register_op() -> OperationRegistration {
    OperationRegistration {
        id: 42,
        expected_output: vec![dynamic_oracle_fixture(&[1, 2, 3])],
    }
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/op_struct.rs", code)]);
    assert!(
            !findings.is_empty(),
            "OperationRegistration struct literal with dynamic expected_output must be convicted: {findings:?}"
        );
    assert!(
        findings
            .iter()
            .any(|f| f.message.contains("dynamic_oracle_fixture")
                || f.message.contains("expected_output")),
        "expected struct literal dynamic expected_output conviction, got: {findings:?}"
    );
}

#[test]
fn mutation_permits_post_dispatch_non_data_diagnostic_telemetry_methods() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub struct DispatchMetrics {
    pub total_ops: u32,
}

pub fn execute_with_metrics(
    dispatcher: &impl SemanticExecutor,
    prog: &Program,
    metrics: &mut DispatchMetrics,
) -> Result<(), SemanticExecutionError> {
    dispatcher.execute(prog, &[], &mut [])?;
    metrics.total_ops = metrics.total_ops.wrapping_add(1);
    Ok(())
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/telemetry_dispatch.rs", code)]);
    assert!(
        findings.is_empty(),
        "non-data diagnostic telemetry in post-dispatch phase must be permitted: {findings:?}"
    );
}

#[test]
fn mutation_permits_inter_dispatch_staging_buffer_operations() {
    let code = r#"
use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutor};

pub fn execute_two_stage_pipeline(
    dispatcher: &impl SemanticExecutor,
    stage1_prog: &Program,
    stage2_prog: &Program,
    intermediate_scratch: &mut Vec<u8>,
) -> Result<(), SemanticExecutionError> {
    dispatcher.execute(stage1_prog, &[], intermediate_scratch.as_mut_slice())?;
    intermediate_scratch.clear();
    intermediate_scratch.resize(64, 0);
    dispatcher.execute(stage2_prog, &[], intermediate_scratch.as_mut_slice())
}
"#;
    let findings = analyze_files(&[("vyre-libs/src/multi_stage.rs", code)]);
    assert!(
            findings.is_empty(),
            "intermediate buffer operations between sequential dispatches must be permitted: {findings:?}"
        );
}

#[test]
fn test_workspace_findings() {
    let root = structure_gate::workspace_root();
    let tree = Tree::open(&root).unwrap();
    let sources = tree.rust(TARGET_ROOTS).unwrap();
    let test_scoped = test_module_files(&tree, &sources).unwrap();
    let findings = analyze_sources(&tree, &sources, &test_scoped).unwrap();
    assert_eq!(
        findings.len(),
        0,
        "host oracle elimination gate must report 0 findings across workspace: {findings:#?}"
    );
}

/// A `#[cfg(test)] #[path = "..."] mod tests;` declaration puts the named file
/// wholly in test scope, so the gate must not read it as production code. Three
/// `vyre-libs` modules use that form; before the resolver honoured `#[path]` it
/// searched only for `tests.rs`, missed all three, and convicted their host
/// reference helpers. The declaration set is enumerated from the workspace at
/// run time, so a new one is covered without editing this test.
#[test]
fn a_cfg_test_module_named_by_a_path_attribute_is_test_scoped() {
    let root = structure_gate::workspace_root();
    let tree = Tree::open(&root).unwrap();
    let sources = tree.rust(TARGET_ROOTS).unwrap();
    let test_scoped = test_module_files(&tree, &sources).unwrap();

    let mut declared: Vec<PathBuf> = Vec::new();
    for path in &sources {
        let text = tree.read(path).unwrap();
        let file_ast = syn::parse_file(&text).unwrap();
        for item in &file_ast.items {
            let syn::Item::Mod(item_mod) = item else {
                continue;
            };
            if item_mod.content.is_some() {
                continue;
            }
            let is_test_gated = item_mod.attrs.iter().any(attribute_is_test_only);
            if !is_test_gated {
                continue;
            }
            for attr in &item_mod.attrs {
                if !attr.path().is_ident("path") {
                    continue;
                }
                let syn::Meta::NameValue(name_value) = &attr.meta else {
                    continue;
                };
                let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(literal),
                    ..
                }) = &name_value.value
                else {
                    continue;
                };
                let named = path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(""))
                    .join(literal.value());
                if sources.contains(&named) {
                    declared.push(named);
                }
            }
        }
    }

    assert!(
        declared.len() >= 3,
        "the workspace declares fewer `#[cfg(test)] #[path = ...]` modules than the three in vyre-libs; \
         if they were removed, delete this test, otherwise the enumeration is broken: {declared:?}"
    );
    let missed: Vec<&PathBuf> = declared
        .iter()
        .filter(|path| !test_scoped.contains(*path))
        .collect();
    assert!(
        missed.is_empty(),
        "every file named by a `#[cfg(test)] #[path = ...]` module is test-scoped: {missed:?}"
    );
}

#[test]
fn mutation_permits_staging_that_binds_its_own_payload_into_a_request() {
    let staging_code = format!(
        "{staging}
pub fn run_demo_traversal(
    dispatcher: &impl SemanticExecutor,
    logical: &LogicalProgramGraph<'_>,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    edges: &[u32],
) -> Result<(), SemanticExecutionError> {{
    let mut packed = Vec::new();
    let request = stage_demo_request(logical, policy, &mut packed, node_count, edges)?;
    dispatcher.execute(&request)?;
    Ok(())
}}
",
        staging = self_binding_staging_source(true)
    );
    let findings = analyze_files(&[("vyre-libs/src/self_binding.rs", staging_code.as_str())]);
    assert!(
        findings.is_empty(),
        "a producer that binds its own payload into a canonical request is proven staging: {findings:?}"
    );
}

#[test]
fn mutation_catches_staging_that_leaves_binding_to_its_caller() {
    let staging_code = format!(
        "{staging}
pub fn run_demo_traversal(
    dispatcher: &impl SemanticExecutor,
    logical: &LogicalProgramGraph<'_>,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    edges: &[u32],
) -> Result<(), SemanticExecutionError> {{
    let packed = stage_demo_request(node_count, edges)?;
    let mut inputs = BTreeMap::new();
    inputs.insert(GraphValueId(0), packed.as_slice());
    let request = SemanticExecutionRequest::new(
{CANONICAL_REQUEST_ARGUMENTS}
    )?;
    dispatcher.execute(&request)?;
    Ok(())
}}
",
        staging = self_binding_staging_source(false)
    );
    let findings = analyze_files(&[("vyre-libs/src/caller_binding.rs", staging_code.as_str())]);
    assert!(
        findings
            .iter()
            .any(|finding| finding.message.contains("stage_demo_request")),
        "host bytes computed outside the seam are convicted even when the caller binds them: {findings:?}"
    );
}
