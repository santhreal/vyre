use super::all_entries_vec::*;
use super::harness::f32_to_ordered;
use proptest::prelude::*;
use std::sync::LazyLock;
use vyre::ir::DataType;
use vyre::ir::{BufferAccess, BufferDecl, Program};
use vyre_driver::DispatchConfig;
use vyre_driver::VyreBackend;
use vyre_foundation::fp_parity;
use vyre_reference::value::Value;

fn entry_cases(entry: &UnifiedEntry) -> Vec<Vec<Vec<u8>>> {
    let cases = entry.test_inputs.map(|f| f()).unwrap_or_default();
    assert!(
        !cases.is_empty(),
        "Fix: {} has no pairwise test inputs; every op registry entry must publish runnable witnesses.",
        entry.id
    );
    cases
}

fn compatible_pairs() -> &'static [(usize, usize)] {
    static PAIRS: LazyLock<Vec<(usize, usize)>> = LazyLock::new(|| {
        let entries = all_entries_vec();
        let cases = entries.iter().map(entry_cases).collect::<Vec<_>>();

        let mut pairs = Vec::new();
        for a_idx in 0..entries.len() {
            for b_idx in 0..entries.len() {
                let a = &entries[a_idx];
                let b = &entries[b_idx];
                let composition = match try_compose(a, b) {
                    Ok(composition) => composition,
                    Err(_) => continue,
                };
                validate_for_backend(&composition.program).unwrap_or_else(|error| {
                    panic!(
                        "Fix: {} -> {} composed successfully but failed validation: {error}",
                        a.id, b.id
                    )
                });
                if let Some(reason) = missing_capability_reason(&composition.program) {
                    panic!(
                        "Fix: {} -> {} requires an unsupported backend capability: {reason}",
                        a.id, b.id
                    );
                }
                let all_witnesses_run = cases[a_idx].iter().all(|a_case| {
                    cases[b_idx].iter().all(|b_case| {
                        let inputs = build_fused_inputs(&composition, a_case, b_case);
                        try_run_reference(a.id, b.id, &composition.program, &inputs).is_ok()
                    })
                });
                if all_witnesses_run {
                    pairs.push((a_idx, b_idx));
                }
            }
        }
        assert!(
            !pairs.is_empty(),
            "Fix: pairwise composition found zero compatible op pairs; repair op metadata or composition wiring."
        );
        pairs
    });
    PAIRS.as_slice()
}

fn compatible_pair_count() -> usize {
    compatible_pairs().len()
}

fn compatible_pair_by_index(idx: usize) -> (&'static UnifiedEntry, &'static UnifiedEntry) {
    let pairs = compatible_pairs();
    let (a_idx, b_idx) = pairs[idx % pairs.len()];
    (entry_by_index(a_idx), entry_by_index(b_idx))
}

// ------------------------------------------------------------------
// Input assembly
// ------------------------------------------------------------------

/// Pair each of an op's witness buffers with the name it was declared under.
///
/// Registry witnesses support the current logical ABI and the legacy ABI that
/// includes a placeholder for each backend-allocated output. Selecting the
/// declaration set from the witness length preserves later inputs when a legacy
/// output appears before them.
fn witness_by_name(prog: &Program, case: &[Vec<u8>]) -> Vec<(String, Vec<u8>)> {
    let legacy = witness_uses_legacy_abi(prog, case.len()).unwrap_or_else(|| {
        let logical_count = prog
            .buffers()
            .iter()
            .filter(|buffer| needs_input(buffer))
            .count();
        let legacy_count = prog
            .buffers()
            .iter()
            .filter(|buffer| buffer.access() != BufferAccess::Workgroup)
            .count();
        panic!(
            "Fix: witness supplies {} buffers, but the program accepts {logical_count} logical or {legacy_count} legacy buffers.",
            case.len()
        )
    });
    prog.buffers()
        .iter()
        .filter(|buffer| {
            buffer.access() != BufferAccess::Workgroup && (legacy || needs_input(buffer))
        })
        .map(|buffer| buffer.name().to_string())
        .zip(case.iter().cloned())
        .collect()
}

fn witness_abi_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("output", 1, DataType::U32).with_count(2),
            BufferDecl::read_write("state", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::new(),
    )
}

/// Legacy output placeholders must not shift a later caller-supplied state
/// buffer onto the output declaration.
#[test]
fn legacy_witness_output_placeholder_preserves_trailing_input() {
    let program = witness_abi_program();
    let witness = vec![vec![1; 4], vec![2; 8], vec![3; 4]];

    assert_eq!(
        witness_by_name(&program, &witness),
        vec![
            ("input".to_string(), vec![1; 4]),
            ("output".to_string(), vec![2; 8]),
            ("state".to_string(), vec![3; 4]),
        ]
    );
}

/// Logical witnesses omit backend-allocated outputs while retaining declaration
/// order for every caller-supplied buffer.
#[test]
fn logical_witness_skips_backend_allocated_output() {
    let program = witness_abi_program();
    let witness = vec![vec![1; 4], vec![3; 4]];

    assert_eq!(
        witness_by_name(&program, &witness),
        vec![
            ("input".to_string(), vec![1; 4]),
            ("state".to_string(), vec![3; 4]),
        ]
    );
}

/// A witness count matching neither supported ABI must be rejected instead of
/// silently truncating the declaration-to-byte association.
#[test]
fn malformed_witness_count_has_no_supported_abi() {
    let program = witness_abi_program();

    assert_eq!(witness_uses_legacy_abi(&program, 1), None);
    assert_eq!(witness_uses_legacy_abi(&program, 4), None);
}

/// A launch-dependent producer followed by a whole-buffer consumer must retain
/// every intermediate lane across the GridSync dispatch split.
#[test]
fn gqa_to_fft_grid_sync_preserves_intermediate_lanes() {
    let entries = all_entries_vec();
    let a = entries
        .iter()
        .find(|entry| entry.id == "vyre-libs::nn::gqa_attention")
        .expect("GQA registry entry");
    let b = entries
        .iter()
        .find(|entry| entry.id == "vyre-libs::math::fft::fft4_complex")
        .expect("FFT4 registry entry");
    let composition = try_compose(a, b).expect("GQA output must compose with FFT4 input");
    let inputs = build_fused_inputs(&composition, &entry_cases(a)[0], &entry_cases(b)[0]);
    let expected = run_reference(a.id, b.id, &composition.program, &inputs);
    let optimized = vyre_foundation::optimizer::optimize(composition.program.clone())
        .expect("registered optimizer must converge");
    let optimized_reference = run_reference(a.id, b.id, &optimized, &inputs);
    let elements = output_elements(&composition.program);
    let tolerance = fp_parity::effective_tolerance(a.id, &composition.program)
        .max(fp_parity::effective_tolerance(b.id, &composition.program));

    assert_outputs_equal(
        a.id,
        "pre-lowering optimizer",
        tolerance,
        &elements,
        &expected,
        &optimized_reference,
    );
    let gpu = run_gpu(&composition.program, &inputs).expect("GridSync GPU dispatch");
    assert_outputs_equal(
        a.id,
        "WGPU GridSync split",
        tolerance,
        &elements,
        &expected,
        &gpu,
    );
}

/// A piped output starts from the upstream buffer's initializer, never from the
/// downstream witness that the output replaces.
#[test]
fn substring_self_composition_does_not_seed_output_from_downstream_input() {
    let entries = all_entries_vec();
    let entry = entries
        .iter()
        .find(|entry| entry.id == "vyre-libs::pattern::substring_search")
        .expect("substring registry entry");
    let composition =
        try_compose(entry, entry).expect("substring output must compose with substring input");
    let case = &entry_cases(entry)[0];
    let inputs = build_fused_inputs(&composition, case, case);
    let expected_intermediate = vyre_primitives::wire::pack_u32_slice(&[1, 0, 0, 1, 0, 0, 0, 0]);
    let expected_final = vec![0u8; 32];

    let reference = run_reference(entry.id, entry.id, &composition.program, &inputs);
    assert_eq!(
        reference,
        vec![expected_intermediate.clone(), expected_final.clone()]
    );
    let gpu = run_gpu(&composition.program, &inputs).expect("substring GridSync GPU dispatch");
    assert_eq!(gpu, vec![expected_intermediate, expected_final]);
}

/// A legacy placeholder for an upstream backend-allocated output must not
/// become the real initializer after that output is demoted into the pipe.
#[test]
fn attention_to_cross_entropy_discards_legacy_output_placeholder() {
    let entries = all_entries_vec();
    let attention = entries
        .iter()
        .find(|entry| entry.id == "vyre-libs::nn::attention")
        .expect("attention registry entry");
    let cross_entropy = entries
        .iter()
        .find(|entry| entry.id == "vyre-libs::nn::cross_entropy")
        .expect("cross-entropy registry entry");
    let composition = try_compose(attention, cross_entropy)
        .expect("attention logical output must compose with cross-entropy logits");
    let inputs = build_fused_inputs(
        &composition,
        &entry_cases(attention)[0],
        &entry_cases(cross_entropy)[0],
    );
    let reference = run_reference(
        attention.id,
        cross_entropy.id,
        &composition.program,
        &inputs,
    );
    assert_eq!(reference[0].len(), 8 * core::mem::size_of::<f32>());
    assert_eq!(reference[1].len(), 2 * core::mem::size_of::<f32>());
    let wired = composition
        .program
        .buffer(&composition.wired_name)
        .expect("wired attention output declaration");
    assert_eq!(wired.count(), 8);
    assert_eq!(wired.output_byte_range(), None);
    let optimized = vyre_foundation::optimizer::optimize(composition.program.clone())
        .expect("registered optimizer must converge");
    let optimized_wired = optimized
        .buffer(&composition.wired_name)
        .expect("optimized wired attention output declaration");
    assert_eq!(optimized_wired.count(), 8);
    assert_eq!(optimized_wired.output_byte_range(), None);

    let gpu = run_gpu(&composition.program, &inputs).expect("attention GridSync GPU dispatch");
    assert_eq!(gpu[0].len(), reference[0].len());
    assert_eq!(gpu[1].len(), reference[1].len());
}

/// Self-composing a multi-segment dataflow operation must supply every
/// caller-owned buffer to each GridSync segment.
#[test]
fn aliases_self_composition_retains_all_segment_inputs() {
    let entries = all_entries_vec();
    let entry = entries
        .iter()
        .find(|entry| entry.id == "vyre-libs::security::aliases_dataflow")
        .expect("aliases-dataflow registry entry");
    let composition =
        try_compose(entry, entry).expect("aliases output must compose with aliases input");
    let case = &entry_cases(entry)[0];
    let inputs = build_fused_inputs(&composition, case, case);
    for (buffer, bytes) in composition
        .program
        .buffers()
        .iter()
        .filter(|buffer| needs_input(buffer))
        .zip(&inputs)
    {
        assert!(
            !bytes.is_empty(),
            "fused input `{}` must have caller bytes",
            buffer.name()
        );
    }

    let reference = run_reference(entry.id, entry.id, &composition.program, &inputs);
    let gpu = run_gpu(&composition.program, &inputs).expect("aliases GridSync GPU dispatch");
    assert_eq!(gpu, reference);
}

/// Dispatch inference must use the largest writable buffer so a downstream
/// multi-workgroup consumer executes every output row.
#[test]
fn fft_to_cross_entropy_uses_largest_output_for_dispatch_grid() {
    let entries = all_entries_vec();
    let fft = entries
        .iter()
        .find(|entry| entry.id == "vyre-libs::math::fft::scale_conjugate_inverse")
        .expect("inverse-scale registry entry");
    let cross_entropy = entries
        .iter()
        .find(|entry| entry.id == "vyre-libs::nn::cross_entropy")
        .expect("cross-entropy registry entry");
    let composition =
        try_compose(fft, cross_entropy).expect("inverse-scale output must compose with logits");
    let inputs = build_fused_inputs(
        &composition,
        &entry_cases(fft)[0],
        &entry_cases(cross_entropy)[0],
    );
    let reference = run_reference(fft.id, cross_entropy.id, &composition.program, &inputs);
    let gpu = run_gpu(&composition.program, &inputs).expect("multi-output WGPU dispatch");
    let tolerance = fp_parity::effective_tolerance(fft.id, &composition.program).max(
        fp_parity::effective_tolerance(cross_entropy.id, &composition.program),
    );

    assert_outputs_equal(
        fft.id,
        cross_entropy.id,
        tolerance,
        &output_elements(&composition.program),
        &reference,
        &gpu,
    );
}

/// A producer with several mutable results and no declared output must be
/// rejected instead of wiring an arbitrary scratch buffer downstream.
#[test]
fn ambiguous_multi_output_producer_is_not_pairwise_composable() {
    let entries = all_entries_vec();
    let quest = entries
        .iter()
        .find(|entry| entry.id == "vyre-libs::nn::attention::quest_paging")
        .expect("quest-paging registry entry");
    let strassen = entries
        .iter()
        .find(|entry| entry.id == "vyre-libs::math::linalg::matmul_strassen_2x2")
        .expect("Strassen registry entry");

    let error = try_compose(quest, strassen)
        .err()
        .expect("ambiguous quest output must be rejected");
    assert!(
        error.contains("writable buffers and no explicit output"),
        "unexpected rejection: {error}"
    );
}

/// A zero-filled placeholder sized to a declaration.
///
/// The piped buffer is op_a's result, so no witness supplies it, but after
/// demotion the fused program still expects a caller-provided value for it.
/// Zeros are the right seed: op_a writes it before op_b reads it.
fn zero_placeholder(buf: &BufferDecl) -> Vec<u8> {
    let len = buf
        .static_byte_len()
        .unwrap_or_else(|error| {
            panic!(
                "Fix: buffer `{}` has no static byte length in a fused pairwise program: {error}",
                buf.name()
            )
        })
        .unwrap_or(0);
    vec![0u8; len]
}

/// Build the input vector for the fused program.
///
/// Assembly is keyed by buffer name against the fused program's own
/// declaration list. Fusion dedups shared buffers, keeps them in first-arm
/// position, and the wiring pass renames several of op_b's, so positional
/// concatenation of the two witness vectors does not describe the result. The
/// symptom was a missing value for whichever buffer the offset landed past.
fn build_fused_inputs(comp: &Composition, a_case: &[Vec<u8>], b_case: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut by_name: std::collections::HashMap<String, Vec<u8>> =
        witness_by_name(&comp.prog_a, a_case).into_iter().collect();
    if comp
        .prog_a
        .buffer(&comp.wired_name)
        .is_some_and(BufferDecl::is_backend_allocated_output)
    {
        by_name.remove(&comp.wired_name);
    }

    let rename = |name: &str| -> String {
        comp.b_renames
            .iter()
            .find(|(from, _)| from == name)
            .map_or_else(|| name.to_string(), |(_, to)| to.clone())
    };

    // op_a wins on any name they share. In particular, op_b's witness for the
    // wired input is not the initial value of op_a's output: the upstream stage
    // owns that storage and an absent upstream initializer must remain zero.
    for (name, bytes) in witness_by_name(&comp.prog_b, b_case) {
        let fused_name = rename(&name);
        if fused_name == comp.wired_name {
            continue;
        }
        by_name.entry(fused_name).or_insert(bytes);
    }

    comp.program
        .buffers()
        .iter()
        .filter(|buf| needs_input(buf))
        .map(|buf| {
            by_name
                .get(buf.name())
                .cloned()
                .unwrap_or_else(|| zero_placeholder(buf))
        })
        .collect()
}

// ------------------------------------------------------------------
// Execution wrappers
// ------------------------------------------------------------------

fn try_run_reference(
    op_a: &str,
    op_b: &str,
    program: &Program,
    inputs: &[Vec<u8>],
) -> Result<Vec<Vec<u8>>, String> {
    let values: Vec<Value> = inputs.iter().cloned().map(Value::from).collect();
    vyre_reference::reference_eval(program, &values)
        .map(|outputs| outputs.into_iter().map(|value| value.to_bytes()).collect())
        .map_err(|error| format!("Fix: {op_a} -> {op_b} reference_eval failed: {error}"))
}

fn run_reference(op_a: &str, op_b: &str, program: &Program, inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    try_run_reference(op_a, op_b, program, inputs).unwrap_or_else(|error| panic!("{error}"))
}

fn run_gpu(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
    let backend = gpu();
    let lowered = vyre_foundation::optimizer::optimize(program.clone())
        .expect("registered optimizer must converge");
    backend
        .dispatch(&lowered, inputs, &DispatchConfig::default())
        .map_err(|e| format!("GPU dispatch error: {e}"))
}

fn f32_matches_with_tolerance(cpu_bits: u32, gpu_bits: u32, tolerance: u32) -> bool {
    let cpu = f32::from_bits(cpu_bits);
    let gpu = f32::from_bits(gpu_bits);
    if cpu.is_nan() || gpu.is_nan() {
        return cpu.is_nan() && gpu.is_nan();
    }
    if f32_to_ordered(cpu_bits).abs_diff(f32_to_ordered(gpu_bits)) <= tolerance {
        return true;
    }
    if !cpu.is_finite() || !gpu.is_finite() {
        return false;
    }

    let scale = cpu.abs().max(gpu.abs()).max(1.0);
    (cpu - gpu).abs() <= tolerance as f32 * f32::EPSILON * scale
}

/// CPU and GPU backends may choose different quiet-NaN payloads while
/// preserving the same IEEE-754 result class.
#[test]
fn float_parity_accepts_distinct_nan_payloads() {
    assert!(f32_matches_with_tolerance(0x7fc0_0000, 0x7fff_ffff, 0));
    assert!(f32_matches_with_tolerance(0xffc0_0000, 0x7fc0_0001, 0));
}

/// A NaN result cannot satisfy parity with a finite value merely because the
/// floating-point comparison supports backend-specific NaN payloads.
#[test]
fn float_parity_rejects_nan_against_finite_value() {
    assert!(!f32_matches_with_tolerance(
        0x7fc0_0000,
        1.0_f32.to_bits(),
        128
    ));
}

/// Finite results continue to honor the exact configured ULP boundary.
#[test]
fn float_parity_preserves_finite_ulp_tolerance() {
    let one = 1.0_f32.to_bits();
    assert!(f32_matches_with_tolerance(one, one + 2, 2));
    assert!(!f32_matches_with_tolerance(one, one + 3, 2));
}

/// Cancellation may turn a few input ULPs into a small residual versus exact
/// zero, so composed floating pipelines also need a scale-aware error bound.
#[test]
fn float_parity_accepts_near_zero_cancellation_residual() {
    assert!(f32_matches_with_tolerance(
        (-1.907_348_6e-6_f32).to_bits(),
        0.0_f32.to_bits(),
        128,
    ));
}

/// The scale-aware bound must still reject a materially wrong result near zero.
#[test]
fn float_parity_rejects_material_near_zero_error() {
    assert!(!f32_matches_with_tolerance(
        1.0e-3_f32.to_bits(),
        0.0_f32.to_bits(),
        128,
    ));
}

/// The element types of the buffers `reference_eval` hands back, in the order
/// it hands them back.
///
/// The tolerance below is a floating-point ULP distance and means nothing on an
/// integer buffer: two u32 counts three apart are not "within 4 ULP", they are
/// two different counts. Pairing each returned buffer with its declared element
/// type is what lets the comparison stay exact wherever exactness is the
/// contract, which for the catalog is most buffers.
fn output_elements(program: &Program) -> Vec<DataType> {
    program
        .buffers()
        .iter()
        .filter(|decl| vyre_reference::is_reference_output(decl))
        .map(BufferDecl::element)
        .collect()
}

/// Is rounding a real degree of freedom for this element type?
///
/// Only floating-point elements may differ between the reference and the
/// device. Everything else is exact, and a tolerance applied to it would accept
/// a wrong integer.
fn tolerates_rounding(element: &DataType) -> bool {
    matches!(
        element,
        DataType::F32 | DataType::F16 | DataType::BF16 | DataType::F64
    )
}

fn assert_outputs_equal(
    op_a: &str,
    op_b: &str,
    tolerance: u32,
    elements: &[DataType],
    cpu: &[Vec<u8>],
    gpu: &[Vec<u8>],
) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "Fix: {op_a} -> {op_b}: CPU produced {} buffers, GPU produced {}",
        cpu.len(),
        gpu.len()
    );
    assert_eq!(
        cpu.len(),
        elements.len(),
        "Fix: {op_a} -> {op_b}: {} returned buffers but {} output declarations. The element types must line up with the buffers or the comparison cannot know which are floating point.",
        cpu.len(),
        elements.len()
    );

    for (buf_idx, (c_buf, g_buf)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(
            c_buf.len(),
            g_buf.len(),
            "Fix: {op_a} -> {op_b}: buffer #{buf_idx} length diverged. CPU={} GPU={}",
            c_buf.len(),
            g_buf.len()
        );

        // A tolerance applies only where rounding is a real degree of freedom.
        let tolerance = if tolerates_rounding(&elements[buf_idx]) {
            tolerance
        } else {
            0
        };

        if tolerance == 0 {
            for (byte_offset, (cb, gb)) in c_buf.iter().zip(g_buf.iter()).enumerate() {
                assert_eq!(
                    cb, gb,
                    "Fix: {op_a} -> {op_b}: buffer #{buf_idx} first divergent byte at offset {byte_offset}. CPU={:02x?} GPU={:02x?}",
                    c_buf, g_buf
                );
            }
        } else {
            assert_eq!(
                c_buf.len() % 4,
                0,
                "Fix: {op_a} -> {op_b}: tolerance-based compare requires f32-aligned bytes"
            );
            for (lane, (c_word, g_word)) in
                c_buf.chunks_exact(4).zip(g_buf.chunks_exact(4)).enumerate()
            {
                let c_bits = u32::from_le_bytes(c_word.try_into().unwrap());
                let g_bits = u32::from_le_bytes(g_word.try_into().unwrap());
                assert!(
                    f32_matches_with_tolerance(c_bits, g_bits, tolerance),
                    "Fix: {op_a} -> {op_b}: buffer #{buf_idx} lane {lane} diverged above the {tolerance}-ULP or scale-aware floating tolerance. CPU bits=0x{c_bits:08x} GPU bits=0x{g_bits:08x}"
                );
            }
        }
    }
}

// ------------------------------------------------------------------
// Proptest configuration
// ------------------------------------------------------------------

fn proptest_config() -> ProptestConfig {
    let cases = if std::env::var("CI_EXHAUSTIVE").is_ok() {
        50_000
    } else {
        5_000
    };
    ProptestConfig {
        cases,
        ..ProptestConfig::default()
    }
}

// ------------------------------------------------------------------
// Proving test  -  composition parity
// ------------------------------------------------------------------

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn pairwise_composition_parity(
        pair_idx in 0..compatible_pair_count(),
        case_idx in any::<usize>(),
    ) {
        let (a, b) = compatible_pair_by_index(pair_idx);

        let a_cases = entry_cases(a);
        let b_cases = entry_cases(b);

        let a_case = &a_cases[case_idx % a_cases.len()];
        let b_case = &b_cases[case_idx % b_cases.len()];

        // Build and validate compatibility.
        let comp =
            try_compose(a, b).expect("Fix: compatible_pair_by_index returned an incompatible pair");
        let composed = &comp.program;

        // Validate the fused IR.
        if let Err(e) = validate_for_backend(composed) {
            panic!(
                "Fix: {} -> {} composed program validation failed: {e}",
                a.id, b.id
            );
        }

        if let Some(reason) = missing_capability_reason(composed) {
            panic!(
                "Fix: {} -> {} backend capability check failed after compatibility precomputation: {reason}",
                a.id, b.id
            );
        }

        // Assemble fused inputs.
        let fused_inputs = build_fused_inputs(&comp, a_case, b_case);

        // CPU reference oracle.
        let cpu = run_reference(a.id, b.id, composed, &fused_inputs);

        // GPU backend.
        let gpu = match run_gpu(composed, &fused_inputs) {
            Ok(out) => out,
            Err(reason) => {
                panic!(
                    "Fix: {} -> {} GPU dispatch failed in pairwise parity: {reason}",
                    a.id, b.id
                )
            }
        };

        let tolerance = fp_parity::effective_tolerance(a.id, composed)
            .max(fp_parity::effective_tolerance(b.id, composed));
        assert_outputs_equal(
            a.id,
            b.id,
            tolerance,
            &output_elements(composed),
            &cpu,
            &gpu,
        );
    }
}

// ------------------------------------------------------------------
// Adversarial test  -  never panic, never silent-wrong
// ------------------------------------------------------------------

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn pairwise_composition_adversarial(
        a_idx in 0..entry_count(),
        b_idx in 0..entry_count(),
        case_idx in any::<usize>(),
    ) {
        let a = entry_by_index(a_idx);
        let b = entry_by_index(b_idx);

        // try_compose must NEVER panic, regardless of compatibility.
        let composed_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            try_compose(a, b)
        }));

        assert!(
            composed_result.is_ok(),
            "Fix: {} -> {}: try_compose panicked  -  composition logic must reject gracefully",
            a.id, b.id
        );

        match composed_result.unwrap() {
            Ok(comp) => {
                let composed = &comp.program;
                // The pair was compatible.  Verify it does not produce silent
                // wrong output by running the reference vs GPU differential.

                let a_cases = a.test_inputs.map(|f| f()).unwrap_or_default();
                let b_cases = b.test_inputs.map(|f| f()).unwrap_or_default();
                if a_cases.is_empty() || b_cases.is_empty() {
                    panic!(
                        "Fix: {} -> {} compatible pair is missing test inputs.",
                        a.id, b.id
                    );
                }
                let a_case = &a_cases[case_idx % a_cases.len()];
                let b_case = &b_cases[case_idx % b_cases.len()];

                if validate_for_backend(composed).is_err() {
                    // Validation failure on a supposedly-compatible pair is a bug.
                    panic!(
                        "Fix: {} -> {}: composed program failed validation despite compatibility check",
                        a.id, b.id
                    );
                }

                // Tolerance is derived from the composed program so FMA
                // contraction and transcendental policy cannot drift by test lane.

                if let Some(reason) = missing_capability_reason(composed) {
                    panic!(
                        "Fix: {} -> {} backend capability check failed: {reason}",
                        a.id, b.id
                    );
                }

                let fused_inputs = build_fused_inputs(&comp, a_case, b_case);

                match try_run_reference(a.id, b.id, composed, &fused_inputs) {
                    Ok(cpu) => {
                        let gpu = run_gpu(composed, &fused_inputs).unwrap_or_else(|reason| {
                            panic!(
                                "Fix: {} -> {} GPU dispatch failed in adversarial pairwise parity: {reason}",
                                a.id, b.id
                            )
                        });
                        let tolerance = fp_parity::effective_tolerance(a.id, composed)
                            .max(fp_parity::effective_tolerance(b.id, composed));
                        assert_outputs_equal(
                            a.id,
                            b.id,
                            tolerance,
                            &output_elements(composed),
                            &cpu,
                            &gpu,
                        );
                    }
                    Err(reason) => {
                        assert!(
                            reason.contains("Fix:"),
                            "Fix: {} -> {}: reference rejection missing actionable hint: {}",
                            a.id,
                            b.id,
                            reason
                        );
                    }
                }
            }
            Err(reason) => {
                // Incompatible pair rejected cleanly  -  this is the expected
                // adversarial path.  The error must be actionable.
                assert!(
                    reason.contains("Fix:"),
                    "Fix: {} -> {}: rejection reason missing actionable hint: {}",
                    a.id, b.id, reason
                );
            }
        }
    }
}

// ------------------------------------------------------------------
// Named-pair contracts
// ------------------------------------------------------------------

fn entry_named(id: &str) -> &'static UnifiedEntry {
    let index = all_entries_vec()
        .iter()
        .position(|entry| entry.id == id)
        .unwrap_or_else(|| panic!("Fix: no op registry entry named `{id}`"));
    entry_by_index(index)
}

/// Piping a 64-wide elementwise op into a 4-wide scan must be refused.
///
/// This pair is why the workgroup-geometry check exists. `scan_prefix_sum` at
/// n=4 is built for a 4-invocation workgroup, keeps two workgroup buffers, and
/// synchronizes five times. Fused behind a 64-wide arm it would run under a
/// 64-invocation workgroup, where the 60 invocations with no work skip the
/// guarded body and never reach the barriers the working four wait on.
///
/// The contract is a clean refusal, not a lucky pass: a racy kernel that is
/// usually right is worse than one that never builds. See BACKLOG.md R73.
#[test]
fn a_narrow_scan_is_not_fused_behind_a_wide_elementwise_op() {
    let a = entry_named("vyre-libs::math::avg_floor");
    let b = entry_named("vyre-libs::math::scan_prefix_sum");
    let reason = match try_compose(a, b) {
        Ok(comp) => panic!(
            "Fix: fusing a 4-wide scan behind a 64-wide arm was accepted and produced a program with workgroup {:?}. That kernel is intermittently wrong.",
            comp.program.workgroup_size()
        ),
        Err(reason) => reason,
    };
    assert!(
        reason.contains("workgroup"),
        "Fix: the refusal must name the workgroup geometry as the cause: {reason}"
    );
    assert!(
        reason.contains("Fix:"),
        "Fix: every rejection reason carries an actionable hint: {reason}"
    );
}

/// A fixed-size producer cannot feed a runtime-sized consumer whose witness
/// proves a different byte extent.
#[test]
fn runtime_sized_input_witness_must_match_upstream_extent() {
    let a = entry_named("vyre-libs::math::avg_floor");
    let b = entry_named("vyre-libs::parsing::ast_shunting_yard");

    let reason = match try_compose(a, b) {
        Ok(_) => {
            panic!("a four-lane producer must not feed a runtime-sized 64 Ki-lane parser input")
        }
        Err(reason) => reason,
    };

    assert!(
        reason.contains("runtime-sized input byte mismatch"),
        "Fix: the refusal must identify the runtime witness extent mismatch: {reason}"
    );
}

/// An op cannot be piped into itself when its body is marked self-exclusive.
///
/// `core_delimiter_match` carries per-instance scratch, so two copies in one
/// kernel would stomp each other. The harness used to call this pair
/// compatible and then report the validator's refusal as a composition bug.
/// See BACKLOG.md R71.
#[test]
fn a_self_exclusive_parser_is_not_piped_into_itself() {
    let entry = entry_named("vyre-libs::parsing::core_delimiter_match");
    let reason = match try_compose(entry, entry) {
        Ok(_) => panic!(
            "Fix: two copies of a self-exclusive region were fused into one kernel; they share scratch storage."
        ),
        Err(reason) => reason,
    };
    assert!(
        reason.contains("self-exclusive") || reason.contains("non-composable"),
        "Fix: the refusal must name the self-exclusivity contract: {reason}"
    );
}
