//! Live CUDA witness for MLA decode shared-memory scaling against `head_dim`.
//!
//! `vyre_libs::nn::attention::mla::mla_decode` pins its workgroup at a fixed
//! 64 lanes (`WORKGROUP_LANES`) while all three of its workgroup scratch
//! buffers scale with the runtime `head_dim`:
//!
//! ```text
//! q_scratch  = 64 * head_dim  f32
//! score_tile = 64 * 64        f32
//! o_acc      = 64 * head_dim  f32
//! bytes      = (128 * head_dim + 4096) * 4
//! ```
//!
//! That puts a fixed lane count against an unbounded extent, so the shared
//! memory a single workgroup demands grows without bound:
//!
//! ```text
//! head_dim =  32 ->  32 KiB, under the conservative cap
//! head_dim =  64 ->  48 KiB, EXACTLY the conservative cap
//! head_dim = 128 ->  80 KiB, over the conservative cap
//! ```
//!
//! The conservative cap is `max_shared_memory_bytes = 48 * 1024` in
//! `vyre-driver/src/device_profile.rs`. `head_dim = 128` is the standard MLA
//! size, so the interesting case is the one real models use.
//!
//! These tests answer the question the arithmetic alone cannot: does the
//! stack refuse an oversized request by name, or does it dispatch and return
//! quietly wrong numbers? A refusal is a documented ceiling. Silently wrong
//! output at the standard size would be a correctness defect.
//!
//! `mla_decode` is behind the `nn-attention` feature and is absent from the
//! default `vyre-libs` test gate, so nothing here has been exercised before.

#![cfg(test)]

mod common;

use common::{live_dispatcher, reference_outputs};
use vyre_driver::DispatchConfig;
use vyre_foundation::ir::Program;
use vyre_libs::nn::attention::mla::mla_decode;

/// Lane count pinned inside `mla_decode`, mirrored here so the byte
/// arithmetic below is checked against the value the builder actually uses.
const WORKGROUP_LANES: u32 = 64;

/// Score tile edge pinned inside `mla_decode`.
const TILE_SIZE: u32 = 64;

/// Conservative shared-memory cap from `vyre-driver/src/device_profile.rs`.
const CONSERVATIVE_SHARED_MEMORY_CAP: u32 = 48 * 1024;

/// Small fixed problem dims, chosen so only `head_dim` moves between cases.
const SEQ_LEN: u32 = 4;
const NUM_HEADS: u32 = 2;
const KV_LORA_RANK: u32 = 8;
const QK_ROPE_HEAD_DIM: u32 = 4;

/// Absolute tolerance for a single fused-multiply-add chain of this depth.
const TOLERANCE: f32 = 1e-3;

/// Shared-memory bytes one `mla_decode` workgroup requests at `head_dim`.
fn scratch_bytes(head_dim: u32) -> u32 {
    let q_scratch = WORKGROUP_LANES * head_dim;
    let score_tile = WORKGROUP_LANES * TILE_SIZE;
    let o_acc = WORKGROUP_LANES * head_dim;
    (q_scratch + score_tile + o_acc) * 4
}

/// Deterministic, non-degenerate f32 filler.
///
/// Values must not be uniform: uniform input hides index errors because every
/// wrong index still reads a right-looking number.
fn filler(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let i = i as u32;
            let mixed = i.wrapping_mul(0x9E37_79B9) ^ salt.wrapping_mul(0x85EB_CA6B);
            let unit = f32::from(((mixed >> 16) & 0x7FF) as u16) / 2048.0;
            unit - 0.5
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Build `mla_decode` plus its five inputs in binding order for one `head_dim`.
fn case(head_dim: u32) -> (Program, Vec<Vec<u8>>) {
    let program = mla_decode(
        "q",
        "kv_cache",
        "kr_cache",
        "w_uk",
        "w_uv",
        "out",
        SEQ_LEN,
        NUM_HEADS,
        head_dim,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
    )
    .expect("mla_decode must build for positive dims");

    let uv_stride = NUM_HEADS * head_dim;
    let inputs = vec![
        f32_bytes(&filler((NUM_HEADS * head_dim) as usize, 1)),
        f32_bytes(&filler((SEQ_LEN * KV_LORA_RANK) as usize, 2)),
        f32_bytes(&filler((SEQ_LEN * QK_ROPE_HEAD_DIM) as usize, 3)),
        f32_bytes(&filler((KV_LORA_RANK * uv_stride) as usize, 4)),
        f32_bytes(&filler((KV_LORA_RANK * uv_stride) as usize, 5)),
    ];
    (program, inputs)
}

/// ULP budget that enables approximate transcendentals in the PTX emitter.
///
/// MLA softmax uses `Exp`, and `vyre-emit-ptx` refuses to lower `Exp` unless
/// this is positive. Matches the value the emitter's own tests use.
const RELAXED_ULP_BUDGET: u8 = 4;

/// Dispatch on live CUDA, returning the backend's own error rather than
/// panicking, so a refusal can be told apart from wrong numbers.
fn cuda_out(
    program: &Program,
    inputs: &[Vec<u8>],
    ulp_budget: Option<u8>,
) -> Result<Vec<f32>, String> {
    let backend = live_dispatcher();
    let mut config = DispatchConfig::default();
    config.ulp_budget = ulp_budget;
    let outputs = backend
        .dispatch(program, inputs, &config)
        .map_err(|error| error.to_string())?;
    let last = outputs
        .last()
        .ok_or_else(|| "CUDA dispatch returned no output buffers".to_string())?;
    Ok(bytes_to_f32(last))
}

/// Compare live CUDA against the CPU reference, naming the first divergence.
fn assert_matches_cpu(head_dim: u32) {
    let (program, inputs) = case(head_dim);
    let expected = bytes_to_f32(
        reference_outputs(&program, &inputs, &format!("mla_head_dim_{head_dim}"))
            .last()
            .expect("reference must produce the out buffer"),
    );

    let actual = match cuda_out(&program, &inputs, Some(RELAXED_ULP_BUDGET)) {
        Ok(actual) => actual,
        Err(error) => panic!(
            "CUDA REFUSED head_dim={head_dim} ({} bytes shared, cap {}): {error}",
            scratch_bytes(head_dim),
            CONSERVATIVE_SHARED_MEMORY_CAP
        ),
    };

    assert_eq!(
        actual.len(),
        expected.len(),
        "head_dim={head_dim}: output length {} != reference {}",
        actual.len(),
        expected.len()
    );

    let mut divergences = Vec::new();
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        if (got - want).abs() > TOLERANCE || got.is_nan() != want.is_nan() {
            divergences.push((index, *got, *want));
        }
    }
    assert!(
        divergences.is_empty(),
        "head_dim={head_dim} ({} bytes shared, conservative cap {}): \
         {} of {} outputs diverge from the CPU reference. First 8: {:?}",
        scratch_bytes(head_dim),
        CONSERVATIVE_SHARED_MEMORY_CAP,
        divergences.len(),
        expected.len(),
        &divergences[..divergences.len().min(8)]
    );
}

/// Locks the scratch arithmetic itself.
///
/// Bug locked out: someone changes `WORKGROUP_LANES`, `TILE_SIZE`, or a
/// scratch count in `mla_decode` without noticing that shared memory per
/// workgroup moves with it. If this fails, the byte figures quoted in the
/// other tests and in the sweep report are stale and must be recomputed
/// before any ceiling claim based on them is repeated.
#[test]
fn mla_scratch_bytes_scale_linearly_with_head_dim() {
    assert_eq!(scratch_bytes(32), 32 * 1024, "head_dim=32 must be 32 KiB");
    assert_eq!(
        scratch_bytes(64),
        49_152,
        "head_dim=64 must land exactly on the 48 KiB conservative cap"
    );
    assert_eq!(
        scratch_bytes(64),
        CONSERVATIVE_SHARED_MEMORY_CAP,
        "head_dim=64 is the boundary case: exactly at, not over"
    );
    assert_eq!(scratch_bytes(128), 81_920, "head_dim=128 must be 80 KiB");
    assert!(
        scratch_bytes(128) > CONSERVATIVE_SHARED_MEMORY_CAP,
        "head_dim=128 must exceed the conservative cap"
    );
    // The head_dim-dependent term is 128*head_dim*4, so doubling head_dim
    // doubles the delta. The score tile contributes a constant 16 KiB.
    assert_eq!(
        scratch_bytes(64) - scratch_bytes(32),
        16_384,
        "32 -> 64 must add 128*32*4 bytes"
    );
    assert_eq!(
        scratch_bytes(128) - scratch_bytes(64),
        2 * (scratch_bytes(64) - scratch_bytes(32)),
        "doubling head_dim must double the growth increment"
    );
}

/// `mla_decode` cannot run on CUDA under `DispatchConfig::default()`.
///
/// Bug locked out: the refusal degrading into a hang, a crash, or output
/// that merely looks plausible. MLA softmax uses `Exp`, and `vyre-emit-ptx`
/// refuses to lower it without a positive `ulp_budget`. The refusal must
/// stay a NAMED compile-time error that says what to set, because that is
/// the whole difference between a documented ceiling and a silent wrong
/// answer. This test is the ceiling's only owner: `Exp` has no strict CUDA
/// lowering, so MLA decode needs an explicit `ulp_budget` and there is no
/// configuration in which the default refuses for a different reason. If this
/// starts passing, `Exp` gained a strict lowering and the ceiling is gone.
#[test]
fn mla_decode_default_config_refuses_by_name_on_cuda() {
    let (program, inputs) = case(32);
    let error = cuda_out(&program, &inputs, None)
        .expect_err("default DispatchConfig must not silently dispatch MLA softmax");
    assert!(
        error.contains("ulp_budget is not positive"),
        "refusal must name the unset ulp_budget, got: {error}"
    );
    assert!(
        error.contains("Exp"),
        "refusal must name the offending operator, got: {error}"
    );
    assert!(
        error.contains("set an explicit ULP budget"),
        "refusal must state the fix, got: {error}"
    );
}

/// Control: comfortably under the cap, so any failure here is not about
/// shared memory.
///
/// Bug locked out: a regression that breaks `mla_decode` on CUDA generally.
/// If this fails alongside the larger cases, the cause is not the shared
/// memory ceiling and the ceiling diagnosis is wrong.
#[test]
fn mla_decode_head_dim_32_under_cap_matches_cpu() {
    assert_matches_cpu(32);
}

/// Boundary: exactly on the conservative cap, the case a `<` versus `<=`
/// mistake in a cap check would get wrong.
///
/// Bug locked out: an off-by-one in any shared-memory admission check that
/// rejects a request landing exactly on the limit, or admits one byte past
/// it. If this fails while 32 passes, the cap check is off by one.
#[test]
fn mla_decode_head_dim_64_exactly_at_cap_matches_cpu() {
    assert_matches_cpu(64);
}

/// The standard MLA size. OBSERVED: CUDA refuses, it does not corrupt.
///
/// This is the case that decided ceiling versus correctness defect, and the
/// answer is ceiling. At 80 KiB of static workgroup scratch the dispatch is
/// refused, so no wrong attention output is ever produced.
///
/// Bug locked out: this refusal degrading into a silent wrong answer. If
/// `mla_decode` at head_dim=128 ever starts RETURNING values on CUDA, they
/// must be checked against the CPU reference before anyone trusts them,
/// because the scratch request still exceeds the static shared limit and
/// the only safe outcomes are a refusal or a switch to dynamic shared
/// memory.
///
/// Second bug locked out: the diagnostic regressing to the unhelpful one.
/// Before the pre-check in `vyre-driver-cuda/src/backend/host_dispatch/mod.rs`,
/// this surfaced as `CUDA_ERROR_INVALID_PTX` from `cuModuleLoadData`, which
/// points at PTX ISA support for sm_120 when the cause is the scratch
/// request. The message must keep naming the measured bytes, the cap, and
/// the buffers responsible.
#[test]
fn mla_decode_head_dim_128_over_cap_refuses_naming_bytes_and_cap() {
    let (program, inputs) = case(128);
    let error = cuda_out(&program, &inputs, Some(RELAXED_ULP_BUDGET)).expect_err(
        "head_dim=128 requests 80 KiB of static shared memory and must not \
         silently produce attention output",
    );
    assert!(
        error.contains(&scratch_bytes(128).to_string()),
        "refusal must name the measured {} bytes, got: {error}",
        scratch_bytes(128)
    );
    assert!(
        error.contains(&CONSERVATIVE_SHARED_MEMORY_CAP.to_string()),
        "refusal must name the {CONSERVATIVE_SHARED_MEMORY_CAP} byte cap, which is the \
         figure quoted in docs/optimization/README.md. If a device reports a different \
         per-workgroup static limit, update that table rather than loosening this. Got: {error}"
    );
    assert!(
        error.contains("per-workgroup static shared memory limit"),
        "refusal must name the limit it checked, got: {error}"
    );
    for buffer in ["q_scratch", "score_tile", "o_acc"] {
        assert!(
            error.contains(buffer),
            "refusal must name contributing buffer `{buffer}`, got: {error}"
        );
    }
    assert!(
        !error.contains("INVALID_PTX"),
        "refusal must not blame the PTX ISA for a shared memory over-request, got: {error}"
    );
}

/// Discriminator proving the boundary is scratch size, not PTX size.
///
/// head_dim=96 needs 64 KiB, which is over the static shared limit but well
/// under the 80 KiB of the head_dim=128 case, and it emits a SMALLER PTX
/// module. If the refusal tracked PTX length or an sm_120 ISA string, this
/// case would load fine.
///
/// Bug locked out: implementing the shared-memory refusal by sniffing PTX
/// length or the ISA string instead of measuring the scratch request. That
/// would pass the head_dim=128 test and fail here. Getting it wrong sends
/// the next person hunting PTX ISA support, which is exactly what the
/// original `INVALID_PTX` message encouraged.
#[test]
fn mla_decode_head_dim_96_over_cap_also_refuses() {
    assert_eq!(scratch_bytes(96), 64 * 1024, "head_dim=96 must be 64 KiB");
    assert!(
        scratch_bytes(96) > CONSERVATIVE_SHARED_MEMORY_CAP
            && scratch_bytes(96) < scratch_bytes(128),
        "head_dim=96 must sit strictly between the cap and the 128 case"
    );
    let (program, inputs) = case(96);
    let error = cuda_out(&program, &inputs, Some(RELAXED_ULP_BUDGET))
        .expect_err("head_dim=96 exceeds the static shared limit and must refuse");
    assert!(
        error.contains(&scratch_bytes(96).to_string()),
        "refusal must name the measured {} bytes, got: {error}",
        scratch_bytes(96)
    );
}
