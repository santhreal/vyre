//! Live CUDA parity for packed INT4 quantized primitives.
//!
//! The CPU references these contracts diff against are the ones
//! `vyre-libs` ships in `math::quantized`. This file used to carry a
//! private reimplementation of every one of them plus its own little-endian
//! packing, so a correction to a shipped oracle left the CUDA arm asserting
//! bit-exact equality against a definition nobody ships. What stays local is
//! the setup the CUDA arm owns: the lane patterns, the shape tables, the
//! deterministic generators and the binding order the programs are dispatched
//! with, each with one owner below.

#![cfg(feature = "device-tests")]

use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_primitives::wire::{
    decode_f32_le_bytes_all, decode_i32_le_bytes_all, pack_f32_slice, pack_u32_slice,
};
use vyre_reference::composition_witness::{
    i4x8_batched_matmul_f32_scaled_witness as i4x8_batched_matmul_f32_scaled_cpu,
    i4x8_batched_matmul_top1_f32_scaled_witness as i4x8_batched_matmul_top1_f32_scaled_cpu,
    i4x8_batched_matvec_f32_scaled_witness as i4x8_batched_matvec_f32_scaled_cpu,
    i4x8_dot_f32_scaled_witness as i4x8_dot_f32_scaled_cpu,
    i4x8_dot_i32_witness as i4x8_dot_i32_cpu,
    i4x8_matvec_f32_scaled_witness as i4x8_matvec_f32_scaled_cpu,
    pack_i4x8_witness as pack_i4x8_cpu,
};

/// Signed INT4 lane pattern the fixed-shape contracts cycle for weights and for
/// the left dot operand. Spans both nibble extremes so a sign-extension defect
/// in the packed lane decode cannot pass.
const WEIGHT_PATTERN: [i32; 16] = [-8, -3, -1, 0, 1, 3, 7, 6, 5, 4, 2, -2, -4, -6, -7, -5];

/// Signed INT4 lane pattern the fixed-shape contracts cycle for activations and
/// for the right dot operand.
const ACTIVATION_PATTERN: [i32; 16] = [7, 5, 3, 1, -1, -3, -5, -7, 6, 4, 2, 0, -2, -4, -6, -8];

/// Lane counts the fixed-pattern dot contracts sweep: sub-word, word-aligned,
/// word+1, and multi-word tails.
const DOT_LANE_COUNTS: [u32; 9] = [1, 7, 8, 9, 16, 31, 32, 33, 65];

/// `(rows, cols)` shapes the fixed-pattern matvec contract sweeps.
const MATVEC_SHAPES: [(u32, u32); 7] = [(1, 1), (2, 7), (3, 8), (4, 9), (5, 17), (6, 33), (7, 65)];

/// `(batch, rows, cols)` shapes the fixed-pattern batched contracts sweep.
const BATCHED_SHAPES: [(u32, u32, u32); 7] = [
    (1, 1, 1),
    (2, 2, 7),
    (3, 3, 8),
    (4, 4, 9),
    (5, 5, 17),
    (6, 6, 33),
    (3, 7, 65),
];

/// Lane counts the generated release sweep covers. Wider than
/// [`DOT_LANE_COUNTS`]: it adds 2, 15 and 96 so the generated corpus crosses a
/// second word boundary the fixed patterns stop short of.
const GENERATED_DOT_LANE_COUNTS: [u32; 12] = [1, 2, 7, 8, 9, 15, 16, 31, 32, 33, 65, 96];

/// `(rows, cols)` shapes the generated matvec sweep covers. Adds the exactly
/// word-aligned `(3, 64)` case that [`MATVEC_SHAPES`] does not carry.
const GENERATED_MATVEC_SHAPES: [(u32, u32); 8] = [
    (1, 1),
    (2, 7),
    (3, 8),
    (4, 9),
    (5, 17),
    (6, 33),
    (3, 64),
    (7, 65),
];

/// `(batch, rows, cols)` shapes the generated batched sweeps cover. The wide
/// tails use smaller batches than [`BATCHED_SHAPES`] so the generated corpus
/// stays within one dispatch while still reaching 65 columns.
const GENERATED_BATCHED_SHAPES: [(u32, u32, u32); 7] = [
    (1, 1, 1),
    (2, 2, 7),
    (3, 3, 8),
    (4, 4, 9),
    (5, 5, 17),
    (3, 6, 33),
    (2, 7, 65),
];

fn cuda_backend() -> CudaBackend {
    CudaBackend::acquire().expect("Fix: CUDA backend acquire failed on a GPU-required host.")
}

/// `count` lane rows cycled from `pattern`, each row starting `stride` lanes
/// further into the pattern so no two rows share a lane sequence.
fn cycled_rows(pattern: &[i32], count: u32, lanes: u32, stride: usize) -> Vec<Vec<i32>> {
    (0..count as usize)
        .map(|index| {
            pattern
                .iter()
                .copied()
                .cycle()
                .skip(index * stride)
                .take(lanes as usize)
                .collect()
        })
        .collect()
}

/// Pack one INT4 row per entry, padding every row to the packed word stride the
/// INT4 matrix programs bind.
fn pack_i4_matrix_rows(rows: &[Vec<i32>]) -> Vec<u32> {
    let cols = rows.first().map_or(0, Vec::len);
    let words_per_row = cols.div_ceil(8);
    rows.iter()
        .flat_map(|row| {
            let mut packed = pack_i4x8_cpu(row);
            packed.resize(words_per_row, 0);
            packed
        })
        .collect()
}

/// Row scales the fixed-shape contracts bind, one distinct power-of-two-exact
/// value per row so a row/scale mispairing cannot cancel out.
fn patterned_row_scales(rows: u32) -> Vec<f32> {
    (0..rows)
        .map(|row| 0.125_f32 + row as f32 * 0.0625)
        .collect()
}

/// Batch scales the fixed-shape batched matmul contracts bind.
fn patterned_batch_scales(batch: u32) -> Vec<f32> {
    (0..batch)
        .map(|batch_index| 0.25_f32 + batch_index as f32 * 0.03125)
        .collect()
}

/// Packed operands for one batched packed-activation INT4 matmul shape.
struct BatchedMatmulInputs {
    weights_packed: Vec<u32>,
    activations_packed: Vec<u32>,
    row_scales: Vec<f32>,
    batch_scales: Vec<f32>,
}

impl BatchedMatmulInputs {
    /// Bindings in the order `i4x8_batched_matmul_*` declares them.
    fn bindings(&self) -> Vec<Vec<u8>> {
        vec![
            pack_u32_slice(&self.weights_packed),
            pack_u32_slice(&self.activations_packed),
            pack_f32_slice(&self.row_scales),
            pack_f32_slice(&self.batch_scales),
        ]
    }
}

/// Fixed-pattern operands for one batched matmul shape.
fn patterned_batched_matmul_inputs(batch: u32, rows: u32, cols: u32) -> BatchedMatmulInputs {
    BatchedMatmulInputs {
        weights_packed: pack_i4_matrix_rows(&cycled_rows(&WEIGHT_PATTERN, rows, cols, 5)),
        activations_packed: pack_i4_matrix_rows(&cycled_rows(&ACTIVATION_PATTERN, batch, cols, 7)),
        row_scales: patterned_row_scales(rows),
        batch_scales: patterned_batch_scales(batch),
    }
}

/// Generated operands for one batched matmul shape at `seed`.
fn generated_batched_matmul_inputs(
    batch: u32,
    rows: u32,
    cols: u32,
    seed: u32,
) -> BatchedMatmulInputs {
    BatchedMatmulInputs {
        weights_packed: pack_i4_matrix_rows(&generated_i4_rows(
            rows,
            cols,
            seed.wrapping_mul(149) + 31,
        )),
        activations_packed: pack_i4_matrix_rows(&generated_i4_rows(
            batch,
            cols,
            seed.wrapping_mul(151) + 37,
        )),
        row_scales: generated_positive_scales(rows as usize, seed + 41),
        batch_scales: generated_positive_scales(batch as usize, seed + 43),
    }
}

/// Split the packed top-1 output into scores then row indices, the layout
/// `i4x8_batched_matmul_top1_f32_scaled` writes.
fn split_top1(bytes: &[u8], batch: u32) -> (Vec<f32>, Vec<u32>) {
    let packed = read_f32_lanes(bytes, (batch * 2) as usize);
    let scores = packed[..batch as usize].to_vec();
    let indices = packed[batch as usize..]
        .iter()
        .map(|index| *index as u32)
        .collect();
    (scores, indices)
}

fn read_f32(bytes: &[u8]) -> f32 {
    *decode_f32_le_bytes_all(bytes)
        .first()
        .expect("Fix: CUDA INT4 scaled dot must emit one f32.")
}

fn read_i32(bytes: &[u8]) -> i32 {
    *decode_i32_le_bytes_all(bytes)
        .first()
        .expect("Fix: CUDA INT4 dot must emit one i32.")
}

fn read_f32_lanes(bytes: &[u8], count: usize) -> Vec<f32> {
    let mut lanes = decode_f32_le_bytes_all(bytes);
    lanes.truncate(count);
    lanes
}

fn generated_i4_values(len: usize, seed: u32) -> Vec<i32> {
    let mut state = seed ^ 0x9E37_79B9;
    (0..len)
        .map(|index| {
            state = state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .rotate_left((index % 17) as u32);
            ((state >> 28) as i32) - 8
        })
        .collect()
}

fn generated_f32_values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed ^ 0xA5A5_5A5A;
    (0..len)
        .map(|index| {
            state = state
                .wrapping_mul(747_796_405)
                .wrapping_add(2_891_336_453)
                .rotate_right((index % 11) as u32);
            (((state >> 27) & 0x1f) as f32 - 16.0) * 0.0625
        })
        .collect()
}

fn generated_positive_scales(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|index| 0.0625_f32 * (1 + ((seed as usize + index * 3) % 13)) as f32)
        .collect()
}

fn generated_i4_rows(rows: u32, cols: u32, seed: u32) -> Vec<Vec<i32>> {
    (0..rows)
        .map(|row| generated_i4_values(cols as usize, seed.wrapping_add(row * 97)))
        .collect()
}

fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn dispatch_i4_dot_i32(backend: &CudaBackend, lhs: &[u32], rhs: &[u32], lane_count: u32) -> i32 {
    let program = vyre_libs::math::quantized::i4x8_dot_i32("lhs", "rhs", "out", lane_count);
    let outputs = backend
        .dispatch(
            &program,
            &[pack_u32_slice(lhs), pack_u32_slice(rhs)],
            &DispatchConfig::default(),
        )
        .expect("Fix: CUDA INT4 i32 dot dispatch failed.");
    read_i32(&outputs[0])
}

fn dispatch_i4_dot_f32_scaled(
    backend: &CudaBackend,
    lhs: &[u32],
    rhs: &[u32],
    lhs_scale: f32,
    rhs_scale: f32,
    lane_count: u32,
) -> f32 {
    let program = vyre_libs::math::quantized::i4x8_dot_f32_scaled(
        "lhs",
        "rhs",
        "lhs_scale",
        "rhs_scale",
        "out",
        lane_count,
    );
    let outputs = backend
        .dispatch(
            &program,
            &[
                pack_u32_slice(lhs),
                pack_u32_slice(rhs),
                pack_f32_slice(&[lhs_scale]),
                pack_f32_slice(&[rhs_scale]),
            ],
            &DispatchConfig::default(),
        )
        .expect("Fix: CUDA INT4 scaled dot dispatch failed.");
    read_f32(&outputs[0])
}

fn dispatch_i4_matvec_f32_scaled(
    backend: &CudaBackend,
    weights_packed: &[u32],
    x: &[f32],
    scales: &[f32],
    rows: u32,
    cols: u32,
) -> Vec<f32> {
    let program = vyre_libs::math::quantized::i4x8_matvec_f32_scaled(
        "weights", "x", "scales", "out", rows, cols,
    );
    let outputs = backend
        .dispatch(
            &program,
            &[
                pack_u32_slice(weights_packed),
                pack_f32_slice(x),
                pack_f32_slice(scales),
            ],
            &DispatchConfig::default(),
        )
        .expect("Fix: CUDA INT4 scaled matvec dispatch failed.");
    read_f32_lanes(&outputs[0], rows as usize)
}

fn dispatch_i4_batched_matvec_f32_scaled(
    backend: &CudaBackend,
    weights_packed: &[u32],
    x_batches: &[f32],
    scales: &[f32],
    batch: u32,
    rows: u32,
    cols: u32,
) -> Vec<f32> {
    let program = vyre_libs::math::quantized::i4x8_batched_matvec_f32_scaled(
        "weights", "x", "scales", "out", batch, rows, cols,
    );
    let outputs = backend
        .dispatch(
            &program,
            &[
                pack_u32_slice(weights_packed),
                pack_f32_slice(x_batches),
                pack_f32_slice(scales),
            ],
            &DispatchConfig::default(),
        )
        .expect("Fix: CUDA INT4 batched matvec dispatch failed.");
    read_f32_lanes(&outputs[0], (batch * rows) as usize)
}

fn dispatch_i4_batched_matmul_f32_scaled(
    backend: &CudaBackend,
    inputs: &BatchedMatmulInputs,
    batch: u32,
    rows: u32,
    cols: u32,
) -> Vec<f32> {
    let program = vyre_libs::math::quantized::i4x8_batched_matmul_f32_scaled(
        "weights",
        "activations",
        "row_scales",
        "batch_scales",
        "out",
        batch,
        rows,
        cols,
    );
    let outputs = backend
        .dispatch(&program, &inputs.bindings(), &DispatchConfig::default())
        .expect("Fix: CUDA INT4 batched matmul dispatch failed.");
    read_f32_lanes(&outputs[0], (batch * rows) as usize)
}

fn dispatch_i4_batched_matmul_top1_f32_scaled(
    backend: &CudaBackend,
    inputs: &BatchedMatmulInputs,
    batch: u32,
    rows: u32,
    cols: u32,
) -> (Vec<f32>, Vec<u32>) {
    let program = vyre_libs::math::quantized::i4x8_batched_matmul_top1_f32_scaled(
        "weights",
        "activations",
        "row_scales",
        "batch_scales",
        "out",
        batch,
        rows,
        cols,
    );
    let outputs = backend
        .dispatch(&program, &inputs.bindings(), &DispatchConfig::default())
        .expect("Fix: CUDA INT4 batched matmul top1 dispatch failed.");
    split_top1(&outputs[0], batch)
}
fn assert_batched_matmul_parity(
    backend: &CudaBackend,
    inputs: &BatchedMatmulInputs,
    batch: u32,
    rows: u32,
    cols: u32,
    context: &str,
) {
    let actual = dispatch_i4_batched_matmul_f32_scaled(backend, inputs, batch, rows, cols);
    let expected = i4x8_batched_matmul_f32_scaled_cpu(
        &inputs.weights_packed,
        &inputs.activations_packed,
        &inputs.row_scales,
        &inputs.batch_scales,
        batch,
        rows,
        cols,
    );
    assert_eq!(
        f32_bits(&actual),
        f32_bits(&expected),
        "{context} batch={batch} rows={rows} cols={cols}"
    );
}

fn assert_batched_matmul_top1_parity(
    backend: &CudaBackend,
    inputs: &BatchedMatmulInputs,
    batch: u32,
    rows: u32,
    cols: u32,
    context: &str,
) {
    let (actual_scores, actual_indices) =
        dispatch_i4_batched_matmul_top1_f32_scaled(backend, inputs, batch, rows, cols);
    let (expected_scores, expected_indices) = i4x8_batched_matmul_top1_f32_scaled_cpu(
        &inputs.weights_packed,
        &inputs.activations_packed,
        &inputs.row_scales,
        &inputs.batch_scales,
        batch,
        rows,
        cols,
    );
    assert_eq!(
        f32_bits(&actual_scores),
        f32_bits(&expected_scores),
        "{context} score batch={batch} rows={rows} cols={cols}"
    );
    assert_eq!(
        actual_indices, expected_indices,
        "{context} index batch={batch} rows={rows} cols={cols}"
    );
}

fn assert_batched_matvec_parity(
    backend: &CudaBackend,
    weights_packed: &[u32],
    x_batches: &[f32],
    scales: &[f32],
    shape: (u32, u32, u32),
    context: &str,
) {
    let (batch, rows, cols) = shape;
    let actual = dispatch_i4_batched_matvec_f32_scaled(
        backend,
        weights_packed,
        x_batches,
        scales,
        batch,
        rows,
        cols,
    );
    let expected =
        i4x8_batched_matvec_f32_scaled_cpu(weights_packed, x_batches, scales, batch, rows, cols);
    assert_eq!(
        f32_bits(&actual),
        f32_bits(&expected),
        "{context} batch={batch} rows={rows} cols={cols}"
    );
}

fn assert_matvec_parity(
    backend: &CudaBackend,
    weights_packed: &[u32],
    x: &[f32],
    scales: &[f32],
    rows: u32,
    cols: u32,
    context: &str,
) {
    let actual = dispatch_i4_matvec_f32_scaled(backend, weights_packed, x, scales, rows, cols);
    let expected = i4x8_matvec_f32_scaled_cpu(weights_packed, x, scales, rows, cols);
    assert_eq!(
        f32_bits(&actual),
        f32_bits(&expected),
        "{context} rows={rows} cols={cols}"
    );
}

fn assert_dot_f32_scaled_parity(
    backend: &CudaBackend,
    lhs: &[u32],
    rhs: &[u32],
    lhs_scale: f32,
    rhs_scale: f32,
    lane_count: u32,
    context: &str,
) {
    let actual = dispatch_i4_dot_f32_scaled(backend, lhs, rhs, lhs_scale, rhs_scale, lane_count);
    let expected = i4x8_dot_f32_scaled_cpu(lhs, rhs, lhs_scale, rhs_scale, lane_count);
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{context} lane_count={lane_count}"
    );
}

mod batched_matmul_contracts;
mod dot_contracts;
mod generated_sweep_contracts;
mod matvec_contracts;
