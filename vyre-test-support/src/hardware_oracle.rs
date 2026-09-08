//! The CPU-oracle floor the hardware registration, out-of-bounds, and
//! generated hardware matrix suites share.
//!
//! Library items, not a `#[path]`-included module: every helper here has a
//! reachable owner, so a suite that reads one of them needs no dead-code
//! suppression for the four it does not.

use vyre_foundation::ir::Program;
use vyre_foundation::operation::SemanticOperation;

/// Execute a registered operation on the CPU reference interpreter.
pub fn run_cpu(entry: &SemanticOperation, inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let program = entry
        .program()
        .expect("Fix: registered hardware intrinsic must provide a neutral builder");
    let values = vyre_reference::reference_inputs(&program, inputs.to_vec());
    vyre_reference::reference_eval(&program, &values)
        .expect("Fix: registered hardware intrinsic must execute on the CPU oracle.")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
}

/// Execute a Program on the CPU reference interpreter, asserting exactly one output buffer.
pub fn run_eval_single(program: &Program, inputs: Vec<Vec<u8>>) -> Vec<u8> {
    let values = vyre_reference::reference_inputs(program, inputs);
    let outputs = vyre_reference::reference_eval(program, &values)
        .expect("Fix: hardware intrinsic builder must execute on the CPU oracle.");
    assert_eq!(
        outputs.len(),
        1,
        "Fix: hardware intrinsic builder must produce exactly one output buffer"
    );
    outputs[0].to_bytes()
}

/// Reference evaluation for inverse_sqrt_f32 with IEEE-754 and domain clamping.
pub fn inverse_sqrt_f32_ref(x: f32) -> f32 {
    let safe = if !x.is_finite() || x <= f32::MIN_POSITIVE {
        f32::MIN_POSITIVE
    } else {
        x
    };
    safe.sqrt().recip()
}

/// Generate u32 test inputs with edge values up to edge.len(), followed by deterministic LCG words.
pub fn generated_u32_with_edges(len: usize, seed: u32, edge: &[u32]) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|idx| {
            if idx < edge.len() {
                edge[idx]
            } else {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state.rotate_left((idx as u32) & 31) ^ ((idx as u32).wrapping_mul(0x9e37_79b9))
            }
        })
        .collect()
}

/// Generate f32 test inputs with edge values up to edge.len(), followed by deterministic LCG floats.
pub fn generated_f32_with_edges(len: usize, seed: u32, edge: &[f32]) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|idx| {
            if idx < edge.len() {
                edge[idx]
            } else {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = f32::from_bits((state >> 9) | 0x3f00_0000) - 1.0;
                if idx & 1 == 0 {
                    unit
                } else {
                    -unit
                }
            }
        })
        .collect()
}
