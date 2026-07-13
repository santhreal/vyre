//! GPU-IR parity for `opt/homotopy::homotopy_euler_predictor`: the Euler predictor step of homotopy
//! continuation `x_pred = x_curr + dt·v` in 16.16 fixed point, driven through
//! `vyre_reference::reference_eval` with SIGNED states and tangent vectors.
//!
//! Why this test exists: the predictor advances along the path TANGENT `v`, which is freely SIGNED
//! (the path turns in every direction as `t: 0→1`), and the state `x_curr` is signed. The step is a
//! SIGNED `fixed_mul_16_16_expr` (`dt·v`) followed by a two's-complement `Expr::add`. Before the
//! signed-multiply fix (BACKLOG `FIXED-amg-fixed-path-unsigned-mul-negatives`) `dt·v[i]` reconstructed
//! its product from the UNSIGNED high word, so a negative tangent component (a u32 with the top bit
//! set, read as ~2^32) produced a garbage predictor step, sending the path the wrong way. This is
//! not a toy: vyre's OWN megakernel-scheduler ILP (#22 self-consumer) is solved by following this
//! homotopy path on GPU, so a sign bug here mis-schedules real kernels. The primitive had NO
//! IR-execution parity coverage; this is the first faithful signed run.
//!
//! BIT-EXACT: pure integer arithmetic 
//! `fixed_mul(a,b) = ((a as i32 as i64 * b as i32 as i64) >> 16) as i32 as u32`, then `wrapping_add`.
#![cfg(feature = "opt")]

use vyre_primitives::opt::homotopy::homotopy_euler_predictor;
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_reference::value::Value;

const FIXED_ONE: f64 = 65536.0;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn to_fixed(v: f64) -> u32 {
    (v * FIXED_ONE).round() as i64 as u32
}

/// Bit-exact replica of the predictor's SIGNED 16.16 multiply.
fn fixed_mul(a: u32, b: u32) -> u32 {
    ((i64::from(a as i32) * i64::from(b as i32)) >> 16) as i32 as u32
}

/// A signed 16.16 value in roughly `[-4.0, 4.0)`: an 18-bit magnitude, optionally negated.
fn signed_fixed(state: &mut u32) -> u32 {
    let magnitude = (xorshift(state) & 0x0003_FFFF) as i32; // [0.0, 4.0) in 16.16
    if xorshift(state) & 1 == 0 {
        magnitude as u32
    } else {
        (-magnitude) as u32
    }
}

/// Exact u32 16.16 oracle: `x_pred[t] = x_curr[t] + fixed_mul(dt, v[t])`.
fn euler_fixed(x_curr: &[u32], v: &[u32], dt: u32) -> Vec<u32> {
    (0..x_curr.len())
        .map(|t| x_curr[t].wrapping_add(fixed_mul(dt, v[t])))
        .collect()
}

fn run_via_reference(x_curr: &[u32], v: &[u32], dt: u32, n_paths: u32, n_dim: u32) -> Vec<u32> {
    let program = homotopy_euler_predictor("x_curr", "v", "dt", "x_pred", n_paths, n_dim);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(x_curr)),
            Value::from(pack_u32(v)),
            Value::from(pack_u32(&[dt])),
            Value::from(pack_u32(&vec![0u32; x_curr.len()])),
        ],
    )
    .expect("homotopy_euler_predictor reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn homotopy_euler_signed_matches_exact_fixed_point_step() {
    let mut state = 0xF00D_1234u32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut moved = 0u32;
    for case in 0..400u32 {
        let n_paths = 1 + (case % 4);
        let n_dim = 1 + ((case / 4) % 4);
        let cells = (n_paths * n_dim) as usize;
        let x_curr: Vec<u32> = (0..cells).map(|_| signed_fixed(&mut state)).collect();
        let v: Vec<u32> = (0..cells).map(|_| signed_fixed(&mut state)).collect();
        // dt is a step size: physically positive, but a reverse/predictor step can carry either sign.
        let dt = signed_fixed(&mut state);

        neg_inputs += x_curr
            .iter()
            .chain(&v)
            .filter(|&&val| (val as i32) < 0)
            .count() as u32;

        let got = run_via_reference(&x_curr, &v, dt, n_paths, n_dim);
        let want = euler_fixed(&x_curr, &v, dt);
        assert_eq!(
            got, want,
            "case {case} (n_paths={n_paths} n_dim={n_dim}): SIGNED Euler predictor _via {got:?} != \
             exact signed oracle {want:?} (x_curr={x_curr:?} v={v:?} dt={dt})"
        );

        if want.iter().any(|&val| val != 0) {
            moved += 1;
        }
        neg_outputs += want.iter().filter(|&&val| (val as i32) < 0).count() as u32;
    }
    assert!(
        neg_inputs > 500,
        "sweep must feed many negative state/tangent entries, got {neg_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "signed predictor steps must produce negative predicted states, got {neg_outputs}"
    );
    assert!(
        moved > 380,
        "only {moved}/400 steps were non-zero, the predictor is not being exercised"
    );
}

#[test]
fn homotopy_euler_hand_checked_signed() {
    // x_curr = [1.0, -2.0], v = [-4.0, 1.0], dt = 0.5:
    //   x_pred[0] = 1.0 + (0.5)(-4.0) = 1.0 - 2.0 = -1.0
    //   x_pred[1] = -2.0 + (0.5)(1.0) = -2.0 + 0.5 = -1.5
    let x_curr = vec![to_fixed(1.0), to_fixed(-2.0)];
    let v = vec![to_fixed(-4.0), to_fixed(1.0)];
    let dt = to_fixed(0.5);
    let got = run_via_reference(&x_curr, &v, dt, 1, 2);
    let want = euler_fixed(&x_curr, &v, dt);
    assert_eq!(
        want,
        vec![to_fixed(-1.0), to_fixed(-1.5)],
        "sanity: signed Euler predictor = [-1.0, -1.5]"
    );
    assert_eq!(
        got, want,
        "the dispatched predictor must preserve sign: [-1.0, -1.5]"
    );
}
