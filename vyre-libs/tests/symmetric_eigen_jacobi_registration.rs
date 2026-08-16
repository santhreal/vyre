//! Canonical registration contract of `vyre-primitives::math::symmetric_eigen_jacobi`.
//!
//! WHY: the eigensolve is a reusable primitive with its own `OP_ID` that
//! `tensor_train_decompose` composes as a child region, but for a while it carried no
//! `OperationRegistration`. `docs/generated/OP_SCHEMA.json` recorded it as
//! `"registered": false` and `cargo xtask abstraction-gate` reported it as an
//! UNREGISTERED-CHILD. Nothing in the crate's own test suite noticed, because every other
//! eigensolver test drives the builder directly and never looks at the registry.
//!
//! This test pins the registry entry itself and the two things a registration is for:
//!
//! 1. the entry exists, is a primitive, and ships BOTH fixtures (a half-registered op passes
//!    the conform harness by covering zero cases),
//! 2. the fixture shapes match the built program's buffer layout, so a later change to `n` or
//!    to the buffer list cannot leave a stale oracle in place,
//! 3. running the registered `test_inputs` on the reference backend reproduces the registered
//!    `expected_output` within the registered tolerance, element by element, and
//! 4. the registered oracle is a genuine eigendecomposition of the registered input:
//!    `A·v_k = λ_k·v_k` and `VᵀV = I`, checked in f64 against the ORIGINAL matrix.
//!
//! Point 4 is what stops this from being a tautology. Points 1-3 would still pass if someone
//! replaced both fixtures with whatever the kernel happens to emit today; point 4 fails unless
//! the pinned values really are eigenpairs.
//!
//! What it does not catch: a rotation-order change inside `jacobi_eigen_body` that still lands
//! on the same eigenbasis for this witness. `symmetric_eigen_jacobi_parity` covers the general
//! matrix, this covers the pinned one.
#![cfg(all(feature = "math", feature = "inventory-registry"))]

use vyre_foundation::fp_parity::ulp_distance;
use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};
use vyre_libs::math::eigenvector_column_sign::EIGENVECTOR_SIGN_EPSILON;
use vyre_libs::math::symmetric_eigen_jacobi::OP_ID;
use vyre_primitives::wire::decode_f32_le_bytes_all as unpack_f32;
use vyre_reference::value::Value;

/// Order of the registered witness matrix.
const N: usize = 4;

fn registered() -> SemanticOperation {
    OperationRegistry::global().get(OP_ID).unwrap_or_else(|| {
        panic!(
            "{OP_ID} is not in the operation registry. Fix: it is composed as a child region by \
             tensor_train_decompose, so it needs its own OperationRegistration::primitive."
        )
    })
}

#[test]
fn registry_entry_ships_both_fixtures_with_the_program_buffer_shape() {
    let entry = registered();
    assert_eq!(
        entry.tier,
        OperationTier::Intrinsic,
        "{OP_ID} is a Category C intrinsic, not a library composition."
    );

    let test_inputs = entry
        .test_inputs
        .expect("registered eigensolve must ship test_inputs");
    let expected_output = entry
        .expected_output
        .expect("registered eigensolve must ship expected_output");
    let inputs = test_inputs();
    let expected = expected_output();
    assert!(!inputs.is_empty(), "empty test_inputs are zero coverage");
    assert_eq!(
        inputs.len(),
        expected.len(),
        "every witness case needs exactly one oracle case"
    );

    let program = entry.program().expect("registration must build a program");
    let buffers = program.buffers();
    assert_eq!(
        inputs[0].len(),
        buffers.len(),
        "test_inputs must carry one entry per declared buffer, in declaration order"
    );
    let writable = buffers
        .iter()
        .filter(|buffer| buffer.access != vyre_foundation::ir::BufferAccess::ReadOnly)
        .count();
    assert_eq!(
        expected[0].len(),
        writable,
        "expected_output must carry one entry per written buffer"
    );
    for (index, buffer) in buffers.iter().enumerate() {
        let declared = buffer.count as usize;
        assert_eq!(
            inputs[0][index].len(),
            declared * 4,
            "buffer `{}` fixture is {} bytes but the program declares {declared} f32",
            buffer.name,
            inputs[0][index].len()
        );
    }
}

#[test]
fn reference_execution_reproduces_the_registered_oracle_within_tolerance() {
    let entry = registered();
    let program = entry.program().expect("registration must build a program");
    let tolerance = entry.tolerance();
    let inputs = (entry.test_inputs.expect("test_inputs"))();
    let expected = (entry.expected_output.expect("expected_output"))();

    for (case, (input_case, expected_case)) in inputs.iter().zip(expected.iter()).enumerate() {
        let values: Vec<Value> = input_case.iter().cloned().map(Value::from).collect();
        let outputs = vyre_reference::reference_eval(&program, &values)
            .expect("reference evaluation of the registered eigensolve must succeed");
        assert_eq!(
            outputs.len(),
            expected_case.len(),
            "case {case}: reference produced {} output buffers, oracle declares {}",
            outputs.len(),
            expected_case.len()
        );

        for (slot, (actual, oracle)) in outputs.iter().zip(expected_case.iter()).enumerate() {
            let actual = unpack_f32(&actual.to_bytes());
            let oracle = unpack_f32(oracle);
            assert_eq!(
                actual.len(),
                oracle.len(),
                "case {case} buffer {slot}: length mismatch"
            );
            for (index, (got, want)) in actual.iter().zip(oracle.iter()).enumerate() {
                assert_eq!(
                    got.is_sign_negative(),
                    want.is_sign_negative(),
                    "case {case} buffer {slot} element {index}: sign differs ({got} vs {want}). \
                     The eigenvector sign convention is part of the contract."
                );
                let drift = ulp_distance(*got, *want)
                    .unwrap_or_else(|| panic!("case {case} buffer {slot} element {index}: NaN"));
                assert!(
                    drift <= tolerance,
                    "case {case} buffer {slot} element {index}: {got} vs oracle {want} is \
                     {drift} ULP, registered tolerance is {tolerance}."
                );
            }
        }
    }
}

#[test]
fn registered_oracle_is_a_real_eigendecomposition_of_the_registered_input() {
    let entry = registered();
    let inputs = (entry.test_inputs.expect("test_inputs"))();
    let expected = (entry.expected_output.expect("expected_output"))();

    // Buffer 0 is the symmetric matrix; the kernel overwrites it, so the ORIGINAL comes from
    // test_inputs and the eigenvectors/eigenvalues from expected_output slots 1 and 2.
    let matrix: Vec<f64> = unpack_f32(&inputs[0][0])
        .into_iter()
        .map(f64::from)
        .collect();
    let eigenvectors: Vec<f64> = unpack_f32(&expected[0][1])
        .into_iter()
        .map(f64::from)
        .collect();
    let eigenvalues: Vec<f64> = unpack_f32(&expected[0][2])
        .into_iter()
        .map(f64::from)
        .collect();
    assert_eq!(matrix.len(), N * N);
    assert_eq!(eigenvectors.len(), N * N);
    assert_eq!(eigenvalues.len(), N);

    // The witness is only pinnable because its eigenvalues are simple. Prove that here rather
    // than asserting it in a comment: a later edit that makes two of them coincide admits a
    // different-but-valid eigenbasis and silently invalidates the fixture.
    let mut sorted = eigenvalues.clone();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite eigenvalues"));
    let norm = eigenvalues.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    for pair in sorted.windows(2) {
        assert!(
            pair[1] - pair[0] > 0.1 * norm,
            "registered eigenvalues {sorted:?} are not well separated relative to norm {norm}"
        );
    }

    // A·v_k = λ_k·v_k.
    for k in 0..N {
        for row in 0..N {
            let lhs: f64 = (0..N)
                .map(|col| matrix[row * N + col] * eigenvectors[col * N + k])
                .sum();
            let rhs = eigenvalues[k] * eigenvectors[row * N + k];
            assert!(
                (lhs - rhs).abs() < 1.0e-5 * norm,
                "eigenpair {k} row {row}: A·v = {lhs}, λ·v = {rhs}"
            );
        }
    }

    // VᵀV = I.
    for left in 0..N {
        for right in 0..N {
            let dot: f64 = (0..N)
                .map(|row| eigenvectors[row * N + left] * eigenvectors[row * N + right])
                .sum();
            let want = if left == right { 1.0 } else { 0.0 };
            assert!(
                (dot - want).abs() < 1.0e-6,
                "VᵀV[{left},{right}] = {dot}, expected {want}"
            );
        }
    }

    // The sign convention the body applies: the first component above the epsilon is positive.
    for k in 0..N {
        let epsilon = f64::from(EIGENVECTOR_SIGN_EPSILON);
        let first = (0..N)
            .map(|row| eigenvectors[row * N + k])
            .find(|value| value.abs() > epsilon)
            .unwrap_or_else(|| panic!("eigenvector column {k} is entirely below the sign epsilon"));
        assert!(
            first > 0.0,
            "eigenvector column {k} is not sign-canonicalized: leading component {first}"
        );
    }
}
