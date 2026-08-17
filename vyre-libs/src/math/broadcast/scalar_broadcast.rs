//! Scalar broadcast  -  copy a single-element `src` to every slot of `dst`.
//!
//! Category A composition. The minimal broadcast case; a full
//! shape-broadcasting version (NumPy semantics) belongs in a future
//! `broadcast_shaped` function that takes source + target shapes.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

/// Broadcast a scalar into every element of `dst`. `n` is the target
/// element count  -  `dst` receives `n × sizeof(U32)` bytes.
#[must_use]
pub fn broadcast(src: &str, dst: &str, n: u32) -> Program {
    if n == 0 {
        return trap_program(
            "vyre-libs::math::broadcast",
            Some((dst, DataType::U32)),
            "Fix: broadcast requires n > 0.".to_string(),
        );
    }
    ElementwiseComposer::broadcast_scalar("vyre-libs::math::broadcast", src, dst, n, DataType::U32)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::math::broadcast",
        || broadcast("src", "dst", 4),
        Some(|| vec![vec![
            42u32.to_le_bytes().to_vec(),                       // src: scalar 42
        ]]),
        Some(|| vec![vec![
            // Only ReadWrite buffer: dst filled with 42
            vec![
                0x2a, 0x00, 0x00, 0x00, // 42
                0x2a, 0x00, 0x00, 0x00, // 42
                0x2a, 0x00, 0x00, 0x00, // 42
                0x2a, 0x00, 0x00, 0x00, // 42
            ],
        ]]),
    )
    .with_category("math")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::{bytes_to_u32 as decode_u32_words, u32_bytes};
    use vyre_reference::value::Value;

    #[test]
    fn broadcast_single_element() {
        let program = broadcast("src", "dst", 1);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[Value::from(u32_bytes(&[99u32])), Value::from(vec![0u8; 4])],
        )
        .expect("Fix: broadcast n=1 must execute");
        let actual = decode_u32_words(&outputs[0].to_bytes());
        assert_eq!(actual, vec![99u32]);
    }

    #[test]
    fn broadcast_zero_elements_should_trap_or_be_consistent() {
        let program = broadcast("src", "dst", 0);
        let error = vyre_reference::reference_eval(
            &program,
            &[Value::from(u32_bytes(&[99u32])), Value::from(vec![0u8; 0])],
        )
        .expect_err("broadcast n=0 must trap instead of succeeding");
        let msg = error.to_string();
        assert!(
            msg.contains("trap") || msg.contains("Fix:"),
            "broadcast n=0 error must be actionable: {msg}"
        );
    }
}
