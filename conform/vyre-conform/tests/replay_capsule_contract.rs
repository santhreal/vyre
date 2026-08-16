//! Replay capsule verification contracts.

use vyre_conform::ReplayCapsule;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

#[test]
fn replay_capsule_round_trip_and_wire_reproduction() {
    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in_buf", 0, vyre_foundation::ir::BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("out_buf", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::Store {
            buffer: "out_buf".into(),
            index: Expr::u32(0),
            value: Expr::wrapping_add(
                Expr::load("in_buf", Expr::u32(0)),
                Expr::u32(100),
            ),
        }],
    );

    let input_bytes = 42u32.to_le_bytes();
    let inputs = [input_bytes.as_slice()];

    let capsule = ReplayCapsule::from_program(&prog, &inputs, Some("SAN003_DATA_RACE".to_string()))
        .expect("capsule construction must succeed");

    assert!(!capsule.wire_bytes.is_empty());
    assert_eq!(capsule.inputs.len(), 1);
    assert_eq!(capsule.inputs[0], input_bytes);
    assert_eq!(
        capsule.expected_diagnostic_code.as_deref(),
        Some("SAN003_DATA_RACE")
    );

    // Serialization round-trip
    let json = serde_json::to_string(&capsule).expect("capsule must serialize");
    let deserialized: ReplayCapsule = serde_json::from_str(&json).expect("capsule must deserialize");
    assert_eq!(deserialized.wire_bytes, capsule.wire_bytes);
    assert_eq!(deserialized.inputs, capsule.inputs);
    assert_eq!(
        deserialized.expected_diagnostic_code,
        capsule.expected_diagnostic_code
    );
}
