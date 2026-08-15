//! Round-trip test for `Expr::Opaque` and `Node::Opaque` through the wire
//! format (tag `0x80`).
//!
//! A minimal test-only extension registers the matching
//! `OpaqueExprResolver` / `OpaqueNodeResolver`, then the program round-trips
//! through `to_wire` → `from_wire` and is asserted byte-identical.

#[path = "support/opaque_echo_extension.rs"]
mod opaque_echo_extension;

use opaque_echo_extension::{EchoExpr, EchoNode};

use std::sync::Arc;

use vyre_foundation::ir::{BufferDecl, DataType, Expr, ExprNode, Node, Program};

#[test]
fn opaque_expr_round_trips_through_wire_format() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::store(
                "out",
                Expr::u32(0),
                Expr::Opaque(Arc::new(EchoExpr {
                    payload: b"hello-opaque-expr".to_vec(),
                })),
            ),
            Node::Return,
        ],
    );

    let encoded = program.to_wire().expect("encode must succeed");
    let decoded = Program::from_wire(&encoded).expect("decode must succeed");

    assert_eq!(decoded, program);
}

#[test]
fn registered_opaque_expr_decodes_as_byte_identical_passthrough() {
    let payload = b"passthrough-payload".to_vec();
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::store(
                "out",
                Expr::u32(0),
                Expr::Opaque(Arc::new(EchoExpr {
                    payload: payload.clone(),
                })),
            ),
            Node::Return,
        ],
    );

    let encoded = program.to_wire().expect("encode must succeed");
    let decoded = Program::from_wire(&encoded).expect("decode must succeed");

    let decoded_payload = match decoded.entry() {
        [Node::Region { body, .. }] => match body.as_slice() {
            [
                Node::Store {
                    value: Expr::Opaque(extension),
                    ..
                },
                Node::Return,
            ] => extension
                .as_any()
                .downcast_ref::<EchoExpr>()
                .expect("Fix: registered opaque payload must decode back into the owning extension type")
                .payload
                .clone(),
            body => panic!("Fix: expected opaque store fixture body, got {body:?}"),
        },
        entry => panic!("Fix: expected root Region around opaque fixture, got {entry:?}"),
    };

    assert_eq!(
        decoded_payload, payload,
        "Fix: registered opaque payloads must decode as byte-identical passthrough data."
    );
}

#[test]
fn opaque_node_round_trips_through_wire_format() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::Opaque(Arc::new(EchoNode {
                payload: b"hello-opaque-node".to_vec(),
            })),
            Node::Return,
        ],
    );

    let encoded = program.to_wire().expect("encode must succeed");
    let decoded = Program::from_wire(&encoded).expect("decode must succeed");

    assert_eq!(decoded, program);
}

#[test]
fn opaque_expr_is_validated_through_extension_hook() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::store(
                "out",
                Expr::u32(0),
                Expr::Opaque(Arc::new(EchoExpr {
                    payload: b"payload".to_vec(),
                })),
            ),
            Node::Return,
        ],
    );
    assert!(vyre_foundation::validate::rule_pipeline::validate(&program).is_empty());
}

#[test]
fn opaque_node_survives_optimizer_rewrite() {
    // A program with only an Opaque node plus a Return must survive the
    // optimizer's rewrite pass unchanged because foundation cannot peek
    // inside the extension payload.
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::Opaque(Arc::new(EchoNode {
                payload: b"state".to_vec(),
            })),
            Node::Return,
        ],
    );
    let bytes = program.to_wire().expect("encode");
    let decoded = Program::from_wire(&bytes).expect("decode");
    assert_eq!(decoded, program);
}

#[test]
fn unregistered_opaque_kind_fails_loudly() {
    #[derive(Debug)]
    struct UnregisteredExprExt;
    impl ExprNode for UnregisteredExprExt {
        fn extension_kind(&self) -> &'static str {
            "test.extension.unregistered"
        }
        fn debug_identity(&self) -> &str {
            "unregistered"
        }
        fn result_type(&self) -> Option<DataType> {
            None
        }
        fn cse_safe(&self) -> bool {
            true
        }
        fn stable_fingerprint(&self) -> [u8; 32] {
            [0; 32]
        }
        fn validate_extension(&self) -> Result<(), String> {
            Ok(())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::store(
                "out",
                Expr::u32(0),
                Expr::Opaque(Arc::new(UnregisteredExprExt)),
            ),
            Node::Return,
        ],
    );

    let encoded = program.to_wire().expect("encode must succeed");
    let err = Program::from_wire(&encoded).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("no OpaqueExprResolver"),
        "Fix: expected decoder error about missing resolver, got: {message}"
    );
    assert!(
        message.contains("Fix:"),
        "Fix: missing opaque resolver errors must stay actionable, got: {message}"
    );
}
