//! Shared diagnostic protocol regression contracts.

use vyre_foundation::diagnostics::{Diagnostic, DiagnosticStage, OpLocation, RetryClass, Severity};

/// WHY: public workflow failures cross compiler, packaging, runtime, and driver
/// boundaries. Serialization must preserve the code, stage, typed location,
/// cause, correction, and retry decision instead of collapsing them into prose.
#[test]
fn diagnostic_round_trip_preserves_workflow_identity() {
    let diagnostic = Diagnostic::error("MKC016_DIGEST_MISMATCH", "artifact digest mismatch")
        .with_stage(DiagnosticStage::Admit)
        .with_location(
            OpLocation::op("artifact.authenticate")
                .with_graph_node(7)
                .with_graph_value(11)
                .with_path("artifact.envelope")
                .with_source_span(128, 256),
        )
        .with_cause(
            "digest_mismatch",
            "declared digest differs from canonical body",
        )
        .with_fix("discard the envelope and recompile the source graph")
        .with_retry(RetryClass::RecompileSource)
        .with_doc_url("https://docs.vyre.dev/errors#mkc016")
        .with_note("expected digest sha256:abcd..., observed sha256:1234...")
        .with_note("envelope version 7 format requires fresh compile");

    let encoded = serde_json::to_vec(&diagnostic).expect("diagnostic must serialize");
    let decoded: Diagnostic =
        serde_json::from_slice(&encoded).expect("diagnostic must deserialize");

    assert_eq!(decoded, diagnostic);
    assert_eq!(decoded.severity, Severity::Error);
    assert_eq!(decoded.code.as_str(), "MKC016_DIGEST_MISMATCH");
    assert_eq!(decoded.stage, DiagnosticStage::Admit);
    assert_eq!(decoded.retry, RetryClass::RecompileSource);
    let location = decoded.location.expect("typed location must survive");
    assert_eq!(location.graph_node, Some(7));
    assert_eq!(location.graph_value, Some(11));
    assert_eq!(location.path.as_deref(), Some("artifact.envelope"));
    assert_eq!(location.source_span, Some([128, 256]));
    let cause = decoded.cause.expect("structured cause must survive");
    assert_eq!(cause.kind, "digest_mismatch");
    assert_eq!(decoded.notes.len(), 2);
    assert_eq!(
        decoded.notes[0].as_ref(),
        "expected digest sha256:abcd..., observed sha256:1234..."
    );
}

#[test]
fn diagnostic_render_human_and_json_snapshot() {
    let diagnostic = Diagnostic::error("V028", "Fma operand has type i32, expected f32")
        .with_stage(DiagnosticStage::Validate)
        .with_location(
            OpLocation::op("math.fma")
                .with_operand(1)
                .with_path("kernel.vyre")
                .with_source_span(42, 58),
        )
        .with_fix("cast operand 1 from i32 to f32 using Expr::cast")
        .with_cause("typecheck", "operand 1 type mismatch in FMA")
        .with_note("FMA requires all three float operands to share the same scalar type");

    let rendered = diagnostic.render_human();
    assert!(rendered.contains("error[V028](Validate): Fma operand has type i32, expected f32"));
    assert!(rendered.contains("--> op `math.fma` operand[1] at kernel.vyre:42..58"));
    assert!(rendered.contains("= help: cast operand 1 from i32 to f32 using Expr::cast"));
    assert!(rendered.contains("= cause[typecheck]: operand 1 type mismatch in FMA"));
    assert!(rendered
        .contains("= note: FMA requires all three float operands to share the same scalar type"));

    let json = diagnostic.to_json();
    assert!(json.contains("\"code\":\"V028\""));
    assert!(json.contains("\"stage\":\"validate\""));
    assert!(json.contains("\"source_span\":[42,58]"));
}

#[test]
fn diagnostic_from_validation_error_covers_invalid_program_and_type_errors() {
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
    use vyre_foundation::validate::validate;

    // Type error program: FMA with mismatched float types
    let type_prog = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::F32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::fma(Expr::f32(1.0), Expr::u32(2), Expr::f32(3.0)),
        )],
    );
    let type_errors = validate(&type_prog);
    assert!(
        !type_errors.is_empty(),
        "type mismatch in FMA must yield validation error"
    );
    let type_diag = type_errors[0].diagnostic();
    assert_eq!(type_diag.severity, Severity::Error);
    assert_eq!(type_diag.stage, DiagnosticStage::Validate);
    assert!(!type_diag.code.as_str().is_empty());
    assert!(type_diag.suggested_fix.is_some());

    // Illegal memory order error program
    let mem_prog = Program::wrapped(
        vec![BufferDecl::storage(
            "buf",
            0,
            vyre_foundation::ir::BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "buf",
            Expr::u32(0),
            Expr::atomic_add_ordered(
                "buf",
                Expr::u32(0),
                Expr::u32(1),
                vyre_foundation::ir::MemoryOrdering::GridSync,
            ),
        )],
    );
    let mem_errors = validate(&mem_prog);
    assert!(
        !mem_errors.is_empty(),
        "illegal memory ordering in atomic RMW must yield validation error"
    );
    let mem_diag = mem_errors[0].diagnostic();
    assert_eq!(mem_diag.stage, DiagnosticStage::Validate);
    assert!(!mem_diag.code.as_str().is_empty());
    assert!(mem_diag.suggested_fix.is_some());
}

#[test]
fn diagnostic_from_ir_error_covers_wire_and_inlining_errors() {
    use vyre_foundation::IrError;

    // Wire format validation error
    let wire_err = IrError::WireFormatValidation {
        message: "corrupted node header magic in section 3".to_string(),
    };
    let wire_diag: Diagnostic = wire_err.diagnostic();
    assert_eq!(wire_diag.code.as_str(), "WIRE001_VALIDATION_FAILED");
    assert_eq!(wire_diag.stage, DiagnosticStage::Validate);
    assert!(wire_diag.suggested_fix.is_some());

    // Wire version mismatch
    let ver_err = IrError::VersionMismatch {
        expected: 7,
        found: 4,
    };
    let ver_diag: Diagnostic = ver_err.diagnostic();
    assert_eq!(ver_diag.code.as_str(), "WIRE002_VERSION_MISMATCH");
    assert_eq!(ver_diag.stage, DiagnosticStage::Admit);
    assert!(ver_diag
        .suggested_fix
        .unwrap()
        .contains("re-encode with a matching vyre version"));

    // Inlining cycle
    let cycle_err = IrError::InlineCycle {
        op_id: "math.recursive_fib".to_string(),
    };
    let cycle_diag: Diagnostic = cycle_err.diagnostic();
    assert_eq!(cycle_diag.code.as_str(), "IRC001_INLINE_CYCLE");
    assert_eq!(cycle_diag.stage, DiagnosticStage::Optimize);
    assert_eq!(cycle_diag.location.unwrap().op_id, "math.recursive_fib");
    assert!(cycle_diag
        .suggested_fix
        .unwrap()
        .contains("remove the recursive Expr::Call chain"));
}
