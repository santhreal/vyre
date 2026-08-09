//! Backend-neutral C source ingestion and typed-IR contract tests.

use vyre_foundation::ir::{validate, BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_frontend_c::{
    lower_source, lower_translation_unit, parse_source, parse_source_bytes, CFrontendError,
    MAX_SOURCE_BYTES,
};

#[test]
fn scalar_kernel_lowers_to_exact_typed_ir_without_execution() {
    let source = "unsigned int kernel(void) { return 6u * 7u; }";
    let parsed = parse_source(source).expect("source ingestion must succeed independently");
    assert_eq!(parsed.source(), source);
    let program =
        lower_translation_unit(&parsed).expect("typed-IR construction must consume parsed source");

    assert!(
        validate(&program).is_empty(),
        "frontend must return structurally valid typed IR"
    );
    let expected = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::mul(Expr::u32(6), Expr::u32(7)),
        )],
    );
    assert_eq!(program, expected);
}

#[test]
fn pointer_kernel_preserves_buffer_types_bindings_and_access() {
    let program = lower_source(
        "void kernel(const unsigned int *input, unsigned int *output) { output[0] = input[0] + 1u; }",
    )
    .expect("supported buffer C must lower");

    assert!(
        validate(&program).is_empty(),
        "lowered buffer program must be valid"
    );
    assert_eq!(program.buffers.len(), 2);
    assert_eq!(program.buffers[0].name(), "input");
    assert_eq!(program.buffers[0].binding(), 0);
    assert_eq!(program.buffers[0].access(), BufferAccess::ReadOnly);
    assert_eq!(program.buffers[0].element(), DataType::U32);
    assert_eq!(program.buffers[1].name(), "output");
    assert_eq!(program.buffers[1].binding(), 1);
    assert_eq!(program.buffers[1].access(), BufferAccess::ReadWrite);
    assert_eq!(program.buffers[1].element(), DataType::U32);
}

#[test]
fn unsupported_semantics_are_rejected_instead_of_silently_mislowered() {
    let error = lower_source("unsigned int kernel(void) { unsigned int x = 42u; return x; }")
        .expect_err("local declarations are outside the explicit lowering contract");

    assert_eq!(
        error.to_string(),
        "C frontend cannot lower a scalar kernel body other than one return statement at byte 26. Fix: use the supported scalar kernel subset or lower the construct before this frontend."
    );
}

#[test]
fn malformed_syntax_preserves_exact_location_diagnostic() {
    let error = parse_source("unsigned int kernel(void) { return (1u + ); }")
        .expect_err("missing operand must be rejected");

    assert_eq!(
        error.to_string(),
        "C frontend parse failed at byte 40 (line 1, column 41) near `<missing>`. Fix: provide a complete C translation unit."
    );
}

#[test]
fn hostile_bytes_preserve_exact_diagnostics() {
    let nul =
        parse_source_bytes(b"int kernel(void) {\0}").expect_err("embedded NUL must be rejected");
    assert_eq!(nul, CFrontendError::EmbeddedNul { offset: 18 },);
    assert_eq!(
        nul.to_string(),
        "C frontend rejected byte 0x00 at byte 18. Fix: remove embedded NUL bytes from C source."
    );

    let invalid_utf8 = parse_source_bytes(&[b'i', b'n', b't', b' ', 0xff])
        .expect_err("non-UTF-8 source must be rejected");
    assert_eq!(invalid_utf8, CFrontendError::InvalidUtf8 { offset: 4 });
    assert_eq!(
        invalid_utf8.to_string(),
        "C frontend source is not UTF-8 at byte 4. Fix: provide UTF-8 encoded C source."
    );
}

#[test]
fn source_size_boundary_accepts_limit_and_rejects_limit_plus_one() {
    let at_limit = vec![b' '; MAX_SOURCE_BYTES];
    parse_source_bytes(&at_limit).expect("exactly the documented limit is accepted");

    let over_limit = vec![b' '; MAX_SOURCE_BYTES + 1];
    let error = parse_source_bytes(&over_limit).expect_err("limit plus one must be rejected");
    assert_eq!(
        error,
        CFrontendError::SourceTooLarge {
            actual: MAX_SOURCE_BYTES + 1,
            max: MAX_SOURCE_BYTES,
        }
    );
}
