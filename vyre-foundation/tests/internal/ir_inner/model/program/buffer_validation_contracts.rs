use super::*;

#[test]
fn buffers_equal_ignoring_declaration_order_handles_permuted_buffers() {
    let buffers_a = [
        BufferDecl::output("out", 0, DataType::U32).with_count(1),
        BufferDecl::read("input", 1, DataType::U32).with_count(1),
    ];
    let buffers_b = [
        BufferDecl::read("input", 1, DataType::U32).with_count(1),
        BufferDecl::output("out", 0, DataType::U32).with_count(1),
    ];
    assert_ne!(buffers_a.as_slice(), buffers_b.as_slice());
    assert!(super::super::meta::buffers_equal_ignoring_declaration_order(&buffers_a, &buffers_b));
}

#[test]
fn validate_preserves_multiple_structured_issues() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [0, 0, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1)), Node::Return],
    );
    match program.validate() {
        Err(IrError::Validation { issues }) => {
            assert_eq!(issues.len(), 2);
            assert_eq!(issues[0].code().as_str(), "V106");
            assert_eq!(
                issues[0].location(),
                &crate::validate::ValidationLocation::WorkgroupAxis(0)
            );
            assert_eq!(issues[1].code().as_str(), "V106");
            assert_eq!(
                issues[1].location(),
                &crate::validate::ValidationLocation::WorkgroupAxis(1)
            );
        }
        other => panic!("expected structured validation issues, got {other:?}"),
    }
}

#[test]
fn validation_skip_cache_hits_on_repeated_validate_calls() {
    // Call validate() twice on the same Program; the second call must
    // return immediately (is_structurally_validated flips to true after
    // the first successful call).
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );

    assert!(
        !program.is_structurally_validated(),
        "fresh program must not be pre-validated"
    );

    program
        .validate()
        .expect("Fix: valid program must pass validation");
    assert!(
        program.is_structurally_validated(),
        "program must be marked validated after first validate()"
    );

    // Second call must hit the cache (returns Ok immediately).
    program
        .validate()
        .expect("Fix: repeated validate must return Ok via cache");
    assert!(program.is_structurally_validated());
}

#[test]
fn validation_skip_cache_clears_after_with_rewritten_entry() {
    // The cache must invalidate when the Program shape changes.
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );

    program
        .validate()
        .expect("Fix: valid program must pass validation");
    assert!(program.is_structurally_validated());

    // Rewrite the entry to a different shape.
    let rewritten =
        program.with_rewritten_entry(vec![Node::store("out", Expr::u32(0), Expr::u32(42))]);
    assert!(
        !rewritten.is_structurally_validated(),
        "with_rewritten_entry must clear the validation cache"
    );
}

#[test]
fn mark_validated_on_distinguishes_backends() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );

    program.mark_validated_on("backend-a");
    assert!(
        program.is_validated_on("backend-a"),
        "must be validated for backend-a after mark"
    );
    assert!(
        !program.is_validated_on("backend-b"),
        "mark_validated_on(\"backend-a\") must not satisfy is_validated_on(\"backend-b\")"
    );
}

#[test]
fn with_rewritten_entry_preserves_buffer_arc_identity() {
    let buffers: Vec<BufferDecl> = (0..20)
        .map(|i| BufferDecl::output(&format!("buf_{i}"), i, DataType::U32).with_count(1))
        .collect();
    let program = Program::wrapped(buffers, [64, 1, 1], vec![Node::Return]);
    let rewritten = program.with_rewritten_entry(vec![Node::let_bind("x", Expr::u32(42))]);

    assert!(
        Arc::ptr_eq(program.buffers_arc(), rewritten.buffers_arc()),
        "Fix: with_rewritten_entry must preserve the same Arc<[BufferDecl]> without deep cloning."
    );
}
