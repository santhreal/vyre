// Evasion cases for the `raw_ir_in_libs` lint: renames, aliases, cfg tricks,
// and module names that must not buy an exemption, plus allowlist and
// determinism behaviour.

use super::*;

#[test]
fn adversarial_module_named_tests_inside_a_real_module() {
    let v = lint_one(
        "nn/op.rs",
        r#"
    mod inner {
        fn build() {
            let _ = Node::let_bind("a", val());
        }
        mod tests {
            fn t() {
                let _ = Node::let_bind("b", val());
            }
        }
    }
    "#,
    );
    assert_eq!(
        v.len(),
        2,
        "module names alone cannot create test exemptions: {v:?}"
    );
}

/// Renaming `Node` at an import site must not hide raw IR construction.
#[test]
fn renamed_node_import_is_tracked() {
    let violations = lint_one(
        "nn/renamed.rs",
        "use vyre::ir::Node as N;\nfn build() { let _ = N::let_bind(\"x\", value()); }\n",
    );
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].kind, ViolationKind::RawNodeConstruction);
}

/// A local type alias must retain the underlying raw IR type identity.
#[test]
fn type_alias_for_expr_is_tracked() {
    let violations = lint_one(
        "math/alias.rs",
        "type E = Expr;\nfn build() { let _ = E::u32(7); }\n",
    );
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].kind, ViolationKind::RawExprConstruction);
}

/// A mixed `cfg(any(...))` remains production-reachable and cannot exempt raw IR.
#[test]
fn mixed_any_cfg_does_not_create_test_exemption() {
    let violations = lint_one(
    "nn/mixed_cfg.rs",
    "#[cfg(any(test, target_os = \"linux\"))]\nfn build() { let _ = Node::let_bind(\"x\", value()); }\n",
    );
    assert_eq!(violations.len(), 1);
}

/// A macro pattern that only reads an IR variant must remain permitted.
#[test]
fn macro_pattern_without_construction_is_not_flagged() {
    let violations = lint_one(
        "nn/pattern.rs",
        "macro_rules! classify { (Node::Store { .. }) => { 1 }; }\n",
    );
    assert!(
        violations.is_empty(),
        "macro patterns only read IR: {violations:?}"
    );
}

// ============== Allowlist behavior ==============

#[test]
fn allowlist_excludes_listed_files() {
    use vyre_lints::run_raw_ir_in_libs;
    let dir = tempfile::tempdir().unwrap();
    write_lib_file(
        dir.path(),
        "nn/exempt_op.rs",
        r#"
        fn build() {
            let _ = Node::let_bind("x", val());
        }
        "#,
    );
    write_lib_file(
        dir.path(),
        "nn/active_op.rs",
        r#"
        fn build() {
            let _ = Node::let_bind("y", val());
        }
        "#,
    );
    let allow_path = dir.path().join("allowlist.toml");
    std::fs::write(
        &allow_path,
        "exempt_files = [\"vyre-libs/src/nn/exempt_op.rs\"]\n",
    )
    .unwrap();
    let lib_src = dir.path().join("vyre-libs").join("src");
    let v = run_raw_ir_in_libs(&[lib_src.as_path()], Some(allow_path.as_path())).unwrap();
    assert_eq!(v.len(), 1);
    assert!(v[0].file.contains("active_op.rs"));
}

// ============== Idempotence / determinism ==============

#[test]
fn idempotent_two_runs_same_violations() {
    let dir = tempfile::tempdir().unwrap();
    write_lib_file(
        dir.path(),
        "nn/op.rs",
        r#"
        fn build() {
            let _ = Node::let_bind("x", val());
            let _ = Expr::add(a(), b());
        }
        "#,
    );
    let v1 = lint(dir.path());
    let v2 = lint(dir.path());
    assert_eq!(v1, v2);
}
