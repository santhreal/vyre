//! Structural gate for the canonical first-workgroup IR predicate.
//!
//! Production builders must call `Expr::is_first_workgroup()` instead of
//! rebuilding its expression tree. Centralizing the predicate keeps every
//! first-workgroup consumer aligned if the IR representation changes.

use std::fs;
use std::path::{Path, PathBuf};

use proc_macro2::LineColumn;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprStruct, Member};

const CANONICAL_BUILDER: &str = "vyre-foundation/src/ir_inner/model/expr.rs";

#[derive(Default)]
struct RawFirstWorkgroupVisitor {
    locations: Vec<LineColumn>,
}

impl<'ast> Visit<'ast> for RawFirstWorkgroupVisitor {
    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if is_raw_first_workgroup_call(call) {
            self.locations.push(call.span().start());
        }
        visit::visit_expr_call(self, call);
    }
}

fn is_raw_first_workgroup_call(call: &ExprCall) -> bool {
    let Expr::Path(callee) = call.func.as_ref() else {
        return false;
    };
    if callee
        .path
        .segments
        .last()
        .is_none_or(|segment| segment.ident != "eq")
    {
        return false;
    }
    let mut arguments = call.args.iter();
    let Some(left) = arguments.next() else {
        return false;
    };
    let Some(right) = arguments.next() else {
        return false;
    };
    if arguments.next().is_some() {
        return false;
    }

    (is_workgroup_x(left) && is_u32_zero(right)) || (is_u32_zero(left) && is_workgroup_x(right))
}

fn is_workgroup_x(expression: &Expr) -> bool {
    let Expr::Struct(ExprStruct { path, fields, .. }) = strip_parens(expression) else {
        return false;
    };
    if path
        .segments
        .last()
        .is_none_or(|segment| segment.ident != "WorkgroupId")
    {
        return false;
    }

    fields.iter().any(|field| {
        matches!(&field.member, Member::Named(name) if name == "axis")
            && is_integer_zero(&field.expr)
    })
}

fn is_u32_zero(expression: &Expr) -> bool {
    let Expr::Call(call) = strip_parens(expression) else {
        return false;
    };
    let Expr::Path(callee) = call.func.as_ref() else {
        return false;
    };
    callee
        .path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "u32")
        && call.args.len() == 1
        && call.args.first().is_some_and(is_integer_zero)
}

fn is_integer_zero(expression: &Expr) -> bool {
    matches!(strip_parens(expression), Expr::Lit(literal) if matches!(&literal.lit, syn::Lit::Int(value) if matches!(value.base10_parse::<u128>(), Ok(0))))
}

fn strip_parens(mut expression: &Expr) -> &Expr {
    loop {
        match expression {
            Expr::Paren(paren) => expression = &paren.expr,
            Expr::Group(group) => expression = &group.expr,
            _ => return expression,
        }
    }
}

fn raw_guard_locations(source: &str) -> Result<Vec<LineColumn>, syn::Error> {
    let file = syn::parse_file(source)?;
    let mut visitor = RawFirstWorkgroupVisitor::default();
    visitor.visit_file(&file);
    Ok(visitor.locations)
}

fn workspace_member_src_dirs(root: &Path) -> Vec<PathBuf> {
    let manifest_path = root.join("Cargo.toml");
    let manifest_text = fs::read_to_string(&manifest_path).unwrap_or_else(|error| {
        panic!(
            "Fix: canonical first-workgroup gate cannot read {}: {error}",
            manifest_path.display()
        )
    });
    let manifest: toml::Value = toml::from_str(&manifest_text).unwrap_or_else(|error| {
        panic!(
            "Fix: canonical first-workgroup gate cannot parse {}: {error}",
            manifest_path.display()
        )
    });
    manifest["workspace"]["members"]
        .as_array()
        .expect("Fix: workspace.members must remain an explicit array")
        .iter()
        .map(|member| {
            root.join(
                member
                    .as_str()
                    .expect("Fix: every workspace member must be a string"),
            )
            .join("src")
        })
        .filter(|path| path.is_dir())
        .collect()
}

/// Every production source must use the canonical first-workgroup builder.
///
/// This scans parsed Rust expressions rather than source substrings, so comments,
/// formatting, and inspection matches cannot hide or fabricate a violation.
#[test]
fn workspace_sources_reject_raw_first_workgroup_predicates() {
    let root = super::common::workspace_root();
    let canonical_path = root.join(CANONICAL_BUILDER);
    let mut violations = Vec::new();

    for src_dir in workspace_member_src_dirs(&root) {
        for entry in walkdir::WalkDir::new(src_dir) {
            let entry = entry.expect("Fix: every workspace source path must be readable");
            let path = entry.path();
            if !entry.file_type().is_file()
                || path.extension().is_none_or(|extension| extension != "rs")
                || path == canonical_path
            {
                continue;
            }
            let source = fs::read_to_string(path).unwrap_or_else(|error| {
                panic!(
                    "Fix: canonical first-workgroup gate cannot read {}: {error}",
                    path.display()
                )
            });
            let locations = raw_guard_locations(&source).unwrap_or_else(|error| {
                panic!(
                    "Fix: canonical first-workgroup gate cannot parse {}: {error}",
                    path.display()
                )
            });
            for location in locations {
                violations.push(format!(
                    "{}:{}:{}",
                    path.strip_prefix(&root).unwrap_or(path).display(),
                    location.line,
                    location.column + 1
                ));
            }
        }
    }

    assert_eq!(
        violations,
        Vec::<String>::new(),
        "replace raw `Expr::eq(Expr::WorkgroupId {{ axis: 0 }}, Expr::u32(0))` predicates with `Expr::is_first_workgroup()`"
    );
}

/// The detector must reject the exact direct form that escaped the prior migration.
///
/// This prevents the repository gate from becoming a clean-tree assertion that cannot
/// recognize the regression it claims to block.
#[test]
fn detector_rejects_direct_first_workgroup_expression() {
    let source = "fn guard() { let _ = Expr::eq(Expr::WorkgroupId { axis: 0 }, Expr::u32(0)); }";
    assert_eq!(raw_guard_locations(source).unwrap().len(), 1);
}

/// Operand reversal and harmless parentheses must not evade the structural gate.
///
/// Equality is commutative, so accepting the reversed spelling would leave a trivial
/// bypass for the same duplicated first-workgroup contract.
#[test]
fn detector_rejects_reversed_parenthesized_expression() {
    let source =
        "fn guard() { let _ = Expr::eq((Expr::u32(0)), (Expr::WorkgroupId { axis: 0 })); }";
    assert_eq!(raw_guard_locations(source).unwrap().len(), 1);
}

/// Canonical calls and non-x workgroup comparisons must remain valid.
///
/// The canonical builder is the intended production API. Comparisons on another axis
/// describe grid geometry rather than the single first-workgroup predicate.
#[test]
fn detector_accepts_canonical_and_non_x_workgroup_expressions() {
    let source = "fn guards() { let _ = Expr::is_first_workgroup(); let _ = Expr::eq(Expr::WorkgroupId { axis: 1 }, Expr::u32(0)); }";
    assert_eq!(raw_guard_locations(source).unwrap(), Vec::new());
}
