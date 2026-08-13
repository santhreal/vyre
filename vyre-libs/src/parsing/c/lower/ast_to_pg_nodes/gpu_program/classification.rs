use super::*;

use crate::parsing::c::lower::ast_to_pg_nodes::semantic::{
    CONTROL_KINDS, DECLARATION_KINDS, EXPRESSION_KINDS, GNU_KINDS, ROLE_BY_KIND,
};

pub(super) fn expr_is_kind(kind: Expr, expected: u32) -> Expr {
    Expr::eq(kind, Expr::u32(expected))
}

fn kind_is(expected: u32) -> Expr {
    expr_is_kind(Expr::var("kind"), expected)
}

/// `kind` is one of `kinds`, as a balanced `or` tree.
///
/// Balanced rather than folded because the wire format rejects `Expr` nesting
/// deeper than 64, and the largest of these sets alone has forty members.
fn kind_in_set(kinds: &[u32]) -> Expr {
    match kinds {
        [] => Expr::bool(false),
        [only] => kind_is(*only),
        _ => {
            let (left, right) = kinds.split_at(kinds.len() / 2);
            Expr::or(kind_in_set(left), kind_in_set(right))
        }
    }
}

/// Category lookup, folded from the same kind sets the CPU oracle keys on.
///
/// Fold order gives the last group the outermost `select`, so precedence runs
/// GNU, declaration, expression, control, none, matching
/// `reference::semantic_category`'s `if`/`else if` chain. A kind may appear in
/// more than one set, so this stays an ordered chain rather than a sum.
///
/// Deriving the chain from those sets is what keeps the two from drifting: this
/// used to encode the sets as literal kind ranges, and
/// `C_AST_KIND_GNU_LOCAL_LABEL_DECL` fell outside every GPU range while
/// remaining in `GNU_KINDS`, so GNU `__label__` declarations lowered with
/// category `NONE` on the GPU and `GNU` on the CPU.
pub(super) fn category_lookup_expr() -> Expr {
    [
        (CONTROL_KINDS, C_AST_PG_CATEGORY_CONTROL),
        (EXPRESSION_KINDS, C_AST_PG_CATEGORY_EXPRESSION),
        (DECLARATION_KINDS, C_AST_PG_CATEGORY_DECLARATION),
        (GNU_KINDS, C_AST_PG_CATEGORY_GNU),
    ]
    .into_iter()
    .fold(
        Expr::u32(C_AST_PG_CATEGORY_NONE),
        |category, (kinds, value)| Expr::select(kind_in_set(kinds), Expr::u32(value), category),
    )
}

/// Role lookup, keyed on the same table the CPU oracle keys on.
///
/// Each row contributes its role or `C_AST_PG_ROLE_NONE`, which is zero, and
/// the rows are combined with a balanced `bitor` tree. That is exact because
/// `first_role_per_kind` leaves one row per kind, so at most one row can
/// contribute. An ordered `select` chain would be ninety deep and the wire
/// format caps `Expr` nesting at 64.
pub(super) fn role_lookup_expr() -> Expr {
    role_lookup(&first_role_per_kind())
}

/// `ROLE_BY_KIND` reduced to its first row per kind, which is the row
/// `semantic::semantic_role`'s `find_map` would select.
fn first_role_per_kind() -> Vec<(u32, u32)> {
    let mut rows: Vec<(u32, u32)> = Vec::with_capacity(ROLE_BY_KIND.len());
    for &(kind, role) in ROLE_BY_KIND {
        if !rows.iter().any(|&(seen, _)| seen == kind) {
            rows.push((kind, role));
        }
    }
    rows
}

fn role_lookup(rows: &[(u32, u32)]) -> Expr {
    match rows {
        [] => Expr::u32(C_AST_PG_ROLE_NONE),
        [(kind, role)] => Expr::select(
            kind_is(*kind),
            Expr::u32(*role),
            Expr::u32(C_AST_PG_ROLE_NONE),
        ),
        _ => {
            let (left, right) = rows.split_at(rows.len() / 2);
            Expr::bitor(role_lookup(left), role_lookup(right))
        }
    }
}

pub(super) fn semantic_classification_nodes() -> Vec<Node> {
    vec![
        Node::let_bind("semantic_category", category_lookup_expr()),
        Node::let_bind("semantic_role", role_lookup_expr()),
        Node::if_then(
            Expr::and(
                expr_is_kind(Expr::var("kind"), C_AST_KIND_POINTER_DECL),
                Expr::or(
                    expr_is_kind(Expr::var("parent_kind"), C_AST_KIND_FUNCTION_DECLARATOR),
                    Expr::or(
                        expr_is_kind(
                            Expr::var("first_child_kind"),
                            C_AST_KIND_FUNCTION_DECLARATOR,
                        ),
                        expr_is_kind(
                            Expr::var("next_sibling_kind"),
                            C_AST_KIND_FUNCTION_DECLARATOR,
                        ),
                    ),
                ),
            ),
            vec![Node::assign(
                "semantic_role",
                Expr::u32(C_AST_PG_ROLE_FUNCTION_POINTER_DECL),
            )],
        ),
    ]
}
