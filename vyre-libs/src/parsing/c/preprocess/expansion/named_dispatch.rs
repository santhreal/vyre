//! The named-macro expansion shell: the invocation scan prefix, the
//! four-leaf dispatch tree over one input token, and the serial driver loop
//! that carries the cursor.

use crate::parsing::c::lex::tokens::{TOK_IDENTIFIER, TOK_LPAREN};
use crate::parsing::c::preprocess::materialization::C_MACRO_SOURCE_COUNT_BYTES;
use crate::region::wrap_anonymous;
use vyre_foundation::ir::{Expr, Node};

use super::macro_lookup::*;
use super::*;

pub(super) struct NamedMacroScanSpec<'a> {
    pub(super) in_tok_types: &'a str,
    pub(super) in_tok_starts: &'a str,
    pub(super) in_tok_lens: &'a str,
    pub(super) source_words: &'a str,
    pub(super) source_layout: MacroByteLayout,
    pub(super) macro_name_hashes: &'a str,
    pub(super) macro_name_starts: &'a str,
    pub(super) macro_name_lens: &'a str,
    pub(super) macro_name_words: &'a str,
    pub(super) macro_name_layout: MacroByteLayout,
    pub(super) macro_vals: &'a str,
    pub(super) macro_kinds: &'a str,
    pub(super) macro_param_counts: &'a str,
    pub(super) source_len: Expr,
    pub(super) decode_variadic_param_count: bool,
}

pub(super) fn emit_named_macro_scan_prefix(spec: NamedMacroScanSpec<'_>) -> Vec<Node> {
    let mut process_current = vec![
        Node::let_bind(
            "named_tok",
            Expr::load(spec.in_tok_types, Expr::var("named_i")),
        ),
        Node::let_bind("named_macro_slot", Expr::u32(EMPTY_MACRO_SLOT)),
        Node::let_bind("named_macro_idx", Expr::u32(EMPTY_MACRO_SLOT)),
        Node::let_bind("named_macro_kind", Expr::u32(C_MACRO_KIND_OBJECT_LIKE)),
        Node::let_bind("named_param_count", Expr::u32(0)),
        Node::let_bind("named_is_variadic", Expr::u32(0)),
        Node::let_bind("named_required_param_count", Expr::u32(0)),
    ];

    process_current.push(Node::if_then(
        Expr::eq(Expr::var("named_tok"), Expr::u32(TOK_IDENTIFIER)),
        {
            let mut ident = emit_source_span_hash(
                "named",
                Expr::var("named_i"),
                spec.in_tok_starts,
                spec.in_tok_lens,
                spec.source_words,
                spec.source_layout,
                spec.source_len,
                "named_name_hash",
            );
            ident.extend(emit_macro_hash_lookup(
                "named_lookup",
                Expr::var("named_name_hash"),
                Expr::var("named_start"),
                Expr::var("named_len"),
                spec.source_words,
                spec.source_layout,
                spec.macro_name_hashes,
                spec.macro_name_starts,
                spec.macro_name_lens,
                spec.macro_name_words,
                spec.macro_name_layout,
                "named_macro_slot",
            ));
            ident
        },
    ));

    let mut found_macro = vec![
        Node::assign(
            "named_macro_idx",
            Expr::load(spec.macro_vals, Expr::var("named_macro_slot")),
        ),
        Node::assign(
            "named_macro_kind",
            Expr::load(spec.macro_kinds, Expr::var("named_macro_slot")),
        ),
    ];
    if spec.decode_variadic_param_count {
        found_macro.extend([
            Node::let_bind(
                "named_param_count_raw",
                Expr::load(spec.macro_param_counts, Expr::var("named_macro_slot")),
            ),
            Node::assign(
                "named_param_count",
                Expr::bitand(Expr::var("named_param_count_raw"), Expr::u32(0x7fff_ffff)),
            ),
            Node::assign(
                "named_is_variadic",
                Expr::shr(Expr::var("named_param_count_raw"), Expr::u32(31)),
            ),
            Node::assign(
                "named_required_param_count",
                Expr::saturating_sub(
                    Expr::var("named_param_count"),
                    Expr::var("named_is_variadic"),
                ),
            ),
        ]);
    } else {
        found_macro.extend([
            Node::assign(
                "named_param_count",
                Expr::load(spec.macro_param_counts, Expr::var("named_macro_slot")),
            ),
            Node::assign("named_required_param_count", Expr::var("named_param_count")),
        ]);
    }
    found_macro.push(Node::if_then(
        Expr::and(
            Expr::ne(
                Expr::var("named_macro_kind"),
                Expr::u32(C_MACRO_KIND_OBJECT_LIKE),
            ),
            Expr::ne(
                Expr::var("named_macro_kind"),
                Expr::u32(C_MACRO_KIND_FUNCTION_LIKE),
            ),
        ),
        vec![Node::trap(
            Expr::var("named_macro_kind"),
            "named-macro-kind-invalid",
        )],
    ));
    process_current.push(Node::if_then(
        Expr::ne(Expr::var("named_macro_slot"), Expr::u32(EMPTY_MACRO_SLOT)),
        found_macro,
    ));
    process_current
}

pub(super) fn emit_named_replacement_prelude(
    macro_sizes: &str,
    in_tok_types: &str,
    num_tokens: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind(
            "named_repl_size",
            Expr::load(macro_sizes, Expr::var("named_macro_idx")),
        ),
        Node::if_then(
            Expr::gt(
                Expr::add(Expr::var("named_macro_idx"), Expr::var("named_repl_size")),
                Expr::u32(MACRO_TABLE_SLOTS),
            ),
            vec![Node::trap(
                Expr::add(Expr::var("named_macro_idx"), Expr::var("named_repl_size")),
                "named-macro-replacement-range-out-of-bounds",
            )],
        ),
        Node::let_bind("named_has_open_paren", Expr::u32(0)),
        Node::if_then(
            Expr::lt(
                Expr::add(Expr::var("named_i"), Expr::u32(1)),
                num_tokens.clone(),
            ),
            vec![Node::if_then(
                Expr::eq(
                    Expr::load(in_tok_types, Expr::add(Expr::var("named_i"), Expr::u32(1))),
                    Expr::u32(TOK_LPAREN),
                ),
                vec![Node::assign("named_has_open_paren", Expr::u32(1))],
            )],
        ),
    ]
}

/// Branch bodies for the named-macro dispatch tree.
///
/// The token-only and materialized expansion kernels agree on the shape of the
/// dispatch: resolve the macro slot, pass the token through when it names no
/// macro, otherwise split object-like from function-like and pass a
/// function-like name through when no `(` follows. They disagree only on what
/// each of those four leaves emits.
pub(super) struct NamedMacroDispatchSpec<'a> {
    /// Prefix that resolves `named_macro_slot` for the current token.
    pub(super) scan: NamedMacroScanSpec<'a>,
    /// Replacement-size prelude input.
    pub(super) macro_sizes: &'a str,
    /// Token count driving the prelude's bounded loops.
    pub(super) num_tokens: Expr,
    /// Emitted when the token names no macro.
    pub(super) unknown_passthrough: Vec<Node>,
    /// Emitted for an object-like macro.
    pub(super) object_like: Vec<Node>,
    /// Emitted when a function-like macro name is not followed by `(`.
    pub(super) function_name_passthrough: Vec<Node>,
    /// Emitted for an invoked function-like macro.
    pub(super) function_like: Vec<Node>,
}

/// Build the named-macro dispatch tree for one input token.
///
/// Both passthrough leaves advance `named_i` by one; every other leaf owns its
/// own cursor update.
pub(super) fn emit_named_macro_dispatch(spec: NamedMacroDispatchSpec<'_>) -> Vec<Node> {
    let in_tok_types = spec.scan.in_tok_types;
    let mut nodes = emit_named_macro_scan_prefix(spec.scan);
    nodes.push(Node::if_then_else(
        Expr::eq(Expr::var("named_macro_slot"), Expr::u32(EMPTY_MACRO_SLOT)),
        advance_named_cursor(spec.unknown_passthrough),
        {
            let mut expanded =
                emit_named_replacement_prelude(spec.macro_sizes, in_tok_types, spec.num_tokens);
            expanded.push(Node::if_then_else(
                Expr::eq(
                    Expr::var("named_macro_kind"),
                    Expr::u32(C_MACRO_KIND_OBJECT_LIKE),
                ),
                spec.object_like,
                vec![Node::if_then_else(
                    Expr::eq(Expr::var("named_has_open_paren"), Expr::u32(0)),
                    advance_named_cursor(spec.function_name_passthrough),
                    spec.function_like,
                )],
            ));
            expanded
        },
    ));
    nodes
}

fn advance_named_cursor(mut passthrough: Vec<Node>) -> Vec<Node> {
    passthrough.push(Node::assign(
        "named_i",
        Expr::add(Expr::var("named_i"), Expr::u32(1)),
    ));
    passthrough
}

/// Serial driver shell around a named-macro dispatch body.
pub(super) struct NamedExpansionDriverSpec<'a> {
    /// Region name and entry op id of the owning kernel.
    pub(super) op_id: &'static str,
    /// Per-token dispatch body.
    pub(super) body: Vec<Node>,
    /// Upper bound of the cursor loop.
    pub(super) num_tokens: Expr,
    /// Buffer receiving the final output token count.
    pub(super) out_tok_counts: &'a str,
    /// Buffer receiving the final output byte count, for the materialized
    /// kernel that also emits a source arena. `None` keeps the token-only
    /// kernel free of a byte cursor.
    pub(super) out_source_counts: Option<&'a str>,
}

/// Wrap a dispatch body in the single-invocation serial cursor loop.
///
/// Only invocation 0 runs: the expansion walk is inherently sequential because
/// `named_i` can jump past a whole macro invocation.
pub(super) fn emit_named_expansion_driver(spec: NamedExpansionDriverSpec<'_>) -> Node {
    let mut serial = vec![
        Node::let_bind("named_i", Expr::u32(0)),
        Node::let_bind("named_out_idx", Expr::u32(0)),
    ];
    if spec.out_source_counts.is_some() {
        serial.push(Node::let_bind("named_source_out_idx", Expr::u32(0)));
    }
    serial.push(Node::loop_for(
        "named_cursor",
        Expr::u32(0),
        spec.num_tokens,
        vec![Node::if_then(
            Expr::eq(Expr::var("named_cursor"), Expr::var("named_i")),
            spec.body,
        )],
    ));
    serial.push(Node::store(
        spec.out_tok_counts,
        Expr::u32(0),
        Expr::var("named_out_idx"),
    ));
    if let Some(out_source_counts) = spec.out_source_counts {
        serial.push(Node::store(
            out_source_counts,
            Expr::u32(C_MACRO_SOURCE_COUNT_BYTES),
            Expr::var("named_source_out_idx"),
        ));
    }
    wrap_anonymous(
        spec.op_id,
        vec![Node::if_then(
            Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
            serial,
        )],
    )
}
