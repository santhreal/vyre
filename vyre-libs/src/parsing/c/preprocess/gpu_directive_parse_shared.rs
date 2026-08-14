use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub(super) use super::gpu_source_bytes::source_buffer_element;
pub(super) use super::gpu_source_bytes::SourceByteLayout as DirectiveSourceLayout;

pub(super) const DIRECTIVE_PARSE_WORKGROUP_SIZE: u32 = 256;
pub(super) const MAX_DIRECTIVE_WS_PREFIX: u32 = 4;

pub(super) fn packed_source_word_count(source_len: u32) -> u32 {
    source_len.div_ceil(4).max(1)
}

/// One per-token U32 column, sized to the token count.
pub(super) fn token_column(
    name: &str,
    binding: u32,
    access: BufferAccess,
    num_tokens: u32,
) -> BufferDecl {
    BufferDecl::storage(name, binding, access, DataType::U32).with_count(num_tokens.max(1))
}

/// One runtime-bound read-only column.
///
/// `count = 0` marks the buffer runtime-sized, so `Expr::buf_len` reports the
/// host-supplied element count and the program's structure stays independent of
/// how large a table the host packs.
pub(super) fn runtime_sized_input(name: &str, binding: u32, element: DataType) -> BufferDecl {
    BufferDecl::storage(name, binding, BufferAccess::ReadOnly, element).with_count(0)
}

pub(super) fn directive_parse_input_buffers(
    num_tokens: u32,
    source_len: u32,
    source_layout: DirectiveSourceLayout,
) -> Vec<BufferDecl> {
    let source_element = source_buffer_element(source_layout);
    let source = match source_layout {
        DirectiveSourceLayout::PackedU32 => {
            BufferDecl::storage("source", 3, BufferAccess::ReadOnly, source_element)
                .with_count(packed_source_word_count(source_len))
        }
        DirectiveSourceLayout::RawU8 => runtime_sized_input("source", 3, source_element),
    };
    vec![
        token_column("tok_starts", 0, BufferAccess::ReadOnly, num_tokens),
        token_column("tok_lens", 1, BufferAccess::ReadOnly, num_tokens),
        token_column("directive_kinds", 2, BufferAccess::ReadOnly, num_tokens),
        source,
    ]
}

#[derive(Clone, Copy)]
pub(super) struct DirectiveOutputColumn {
    pub(super) name: &'static str,
    pub(super) binding: u32,
}

#[derive(Clone, Copy)]
pub(super) enum DirectiveThreadLayout {
    InvocationId,
    WorkgroupLinear,
}

pub(super) fn directive_output_buffers(
    num_tokens: u32,
    columns: &[DirectiveOutputColumn],
) -> Vec<BufferDecl> {
    columns
        .iter()
        .map(|column| {
            token_column(
                column.name,
                column.binding,
                BufferAccess::ReadWrite,
                num_tokens,
            )
        })
        .collect()
}

pub(super) fn directive_parse_body(
    layout: DirectiveThreadLayout,
    output_columns: &[DirectiveOutputColumn],
    kind_guard: Expr,
    parse: Vec<Node>,
) -> Vec<Node> {
    let t = Expr::var("t");
    let mut body = match layout {
        DirectiveThreadLayout::InvocationId => {
            vec![Node::let_bind("t", Expr::InvocationId { axis: 0 })]
        }
        DirectiveThreadLayout::WorkgroupLinear => vec![
            Node::let_bind("lane", Expr::LocalId { axis: 0 }),
            Node::let_bind("block", Expr::WorkgroupId { axis: 0 }),
            Node::let_bind(
                "t",
                Expr::add(
                    Expr::mul(
                        Expr::var("block"),
                        Expr::u32(DIRECTIVE_PARSE_WORKGROUP_SIZE),
                    ),
                    Expr::var("lane"),
                ),
            ),
        ],
    };

    let mut guarded = vec![Node::let_bind(
        "kind",
        Expr::load("directive_kinds", t.clone()),
    )];
    guarded.extend(
        output_columns
            .iter()
            .map(|column| Node::store(column.name, t.clone(), Expr::u32(0))),
    );
    guarded.push(Node::if_then(kind_guard, parse));
    body.push(Node::if_then(
        Expr::lt(t, Expr::buf_len("tok_starts")),
        guarded,
    ));
    body
}

pub(super) fn directive_parse_program(
    op_id: &'static str,
    buffers: Vec<BufferDecl>,
    body: Vec<Node>,
) -> Program {
    Program::wrapped(buffers, [DIRECTIVE_PARSE_WORKGROUP_SIZE, 1, 1], body).with_entry_op_id(op_id)
}

pub(super) fn directive_program_from_parse(
    op_id: &'static str,
    num_tokens: u32,
    source_len: u32,
    output_columns: &[DirectiveOutputColumn],
    layout: DirectiveThreadLayout,
    kind_guard: Expr,
    parse: Vec<Node>,
) -> Program {
    directive_program_from_parse_with_source_layout(
        op_id,
        num_tokens,
        source_len,
        DirectiveSourceLayout::PackedU32,
        output_columns,
        layout,
        kind_guard,
        parse,
    )
}

pub(super) fn directive_program_from_parse_with_source_layout(
    op_id: &'static str,
    num_tokens: u32,
    source_len: u32,
    source_layout: DirectiveSourceLayout,
    output_columns: &[DirectiveOutputColumn],
    layout: DirectiveThreadLayout,
    kind_guard: Expr,
    parse: Vec<Node>,
) -> Program {
    let mut buffers = directive_parse_input_buffers(num_tokens, source_len, source_layout);
    buffers.extend(directive_output_buffers(num_tokens, output_columns));
    let body = directive_parse_body(layout, output_columns, kind_guard, parse);
    directive_parse_program(op_id, buffers, body)
}

pub(super) fn push_directive_row_bounds(parse: &mut Vec<Node>) {
    let t = Expr::var("t");
    parse.push(Node::let_bind(
        "tok_start",
        Expr::load("tok_starts", t.clone()),
    ));
    parse.push(Node::let_bind("tok_len", Expr::load("tok_lens", t)));
    parse.push(Node::let_bind(
        "tok_end",
        Expr::add(Expr::var("tok_start"), Expr::var("tok_len")),
    ));
}

pub(super) fn push_ws_skip_from_expr(
    parse: &mut Vec<Node>,
    source_layout: DirectiveSourceLayout,
    prefix: &str,
    base: Expr,
    skip_name: &'static str,
    target_name: &'static str,
) {
    for q in 0..MAX_DIRECTIVE_WS_PREFIX {
        parse.push(Node::let_bind(
            format!("{prefix}_{q}"),
            safe_source_byte_expr(source_layout, Expr::add(base.clone(), Expr::u32(q))),
        ));
    }
    for q in 0..MAX_DIRECTIVE_WS_PREFIX {
        parse.push(Node::let_bind(
            format!("{prefix}_ws_{q}"),
            horizontal_ws_flag(Expr::var(format!("{prefix}_{q}"))),
        ));
    }
    parse.push(Node::let_bind(
        skip_name,
        ws_skip_expr(prefix, MAX_DIRECTIVE_WS_PREFIX),
    ));
    parse.push(Node::let_bind(
        target_name,
        Expr::add(base, Expr::var(skip_name)),
    ));
}

/// Leading horizontal-whitespace run, `#`, and the byte index of that `#`.
///
/// `prefix` names the byte and whitespace-flag bindings this stage leaves in
/// scope. Consumers bind them in their own namespace: the classifier reads
/// `s_*`, the payload evaluators read `hs_*`. The scan itself is the same
/// straight-line chain either way, capped at [`MAX_DIRECTIVE_WS_PREFIX`]
/// leading whitespace bytes.
pub(super) fn push_hash_scan(
    parse: &mut Vec<Node>,
    source_layout: DirectiveSourceLayout,
    prefix: &str,
) {
    for p in 0..=MAX_DIRECTIVE_WS_PREFIX {
        parse.push(Node::let_bind(
            format!("{prefix}_{p}"),
            safe_source_byte_expr(
                source_layout,
                Expr::add(Expr::var("tok_start"), Expr::u32(p)),
            ),
        ));
    }
    for p in 0..=MAX_DIRECTIVE_WS_PREFIX {
        parse.push(Node::let_bind(
            format!("{prefix}_ws_{p}"),
            horizontal_ws_flag(Expr::var(format!("{prefix}_{p}"))),
        ));
    }
    parse.push(Node::let_bind(
        "hash_off",
        hash_offset_expr(prefix, MAX_DIRECTIVE_WS_PREFIX),
    ));
    parse.push(Node::let_bind(
        "hash_idx",
        Expr::add(Expr::var("tok_start"), Expr::var("hash_off")),
    ));
}

/// Whether [`push_hash_scan`] found a `#` inside the leading-whitespace cap.
///
/// Consumers place this where their own let-chain wants it, which is why it is
/// a stage rather than part of [`push_hash_scan`].
pub(super) fn push_found_hash(parse: &mut Vec<Node>) {
    parse.push(Node::let_bind(
        "found_hash",
        Expr::select(
            Expr::lt(
                Expr::var("hash_off"),
                Expr::u32(MAX_DIRECTIVE_WS_PREFIX + 1),
            ),
            Expr::u32(1),
            Expr::u32(0),
        ),
    ));
}

/// Whitespace run between `#` and the directive keyword, leaving `kw_start`
/// at the first keyword byte.
pub(super) fn push_keyword_start(
    parse: &mut Vec<Node>,
    source_layout: DirectiveSourceLayout,
    prefix: &str,
) {
    push_ws_skip_from_expr(
        parse,
        source_layout,
        prefix,
        Expr::add(Expr::var("hash_idx"), Expr::u32(1)),
        "kw_skip",
        "kw_start",
    );
}

/// The payload builders' composition of the scan: hash, found-hash flag, then
/// keyword start.
pub(super) fn push_hash_and_keyword_start(
    parse: &mut Vec<Node>,
    source_layout: DirectiveSourceLayout,
) {
    push_hash_scan(parse, source_layout, "hs");
    push_found_hash(parse);
    push_keyword_start(parse, source_layout, "kp");
}

pub(super) fn push_keyword_end(parse: &mut Vec<Node>, keyword_len: Expr) {
    parse.push(Node::let_bind(
        "post_kw",
        Expr::add(Expr::var("kw_start"), keyword_len),
    ));
}

/// Keyword bytes `k_0..=k_count` plus their ident-continue flags. The extra
/// trailing byte is the sentinel that keeps `define` from matching `defined`.
pub(super) fn push_keyword_bytes(
    parse: &mut Vec<Node>,
    source_layout: DirectiveSourceLayout,
    count: u32,
) {
    for i in 0..=count {
        parse.push(Node::let_bind(
            format!("k_{i}"),
            safe_source_byte_expr(
                source_layout,
                Expr::add(Expr::var("kw_start"), Expr::u32(i)),
            ),
        ));
    }
    for i in 0..=count {
        parse.push(Node::let_bind(
            format!("k_is_continue_{i}"),
            c_ident_continue_flag(Expr::var(format!("k_{i}"))),
        ));
    }
}

/// `1` when the bytes [`push_keyword_bytes`] read spell `expected` and the
/// next byte is not an ident-continue byte, else `0`.
pub(super) fn keyword_match_expr(expected: &[u32]) -> Expr {
    let mut all_eq = Expr::u32(1);
    for (i, byte) in expected.iter().copied().enumerate() {
        let eq_byte = Expr::select(
            Expr::eq(Expr::var(format!("k_{i}")), Expr::u32(byte)),
            Expr::u32(1),
            Expr::u32(0),
        );
        all_eq = Expr::bitand(all_eq, eq_byte);
    }
    let next_not_ident = Expr::select(
        Expr::eq(
            Expr::var(format!("k_is_continue_{}", expected.len())),
            Expr::u32(0),
        ),
        Expr::u32(1),
        Expr::u32(0),
    );
    Expr::bitand(all_eq, next_not_ident)
}

pub(super) fn push_c_identifier_span(
    parse: &mut Vec<Node>,
    source_layout: DirectiveSourceLayout,
    start_var: &'static str,
    len_var: &'static str,
    done_var: &'static str,
) {
    let scan_limit = format!("{len_var}_scan_limit");
    let iter = format!("{len_var}_i");
    let byte = format!("{len_var}_byte");
    let byte_ok = format!("{len_var}_byte_ok");

    parse.push(Node::let_bind(
        scan_limit.clone(),
        Expr::select(
            Expr::lt(Expr::var(start_var), Expr::var("tok_end")),
            Expr::sub(Expr::var("tok_end"), Expr::var(start_var)),
            Expr::u32(0),
        ),
    ));
    parse.push(Node::let_bind(len_var, Expr::u32(0)));
    parse.push(Node::let_bind(done_var, Expr::u32(0)));
    parse.push(Node::loop_for(
        iter.clone(),
        Expr::u32(0),
        Expr::var(scan_limit),
        vec![Node::if_then(
            Expr::eq(Expr::var(done_var), Expr::u32(0)),
            vec![
                Node::let_bind(
                    byte.clone(),
                    safe_source_byte_expr(
                        source_layout,
                        Expr::add(Expr::var(start_var), Expr::var(iter.clone())),
                    ),
                ),
                Node::let_bind(
                    byte_ok.clone(),
                    Expr::select(
                        Expr::eq(Expr::var(iter.clone()), Expr::u32(0)),
                        c_ident_start_flag(Expr::var(byte.clone())),
                        c_ident_continue_flag(Expr::var(byte)),
                    ),
                ),
                Node::if_then_else(
                    Expr::eq(Expr::var(byte_ok), Expr::u32(1)),
                    vec![Node::assign(
                        len_var,
                        Expr::add(Expr::var(iter), Expr::u32(1)),
                    )],
                    vec![Node::assign(done_var, Expr::u32(1))],
                ),
            ],
        )],
    ));
}

pub(super) fn push_bounded_byte_scan_until(
    parse: &mut Vec<Node>,
    source_layout: DirectiveSourceLayout,
    iter_var: &'static str,
    start_var: &'static str,
    limit_var: &'static str,
    byte_var: &'static str,
    len_var: &'static str,
    done_var: &'static str,
    close_byte: Expr,
    active_guard: Expr,
) {
    parse.push(Node::let_bind(
        limit_var,
        Expr::select(
            Expr::lt(Expr::var(start_var), Expr::var("tok_end")),
            Expr::sub(Expr::var("tok_end"), Expr::var(start_var)),
            Expr::u32(0),
        ),
    ));
    parse.push(Node::let_bind(len_var, Expr::u32(0)));
    parse.push(Node::let_bind(done_var, Expr::u32(0)));
    parse.push(Node::loop_for(
        iter_var,
        Expr::u32(0),
        Expr::var(limit_var),
        vec![Node::if_then(
            Expr::and(active_guard, Expr::eq(Expr::var(done_var), Expr::u32(0))),
            vec![
                Node::let_bind(
                    byte_var,
                    safe_source_byte_expr(
                        source_layout,
                        Expr::add(Expr::var(start_var), Expr::var(iter_var)),
                    ),
                ),
                Node::if_then(
                    Expr::eq(Expr::var(byte_var), close_byte),
                    vec![
                        Node::assign(len_var, Expr::var(iter_var)),
                        Node::assign(done_var, Expr::u32(1)),
                    ],
                ),
            ],
        )],
    ));
}

pub(super) fn source_byte_expr(source_layout: DirectiveSourceLayout, addr: Expr) -> Expr {
    super::gpu_source_bytes::load_source_layout_byte_expr("source", source_layout, addr)
}

pub(super) fn safe_source_byte_expr(source_layout: DirectiveSourceLayout, addr: Expr) -> Expr {
    super::gpu_source_bytes::safe_load_source_layout_byte_expr(
        "source",
        source_layout,
        addr,
        super::gpu_source_bytes::source_byte_len_expr("source", source_layout),
    )
}

fn whitespace_flag(byte: Expr, include_line_endings: bool) -> Expr {
    let horizontal = Expr::or(
        Expr::or(
            Expr::eq(byte.clone(), Expr::u32(b' ' as u32)),
            Expr::eq(byte.clone(), Expr::u32(b'\t' as u32)),
        ),
        Expr::or(
            Expr::eq(byte.clone(), Expr::u32(0x0B)),
            Expr::eq(byte.clone(), Expr::u32(0x0C)),
        ),
    );
    let condition = if include_line_endings {
        Expr::or(
            horizontal,
            Expr::or(
                Expr::eq(byte.clone(), Expr::u32(b'\n' as u32)),
                Expr::eq(byte, Expr::u32(b'\r' as u32)),
            ),
        )
    } else {
        horizontal
    };
    Expr::select(condition, Expr::u32(1), Expr::u32(0))
}

pub(super) fn horizontal_ws_flag(byte: Expr) -> Expr {
    whitespace_flag(byte, false)
}

pub(super) fn trailing_ws_flag(byte: Expr) -> Expr {
    whitespace_flag(byte, true)
}

fn c_ident_start_condition(byte: &Expr) -> Expr {
    let is_lower = Expr::and(
        Expr::ge(byte.clone(), Expr::u32(b'a' as u32)),
        Expr::le(byte.clone(), Expr::u32(b'z' as u32)),
    );
    let is_upper = Expr::and(
        Expr::ge(byte.clone(), Expr::u32(b'A' as u32)),
        Expr::le(byte.clone(), Expr::u32(b'Z' as u32)),
    );
    let is_under = Expr::eq(byte.clone(), Expr::u32(b'_' as u32));
    Expr::or(Expr::or(is_lower, is_upper), is_under)
}

pub(super) fn c_ident_continue_flag(byte: Expr) -> Expr {
    let is_digit = Expr::and(
        Expr::ge(byte.clone(), Expr::u32(b'0' as u32)),
        Expr::le(byte.clone(), Expr::u32(b'9' as u32)),
    );
    Expr::select(
        Expr::or(c_ident_start_condition(&byte), is_digit),
        Expr::u32(1),
        Expr::u32(0),
    )
}

pub(super) fn c_ident_start_flag(byte: Expr) -> Expr {
    Expr::select(c_ident_start_condition(&byte), Expr::u32(1), Expr::u32(0))
}

pub(super) fn hash_offset_expr(prefix: &str, max_ws_prefix: u32) -> Expr {
    let mut acc = Expr::u32(0xFFFF_FFFF);
    for p in (0..=max_ws_prefix).rev() {
        let mut prefix_ws = Expr::u32(1);
        for q in 0..p {
            prefix_ws = Expr::bitand(prefix_ws, Expr::var(format!("{prefix}_ws_{q}")));
        }
        let byte_is_hash = Expr::select(
            Expr::eq(Expr::var(format!("{prefix}_{p}")), Expr::u32(b'#' as u32)),
            Expr::u32(1),
            Expr::u32(0),
        );
        let cond_u32 = Expr::bitand(byte_is_hash, prefix_ws);
        acc = Expr::select(Expr::eq(cond_u32, Expr::u32(1)), Expr::u32(p), acc);
    }
    acc
}

pub(super) fn ws_skip_expr(prefix: &str, width: u32) -> Expr {
    let mut acc = Expr::u32(width);
    for q in (0..width).rev() {
        let mut prefix_ws = Expr::u32(1);
        for r in 0..q {
            prefix_ws = Expr::bitand(prefix_ws, Expr::var(format!("{prefix}_ws_{r}")));
        }
        let current_not_ws = Expr::select(
            Expr::eq(Expr::var(format!("{prefix}_ws_{q}")), Expr::u32(0)),
            Expr::u32(1),
            Expr::u32(0),
        );
        let cond_u32 = Expr::bitand(current_not_ws, prefix_ws);
        acc = Expr::select(Expr::eq(cond_u32, Expr::u32(1)), Expr::u32(q), acc);
    }
    acc
}
