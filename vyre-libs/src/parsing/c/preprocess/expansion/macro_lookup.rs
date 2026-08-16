//! Macro-table lookup builders: name-key scan, hashed name scan, and the
//! source-span hash the hashed scan is keyed on.

use vyre_foundation::ir::{Expr, Node};
use crate::hash::fnv1a::{fnv1a32_initial_expr, fnv1a32_update_byte_node};

use super::*;

pub(super) fn emit_macro_lookup(
    prefix: &str,
    token: Expr,
    macro_keys: &str,
    macro_vals: &str,
    output_var: &str,
) -> Vec<Node> {
    let token_name = format!("{prefix}_tok");
    let probe_slot = format!("{prefix}_probe_slot");
    let probed_key = format!("{prefix}_probed_key");
    let probe = format!("{prefix}_probe");
    let lookup_done = format!("{prefix}_lookup_done");
    let lookup_seen_empty = format!("{prefix}_lookup_seen_empty");
    vec![
        Node::let_bind(&token_name, token),
        Node::let_bind(
            &probe_slot,
            Expr::bitand(
                Expr::mul(Expr::var(&token_name), Expr::u32(2_654_435_769)),
                Expr::u32(MACRO_TABLE_MASK),
            ),
        ),
        Node::let_bind(output_var, Expr::u32(EMPTY_MACRO_SLOT)),
        Node::let_bind(&lookup_done, Expr::u32(0)),
        Node::let_bind(&lookup_seen_empty, Expr::u32(0)),
        Node::loop_for(
            probe,
            Expr::u32(0),
            Expr::u32(MACRO_TABLE_SLOTS),
            vec![Node::if_then(
                Expr::eq(Expr::var(&lookup_done), Expr::u32(0)),
                vec![
                    Node::let_bind(&probed_key, Expr::load(macro_keys, Expr::var(&probe_slot))),
                    Node::if_then(
                        Expr::eq(Expr::var(&probed_key), Expr::var(&token_name)),
                        vec![
                            Node::assign(
                                output_var,
                                Expr::load(macro_vals, Expr::var(&probe_slot)),
                            ),
                            Node::assign(&lookup_done, Expr::u32(1)),
                        ],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var(&probed_key), Expr::u32(EMPTY_MACRO_SLOT)),
                        vec![
                            Node::assign(&lookup_seen_empty, Expr::u32(1)),
                            Node::assign(&lookup_done, Expr::u32(1)),
                        ],
                    ),
                    Node::assign(
                        &probe_slot,
                        Expr::bitand(
                            Expr::add(Expr::var(&probe_slot), Expr::u32(1)),
                            Expr::u32(MACRO_TABLE_MASK),
                        ),
                    ),
                ],
            )],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var(output_var), Expr::u32(EMPTY_MACRO_SLOT)),
                Expr::eq(Expr::var(&lookup_seen_empty), Expr::u32(0)),
            ),
            vec![Node::trap(
                Expr::var(&token_name),
                "macro-lookup-table-full-without-empty-slot",
            )],
        ),
    ]
}

pub(super) fn emit_macro_hash_lookup(
    prefix: &str,
    name_hash: Expr,
    source_start: Expr,
    source_len: Expr,
    source_words: &str,
    source_layout: MacroByteLayout,
    macro_name_hashes: &str,
    macro_name_starts: &str,
    macro_name_lens: &str,
    macro_name_words: &str,
    macro_name_layout: MacroByteLayout,
    output_var: &str,
) -> Vec<Node> {
    let hash_name = format!("{prefix}_name_hash");
    let probe_slot = format!("{prefix}_probe_slot");
    let probed_key = format!("{prefix}_probed_key");
    let probe = format!("{prefix}_probe");
    let lookup_done = format!("{prefix}_lookup_done");
    let lookup_seen_empty = format!("{prefix}_lookup_seen_empty");
    let candidate_name_start = format!("{prefix}_candidate_name_start");
    let candidate_name_len = format!("{prefix}_candidate_name_len");
    let candidate_name_end = format!("{prefix}_candidate_name_end");
    let candidate_name_matches = format!("{prefix}_candidate_name_matches");
    let candidate_byte_i = format!("{prefix}_candidate_byte_i");
    let source_byte = format!("{prefix}_source_byte");
    let macro_name_byte = format!("{prefix}_macro_name_byte");
    vec![
        Node::let_bind(&hash_name, name_hash),
        Node::let_bind(
            &probe_slot,
            Expr::bitand(
                Expr::mul(Expr::var(&hash_name), Expr::u32(2_654_435_769)),
                Expr::u32(MACRO_TABLE_MASK),
            ),
        ),
        Node::assign(output_var, Expr::u32(EMPTY_MACRO_SLOT)),
        Node::let_bind(&lookup_done, Expr::u32(0)),
        Node::let_bind(&lookup_seen_empty, Expr::u32(0)),
        Node::loop_for(
            probe,
            Expr::u32(0),
            Expr::u32(MACRO_TABLE_SLOTS),
            vec![Node::if_then(
                Expr::eq(Expr::var(&lookup_done), Expr::u32(0)),
                vec![
                    Node::let_bind(
                        &probed_key,
                        Expr::load(macro_name_hashes, Expr::var(&probe_slot)),
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var(&probed_key), Expr::var(&hash_name)),
                        vec![
                            Node::let_bind(
                                &candidate_name_start,
                                Expr::load(macro_name_starts, Expr::var(&probe_slot)),
                            ),
                            Node::let_bind(
                                &candidate_name_len,
                                Expr::load(macro_name_lens, Expr::var(&probe_slot)),
                            ),
                            Node::let_bind(
                                &candidate_name_end,
                                Expr::add(
                                    Expr::var(&candidate_name_start),
                                    Expr::var(&candidate_name_len),
                                ),
                            ),
                            Node::if_then(
                                Expr::or(
                                    Expr::lt(
                                        Expr::var(&candidate_name_end),
                                        Expr::var(&candidate_name_start),
                                    ),
                                    Expr::gt(
                                        Expr::var(&candidate_name_end),
                                        Expr::buf_len(macro_name_words),
                                    ),
                                ),
                                vec![Node::trap(
                                    Expr::var(&candidate_name_end),
                                    "macro-name-candidate-span-out-of-bounds",
                                )],
                            ),
                            Node::let_bind(
                                &candidate_name_matches,
                                Expr::select(
                                    Expr::eq(source_len.clone(), Expr::var(&candidate_name_len)),
                                    Expr::u32(1),
                                    Expr::u32(0),
                                ),
                            ),
                            Node::loop_for(
                                candidate_byte_i.clone(),
                                Expr::u32(0),
                                Expr::var(&candidate_name_len),
                                vec![Node::if_then(
                                    Expr::eq(Expr::var(&candidate_name_matches), Expr::u32(1)),
                                    vec![
                                        Node::let_bind(
                                            &source_byte,
                                            load_macro_byte(
                                                source_words,
                                                source_layout,
                                                Expr::add(
                                                    source_start.clone(),
                                                    Expr::var(&candidate_byte_i),
                                                ),
                                            ),
                                        ),
                                        Node::let_bind(
                                            &macro_name_byte,
                                            load_macro_byte(
                                                macro_name_words,
                                                macro_name_layout,
                                                Expr::add(
                                                    Expr::var(&candidate_name_start),
                                                    Expr::var(&candidate_byte_i),
                                                ),
                                            ),
                                        ),
                                        Node::if_then(
                                            Expr::ne(
                                                Expr::var(&source_byte),
                                                Expr::var(&macro_name_byte),
                                            ),
                                            vec![Node::assign(
                                                &candidate_name_matches,
                                                Expr::u32(0),
                                            )],
                                        ),
                                    ],
                                )],
                            ),
                            Node::if_then(
                                Expr::eq(Expr::var(&candidate_name_matches), Expr::u32(1)),
                                vec![
                                    Node::assign(output_var, Expr::var(&probe_slot)),
                                    Node::assign(&lookup_done, Expr::u32(1)),
                                ],
                            ),
                        ],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var(&probed_key), Expr::u32(EMPTY_MACRO_SLOT)),
                        vec![
                            Node::assign(&lookup_seen_empty, Expr::u32(1)),
                            Node::assign(&lookup_done, Expr::u32(1)),
                        ],
                    ),
                    Node::assign(
                        &probe_slot,
                        Expr::bitand(
                            Expr::add(Expr::var(&probe_slot), Expr::u32(1)),
                            Expr::u32(MACRO_TABLE_MASK),
                        ),
                    ),
                ],
            )],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var(output_var), Expr::u32(EMPTY_MACRO_SLOT)),
                Expr::eq(Expr::var(&lookup_seen_empty), Expr::u32(0)),
            ),
            vec![Node::trap(
                Expr::var(&hash_name),
                "macro-name-lookup-table-full-without-empty-slot",
            )],
        ),
    ]
}

pub(super) fn emit_source_span_hash(
    prefix: &str,
    token_index: Expr,
    in_tok_starts: &str,
    in_tok_lens: &str,
    source_words: &str,
    source_layout: MacroByteLayout,
    source_len: Expr,
    output_var: &str,
) -> Vec<Node> {
    let start = format!("{prefix}_start");
    let len = format!("{prefix}_len");
    let end = format!("{prefix}_end");
    let byte_idx = format!("{prefix}_byte_idx");
    let byte = format!("{prefix}_byte");
    vec![
        Node::let_bind(&start, Expr::load(in_tok_starts, token_index.clone())),
        Node::let_bind(&len, Expr::load(in_tok_lens, token_index)),
        Node::let_bind(&end, Expr::add(Expr::var(&start), Expr::var(&len))),
        Node::if_then(
            Expr::or(
                Expr::lt(Expr::var(&end), Expr::var(&start)),
                Expr::gt(Expr::var(&end), source_len),
            ),
            vec![Node::trap(
                Expr::var(&end),
                "macro-name-source-span-out-of-bounds",
            )],
        ),
        Node::let_bind(output_var, fnv1a32_initial_expr()),
        Node::loop_for(
            byte_idx.clone(),
            Expr::u32(0),
            Expr::var(&len),
            vec![
                Node::let_bind(
                    &byte,
                    load_macro_byte(
                        source_words,
                        source_layout,
                        Expr::add(Expr::var(&start), Expr::var(&byte_idx)),
                    ),
                ),
                fnv1a32_update_byte_node(output_var, Expr::var(&byte)),
            ],
        ),
    ]
}
