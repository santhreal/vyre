//! Byte classification and sequence-structure rules emitted per lane.

use vyre_foundation::ir::{Expr, Node};

use super::{UTF8_CONT, UTF8_LEAD_2, UTF8_LEAD_3, UTF8_LEAD_4};

pub(super) fn byte_expr(source: &str, index: Expr) -> Expr {
    crate::builder::state_machine::TableStateMachineComposer::masked_byte_load(source, index)
}

pub(super) fn in_range(value: Expr, lo: u32, hi: u32) -> Expr {
    Expr::and(
        Expr::ge(value.clone(), Expr::u32(lo)),
        Expr::le(value, Expr::u32(hi)),
    )
}

pub(super) fn valid_three_byte_first(lead: Expr, first: Expr) -> Expr {
    Expr::or(
        Expr::or(
            Expr::and(
                Expr::eq(lead.clone(), Expr::u32(0xE0)),
                in_range(first.clone(), 0xA0, 0xBF),
            ),
            Expr::and(
                Expr::eq(lead.clone(), Expr::u32(0xED)),
                in_range(first.clone(), 0x80, 0x9F),
            ),
        ),
        Expr::and(
            Expr::or(
                in_range(lead.clone(), 0xE1, 0xEC),
                in_range(lead, 0xEE, 0xEF),
            ),
            in_range(first, 0x80, 0xBF),
        ),
    )
}

pub(super) fn valid_four_byte_first(lead: Expr, first: Expr) -> Expr {
    Expr::or(
        Expr::or(
            Expr::and(
                Expr::eq(lead.clone(), Expr::u32(0xF0)),
                in_range(first.clone(), 0x90, 0xBF),
            ),
            Expr::and(
                Expr::eq(lead.clone(), Expr::u32(0xF4)),
                in_range(first.clone(), 0x80, 0x8F),
            ),
        ),
        Expr::and(in_range(lead, 0xF1, 0xF3), in_range(first, 0x80, 0xBF)),
    )
}

pub(super) fn continuation_validation_body(source: &str) -> Vec<Node> {
    vec![
        Node::if_then(
            Expr::lt(Expr::u32(0), Expr::var("idx")),
            vec![
                Node::let_bind(
                    "prev1",
                    byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(u32::MAX))),
                ),
                Node::if_then(
                    in_range(Expr::var("prev1"), 0xC2, 0xDF),
                    vec![Node::assign("class", Expr::u32(UTF8_CONT))],
                ),
                Node::if_then(
                    Expr::lt(
                        Expr::add(Expr::var("idx"), Expr::u32(1)),
                        Expr::buf_len(source),
                    ),
                    vec![
                        Node::let_bind(
                            "next1_after_cont3",
                            byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(1))),
                        ),
                        Node::if_then(
                            Expr::and(
                                valid_three_byte_first(Expr::var("prev1"), Expr::var("byte")),
                                in_range(Expr::var("next1_after_cont3"), 0x80, 0xBF),
                            ),
                            vec![Node::assign("class", Expr::u32(UTF8_CONT))],
                        ),
                    ],
                ),
                Node::if_then(
                    Expr::lt(
                        Expr::add(Expr::var("idx"), Expr::u32(2)),
                        Expr::buf_len(source),
                    ),
                    vec![
                        Node::let_bind(
                            "next1_after_cont4",
                            byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(1))),
                        ),
                        Node::let_bind(
                            "next2_after_cont4",
                            byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(2))),
                        ),
                        Node::if_then(
                            Expr::and(
                                Expr::and(
                                    valid_four_byte_first(Expr::var("prev1"), Expr::var("byte")),
                                    in_range(Expr::var("next1_after_cont4"), 0x80, 0xBF),
                                ),
                                in_range(Expr::var("next2_after_cont4"), 0x80, 0xBF),
                            ),
                            vec![Node::assign("class", Expr::u32(UTF8_CONT))],
                        ),
                    ],
                ),
            ],
        ),
        Node::if_then(
            Expr::lt(Expr::u32(1), Expr::var("idx")),
            vec![
                Node::let_bind(
                    "prev2",
                    byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(u32::MAX - 1))),
                ),
                Node::let_bind(
                    "prev1_for_3",
                    byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(u32::MAX))),
                ),
                Node::if_then(
                    valid_three_byte_first(Expr::var("prev2"), Expr::var("prev1_for_3")),
                    vec![Node::assign("class", Expr::u32(UTF8_CONT))],
                ),
                Node::if_then(
                    Expr::lt(
                        Expr::add(Expr::var("idx"), Expr::u32(1)),
                        Expr::buf_len(source),
                    ),
                    vec![
                        Node::let_bind(
                            "next1_after_cont4_mid",
                            byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(1))),
                        ),
                        Node::if_then(
                            Expr::and(
                                Expr::and(
                                    valid_four_byte_first(
                                        Expr::var("prev2"),
                                        Expr::var("prev1_for_3"),
                                    ),
                                    in_range(Expr::var("byte"), 0x80, 0xBF),
                                ),
                                in_range(Expr::var("next1_after_cont4_mid"), 0x80, 0xBF),
                            ),
                            vec![Node::assign("class", Expr::u32(UTF8_CONT))],
                        ),
                    ],
                ),
            ],
        ),
        Node::if_then(
            Expr::lt(Expr::u32(2), Expr::var("idx")),
            vec![
                Node::let_bind(
                    "prev3",
                    byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(u32::MAX - 2))),
                ),
                Node::let_bind(
                    "prev2_for_4",
                    byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(u32::MAX - 1))),
                ),
                Node::let_bind(
                    "prev1_for_4",
                    byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(u32::MAX))),
                ),
                Node::if_then(
                    Expr::and(
                        valid_four_byte_first(Expr::var("prev3"), Expr::var("prev2_for_4")),
                        in_range(Expr::var("prev1_for_4"), 0x80, 0xBF),
                    ),
                    vec![Node::assign("class", Expr::u32(UTF8_CONT))],
                ),
            ],
        ),
    ]
}

pub(super) fn lead2_validation_body(source: &str, n: u32) -> Vec<Node> {
    vec![Node::if_then(
        Expr::lt(Expr::add(Expr::var("idx"), Expr::u32(1)), Expr::u32(n)),
        vec![
            Node::let_bind(
                "next1_for_2",
                byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(1))),
            ),
            Node::if_then(
                in_range(Expr::var("next1_for_2"), 0x80, 0xBF),
                vec![Node::assign("class", Expr::u32(UTF8_LEAD_2))],
            ),
        ],
    )]
}

pub(super) fn lead3_validation_body(source: &str, n: u32) -> Vec<Node> {
    vec![Node::if_then(
        Expr::lt(Expr::add(Expr::var("idx"), Expr::u32(2)), Expr::u32(n)),
        vec![
            Node::let_bind(
                "next1_for_3",
                byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(1))),
            ),
            Node::let_bind(
                "next2_for_3",
                byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(2))),
            ),
            Node::if_then(
                Expr::and(
                    valid_three_byte_first(Expr::var("byte"), Expr::var("next1_for_3")),
                    in_range(Expr::var("next2_for_3"), 0x80, 0xBF),
                ),
                vec![Node::assign("class", Expr::u32(UTF8_LEAD_3))],
            ),
        ],
    )]
}

pub(super) fn lead4_validation_body(source: &str, n: u32) -> Vec<Node> {
    vec![Node::if_then(
        Expr::lt(Expr::add(Expr::var("idx"), Expr::u32(3)), Expr::u32(n)),
        vec![
            Node::let_bind(
                "next1_for_4",
                byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(1))),
            ),
            Node::let_bind(
                "next2_for_4",
                byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(2))),
            ),
            Node::let_bind(
                "next3_for_4",
                byte_expr(source, Expr::add(Expr::var("idx"), Expr::u32(3))),
            ),
            Node::if_then(
                Expr::and(
                    Expr::and(
                        valid_four_byte_first(Expr::var("byte"), Expr::var("next1_for_4")),
                        in_range(Expr::var("next2_for_4"), 0x80, 0xBF),
                    ),
                    in_range(Expr::var("next3_for_4"), 0x80, 0xBF),
                ),
                vec![Node::assign("class", Expr::u32(UTF8_LEAD_4))],
            ),
        ],
    )]
}
