use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::region::wrap_anonymous;
use crate::scan::builders::load_packed_byte;
use crate::scan::dfa::CompiledDfa;

/// Build a bounded-window AC program that returns only the total match count.
///
/// This is the GPU preflight shape for irregular scans: one pass over the
/// packed haystack, no match-triple output allocation, and a four-byte readback.
#[must_use]
pub fn classic_ac_bounded_count_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    haystack_len: &str,
    match_count: &str,
    state_count: u32,
    max_pattern_len: u32,
) -> Program {
    let max_pattern_len = max_pattern_len.max(1);
    let i = Expr::var("i");
    let end = Expr::add(i.clone(), Expr::u32(1));
    let scan_start = Expr::select(
        Expr::lt(i.clone(), Expr::u32(max_pattern_len - 1)),
        Expr::u32(0),
        Expr::sub(end.clone(), Expr::u32(max_pattern_len)),
    );
    let (load_step_byte, step_byte) = load_packed_byte(haystack, Expr::var("step"));

    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::load(haystack_len, Expr::u32(0))),
            vec![
                Node::let_bind("state", Expr::u32(0)),
                Node::let_bind("scan_start", scan_start),
                Node::let_bind("scan_end", end),
                Node::loop_for(
                    "step",
                    Expr::var("scan_start"),
                    Expr::var("scan_end"),
                    vec![
                        load_step_byte,
                        Node::assign(
                            "state",
                            Expr::load(
                                transitions,
                                Expr::add(Expr::mul(Expr::var("state"), Expr::u32(256)), step_byte),
                            ),
                        ),
                    ],
                ),
                Node::let_bind("out_begin", Expr::load(output_offsets, Expr::var("state"))),
                Node::let_bind(
                    "out_end",
                    Expr::load(output_offsets, Expr::add(Expr::var("state"), Expr::u32(1))),
                ),
                Node::let_bind(
                    "out_count",
                    Expr::sub(Expr::var("out_end"), Expr::var("out_begin")),
                ),
                Node::if_then(
                    Expr::ne(Expr::var("out_count"), Expr::u32(0)),
                    vec![Node::let_bind(
                        "_count_old",
                        Expr::atomic_add(match_count, Expr::u32(0), Expr::var("out_count")),
                    )],
                ),
            ],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(transitions, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(output_offsets, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_add(1)),
            BufferDecl::storage(haystack_len, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::read_write(match_count, 4, DataType::U32).with_count(1),
        ],
        [128, 1, 1],
        vec![wrap_anonymous(
            "vyre-libs::matching::classic_ac_bounded_count",
            body,
        )],
    )
}

/// Build a bounded-window AC count-only program for a compiled DFA.
#[must_use]
pub fn build_ac_bounded_count_program(dfa: &CompiledDfa) -> Program {
    classic_ac_bounded_count_program(
        "haystack",
        "transitions",
        "output_offsets",
        "haystack_len",
        "match_count",
        dfa.state_count,
        dfa.max_pattern_len,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::classic_ac::{classic_ac_compile, classic_ac_scan_counts};
    use crate::scan::{pack_haystack_u32, pack_u32_slice};

    fn decode_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn with_reference_dispatch_lanes(program: Program, lanes: u32) -> Program {
        let buffers = program
            .buffers()
            .iter()
            .cloned()
            .map(|buffer| {
                if buffer.name() == "match_count" {
                    buffer.with_count(lanes.max(1)).with_output_byte_range(0..4)
                } else {
                    buffer
                }
            })
            .collect();
        program.with_rewritten_buffers(buffers)
    }

    #[test]
    fn bounded_count_program_reference_eval_matches_cpu_count() {
        let patterns: [&[u8]; 4] = [b"a", b"aa", b"she", b"he"];
        let haystack = b"aaashehe";
        let ac = classic_ac_compile(&patterns);
        let expected = classic_ac_scan_counts(&ac, haystack).iter().sum::<u32>();
        let program = with_reference_dispatch_lanes(
            build_ac_bounded_count_program(&ac.dfa),
            haystack.len() as u32,
        );
        let inputs = vec![
            vyre_reference::value::Value::from(pack_haystack_u32(haystack)),
            vyre_reference::value::Value::from(pack_u32_slice(&ac.dfa.transitions)),
            vyre_reference::value::Value::from(pack_u32_slice(&ac.dfa.output_offsets)),
            vyre_reference::value::Value::from(pack_u32_slice(&[haystack.len() as u32])),
            vyre_reference::value::Value::from(vec![0_u8; haystack.len() * 4]),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: AC bounded count program should evaluate in reference backend.");

        assert_eq!(decode_u32(&outputs[0].to_bytes()), vec![expected]);
    }

    #[test]
    fn bounded_count_program_has_compact_stable_shape() {
        let ac = classic_ac_compile(&[b"Authorization: Bearer ", b"token", b"tok"]);
        let program = build_ac_bounded_count_program(&ac.dfa);

        assert_eq!(program.workgroup_size(), [128, 1, 1]);
        assert_eq!(program.buffers().len(), 5);
        assert_eq!(program.buffers()[4].name(), "match_count");
        assert_eq!(program.buffers()[4].count, 1);
    }
}
