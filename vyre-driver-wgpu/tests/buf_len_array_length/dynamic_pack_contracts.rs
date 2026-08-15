use super::*;

/// Pack four byte lanes into one output word with a single non-atomic store.
fn dynamic_four_byte_pack_writer_program(words: u32) -> Program {
    let packed = Expr::bitor(
        input_byte(lane_addr(0)),
        Expr::bitor(
            Expr::shl(input_byte(lane_addr(1)), Expr::u32(8)),
            Expr::bitor(
                Expr::shl(input_byte(lane_addr(2)), Expr::u32(16)),
                Expr::shl(input_byte(lane_addr(3)), Expr::u32(24)),
            ),
        ),
    );
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U8),
            BufferDecl::output("out", 1, DataType::U32).with_count(words),
        ],
        [256, 1, 1],
        invocation_gated(
            words,
            vec![Node::store("out", Expr::var("w"), packed)],
        ),
    )
}

/// Pack the same four lanes with one atomic or per lane instead of one store.
fn dynamic_four_byte_atomic_pack_writer_program(words: u32) -> Program {
    packing_program(
        words,
        Vec::new(),
        four_lanes(|k| {
            vec![atomic_or_lane(
                k,
                Expr::var("w"),
                Expr::shl(input_byte(lane_addr(k)), Expr::u32(k * 8)),
            )]
        }),
    )
}

/// As above, but each lane's byte reaches the atomic through an assigned
/// binding written from both arms of an `if_then_else`.
fn dynamic_four_byte_assigned_atomic_pack_writer_program(words: u32) -> Program {
    packing_program(
        words,
        Vec::new(),
        four_lanes(|k| {
            let mut lane = assigned_byte_nodes(
                k,
                Expr::eq(Expr::u32(0), Expr::u32(1)),
                input_byte(lane_addr(k)),
            );
            lane.push(atomic_or_lane(
                k,
                Expr::var("w"),
                Expr::shl(Expr::var(format!("in_byte_{k}")), Expr::u32(k * 8)),
            ));
            lane
        }),
    )
}

/// As above, with the lane's load clamped against `buf_len(input)`.
fn dynamic_four_byte_clamped_pack_writer_program(words: u32) -> Program {
    packing_program(
        words,
        Vec::new(),
        four_lanes(|k| {
            let mut lane = assigned_byte_nodes(
                k,
                Expr::eq(Expr::u32(0), Expr::u32(1)),
                clamped_input_byte(lane_addr(k)),
            );
            lane.push(atomic_or_lane(
                k,
                Expr::var("w"),
                Expr::shl(Expr::var(format!("in_byte_{k}")), Expr::u32(k * 8)),
            ));
            lane
        }),
    )
}

#[test]
fn dynamic_byte_loads_pack_invocation_indexed_lanes_from_u8_input() {
    let program = dynamic_four_byte_pack_writer_program(4);
    let words = dispatch_and_read_words(&program, b"int x = 1; // trailing\n".to_vec());
    assert_eq!(
        words.get(2).copied().unwrap_or_default().to_le_bytes(),
        [b'1', b';', b' ', b'/'],
        "invocation-indexed U8 loads must preserve byte-addressed lanes before byte compaction."
    );
}

#[test]
fn dynamic_byte_loads_atomic_or_pack_invocation_indexed_lanes_from_u8_input() {
    let program = dynamic_four_byte_atomic_pack_writer_program(4);
    let words = dispatch_and_read_words(&program, b"int x = 1; // trailing\n".to_vec());
    assert_eq!(
        words.get(2).copied().unwrap_or_default().to_le_bytes(),
        [b'1', b';', b' ', b'/'],
        "atomic-or byte packing must preserve invocation-indexed U8 lanes before byte compaction."
    );
}

#[test]
fn assigned_dynamic_byte_loads_atomic_or_pack_invocation_indexed_lanes_from_u8_input() {
    let program = dynamic_four_byte_assigned_atomic_pack_writer_program(4);
    let words = dispatch_and_read_words(&program, b"int x = 1; // trailing\n".to_vec());
    assert_eq!(
        words.get(2).copied().unwrap_or_default().to_le_bytes(),
        [b'1', b';', b' ', b'/'],
        "assigned byte variables must preserve invocation-indexed U8 lanes before byte compaction."
    );
}

#[test]
fn clamped_dynamic_byte_loads_atomic_or_pack_invocation_indexed_lanes_from_u8_input() {
    let program = dynamic_four_byte_clamped_pack_writer_program(4);
    let words = dispatch_and_read_words(&program, b"int x = 1; // trailing\n".to_vec());
    assert_eq!(
        words.get(2).copied().unwrap_or_default().to_le_bytes(),
        [b'1', b';', b' ', b'/'],
        "buf_len-clamped byte variables must preserve invocation-indexed U8 lanes before byte compaction."
    );
}

