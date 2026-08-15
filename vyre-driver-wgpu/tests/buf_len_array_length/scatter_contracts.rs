use super::*;

/// Scatter each lane's clamped input byte to the offset `offsets[i]` names.
fn dynamic_offset_scatter_pack_writer_program(words: u32) -> Program {
    packing_program(
        words,
        vec![
            BufferDecl::storage("offsets", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words * 4),
        ],
        four_lanes(|k| {
            let mut lane = vec![Node::let_bind(
                format!("off_{k}"),
                Expr::load("offsets", lane_addr(k)),
            )];
            lane.extend(scatter_position_nodes(k));
            lane.push(Node::let_bind(
                format!("in_byte_{k}"),
                clamped_input_byte(lane_addr(k)),
            ));
            lane.push(atomic_or_lane(
                k,
                Expr::var(format!("out_word_idx_{k}")),
                Expr::shl(
                    Expr::var(format!("in_byte_{k}")),
                    Expr::var(format!("out_shift_{k}")),
                ),
            ));
            lane
        }),
    )
}

/// The same scatter gated on a keep mask, substituting a space for a byte the
/// comment mask marks as comment interior.
fn dynamic_masked_comment_scatter_pack_writer_program(words: u32) -> Program {
    let total_bytes = words * 4;
    packing_program(
        words,
        vec![
            BufferDecl::storage("mask", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_bytes),
            BufferDecl::storage("comment_mask", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_bytes),
            BufferDecl::storage("offsets", 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_bytes),
        ],
        four_lanes(|k| {
            let addr = lane_addr(k);
            let mut kept = vec![Node::let_bind(
                format!("cm_{k}"),
                Expr::load("comment_mask", addr.clone()),
            )];
            kept.extend(assigned_byte_nodes(
                k,
                Expr::eq(Expr::var(format!("cm_{k}")), Expr::u32(2)),
                clamped_input_byte(addr.clone()),
            ));
            kept.extend(scatter_position_nodes(k));
            kept.push(atomic_or_lane(
                k,
                Expr::var(format!("out_word_idx_{k}")),
                Expr::shl(
                    Expr::var(format!("in_byte_{k}")),
                    Expr::var(format!("out_shift_{k}")),
                ),
            ));

            vec![Node::if_then(
                Expr::lt(addr.clone(), Expr::u32(total_bytes)),
                vec![
                    Node::let_bind(format!("m_{k}"), Expr::load("mask", addr.clone())),
                    Node::let_bind(format!("off_{k}"), Expr::load("offsets", addr)),
                    Node::if_then(
                        Expr::eq(Expr::var(format!("m_{k}")), Expr::u32(1)),
                        kept,
                    ),
                ],
            )]
        }),
    )
}

#[test]
fn dynamic_offset_scatter_packs_invocation_indexed_lanes_from_u8_input() {
    let program = dynamic_offset_scatter_pack_writer_program(4);
    let offsets: Vec<u32> = (1..=16).collect();
    let words = dispatch_and_read_words_with_inputs(
        &program,
        vec![
            b"int x = 1; // trailing\n".to_vec(),
            u32_bytes(&offsets),
            vec![0u8; 16],
        ],
    );
    assert_eq!(
        words.get(2).copied().unwrap_or_default().to_le_bytes(),
        [b'1', b';', b' ', b'/'],
        "offset-driven byte scatter must preserve invocation-indexed U8 lanes before byte compaction."
    );
}

#[test]
fn dynamic_masked_comment_scatter_packs_expected_lanes_from_u8_input() {
    let program = dynamic_masked_comment_scatter_pack_writer_program(256);
    let keep_prefix = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    ];
    let comment_prefix = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let offsets_prefix = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 24, 24, 24, 24, 24, 24,
    ];
    let mut keep = vec![0u32; 1024];
    let mut comment = vec![0u32; 1024];
    let mut offsets = vec![24u32; 1024];
    keep[..keep_prefix.len()].copy_from_slice(&keep_prefix);
    comment[..comment_prefix.len()].copy_from_slice(&comment_prefix);
    offsets[..offsets_prefix.len()].copy_from_slice(&offsets_prefix);
    let words = dispatch_and_read_words_with_inputs(
        &program,
        vec![
            b"int x = 1; // trailing\nint y = 2;\n".to_vec(),
            u32_bytes(&keep),
            u32_bytes(&comment),
            u32_bytes(&offsets),
            vec![0u8; 1024],
        ],
    );
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    assert_eq!(
        &bytes[..24],
        b"int x = 1;  \nint y = 2;\n",
        "mask/comment-driven byte scatter must match simple line comment compaction."
    );
}

