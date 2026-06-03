//! Q3 reproducer: `Expr::buf_len(buffer)` lowers to `naga::ArrayLength`
//! on the wgpu/Vulkan path. ArrayLength must equal the bound storage
//! buffer's element count at dispatch time. The cat_a_gpu_differential
//! pass on 2026-05-02 surfaced a regression where the unbounded
//! `vyre-primitives::hash::fnv1a64` registration (loop bound = buf_len)
//! caused the GPU loop to run zero iterations, returning the unchanged
//! FNV1A64_OFFSET.
//!
//! These tests build the smallest possible Program that exercises
//! `Expr::buf_len` at runtime and assert that the dispatched output
//! reflects the actual bound buffer length. They are written to fail
//! before a Q3 fix lands and pass after, so the workaround in
//! `vyre_primitives::hash::fnv1a` (using `fnv1a64_program_n` instead of
//! `fnv1a64_program`) can be reverted with confidence.
//!
//! Lane: `driver_wgpu` (per `docs/optimization/OWNERSHIP.toml`).

use std::sync::{Arc, OnceLock};

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

fn backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        WgpuBackend::acquire().expect(
            "Fix: GPU adapter required for buf_len_array_length tests. Run on a host with a working wgpu adapter.",
        )
    })
}

/// Build a Program whose body writes `buf_len(input)` to `out[0]`.
/// `input` is declared without a static count, so the lowering uses
/// `naga::ArrayLength` to read the bound buffer's element count at
/// runtime. `out` is one u32 with explicit count = 1.
fn buf_len_writer_program() -> Program {
    let body = vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        vec![Node::store("out", Expr::u32(0), Expr::buf_len("input"))],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        body,
    )
}

fn byte_buf_len_writer_program() -> Program {
    let body = vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        vec![Node::store("out", Expr::u32(0), Expr::buf_len("input"))],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U8),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        body,
    )
}

fn byte_load_writer_program(index: u32) -> Program {
    let body = vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::cast(DataType::U32, Expr::load("input", Expr::u32(index))),
        )],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U8),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        body,
    )
}

fn four_byte_pack_writer_program(start: u32) -> Program {
    let mut body = vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        vec![
            Node::let_bind(
                "b0",
                Expr::cast(DataType::U32, Expr::load("input", Expr::u32(start))),
            ),
            Node::let_bind(
                "b1",
                Expr::cast(
                    DataType::U32,
                    Expr::load("input", Expr::u32(start.saturating_add(1))),
                ),
            ),
            Node::let_bind(
                "b2",
                Expr::cast(
                    DataType::U32,
                    Expr::load("input", Expr::u32(start.saturating_add(2))),
                ),
            ),
            Node::let_bind(
                "b3",
                Expr::cast(
                    DataType::U32,
                    Expr::load("input", Expr::u32(start.saturating_add(3))),
                ),
            ),
            Node::store(
                "out",
                Expr::u32(0),
                Expr::bitor(
                    Expr::var("b0"),
                    Expr::bitor(
                        Expr::shl(Expr::var("b1"), Expr::u32(8)),
                        Expr::bitor(
                            Expr::shl(Expr::var("b2"), Expr::u32(16)),
                            Expr::shl(Expr::var("b3"), Expr::u32(24)),
                        ),
                    ),
                ),
            ),
        ],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U8),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        std::mem::take(&mut body),
    )
}

fn dynamic_four_byte_pack_writer_program(words: u32) -> Program {
    let w = Expr::var("w");
    fn byte_expr(k: u32) -> Expr {
        Expr::cast(
            DataType::U32,
            Expr::load(
                "input",
                Expr::add(Expr::mul(Expr::var("w"), Expr::u32(4)), Expr::u32(k)),
            ),
        )
    }
    let body = vec![
        Node::let_bind("w", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(w.clone(), Expr::u32(words)),
            vec![Node::store(
                "out",
                w,
                Expr::bitor(
                    byte_expr(0),
                    Expr::bitor(
                        Expr::shl(byte_expr(1), Expr::u32(8)),
                        Expr::bitor(
                            Expr::shl(byte_expr(2), Expr::u32(16)),
                            Expr::shl(byte_expr(3), Expr::u32(24)),
                        ),
                    ),
                ),
            )],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U8),
            BufferDecl::output("out", 1, DataType::U32).with_count(words),
        ],
        [256, 1, 1],
        body,
    )
}

fn dynamic_four_byte_atomic_pack_writer_program(words: u32) -> Program {
    let w = Expr::var("w");
    fn byte_expr(k: u32) -> Expr {
        Expr::cast(
            DataType::U32,
            Expr::load(
                "input",
                Expr::add(Expr::mul(Expr::var("w"), Expr::u32(4)), Expr::u32(k)),
            ),
        )
    }
    fn atomic_lane(k: u32) -> Node {
        Node::let_bind(
            format!("prev_{k}"),
            Expr::atomic_or(
                "out",
                Expr::var("w"),
                Expr::shl(byte_expr(k), Expr::u32(k * 8)),
            ),
        )
    }
    let body = vec![
        Node::let_bind("w", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(w.clone(), Expr::u32(words)),
            vec![
                atomic_lane(0),
                atomic_lane(1),
                atomic_lane(2),
                atomic_lane(3),
            ],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U8),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(words),
        ],
        [256, 1, 1],
        body,
    )
}

fn dynamic_four_byte_assigned_atomic_pack_writer_program(words: u32) -> Program {
    let w = Expr::var("w");
    fn byte_expr(k: u32) -> Expr {
        Expr::cast(
            DataType::U32,
            Expr::load(
                "input",
                Expr::add(Expr::mul(Expr::var("w"), Expr::u32(4)), Expr::u32(k)),
            ),
        )
    }
    fn lane_nodes(k: u32) -> Vec<Node> {
        vec![
            Node::let_bind(format!("in_byte_{k}"), Expr::u32(0)),
            Node::if_then_else(
                Expr::eq(Expr::u32(0), Expr::u32(1)),
                vec![Node::assign(
                    &format!("in_byte_{k}"),
                    Expr::u32(b' ' as u32),
                )],
                vec![Node::assign(&format!("in_byte_{k}"), byte_expr(k))],
            ),
            Node::let_bind(
                format!("prev_{k}"),
                Expr::atomic_or(
                    "out",
                    Expr::var("w"),
                    Expr::shl(Expr::var(format!("in_byte_{k}")), Expr::u32(k * 8)),
                ),
            ),
        ]
    }
    let mut lanes = Vec::new();
    for k in 0..4 {
        lanes.extend(lane_nodes(k));
    }
    let body = vec![
        Node::let_bind("w", Expr::InvocationId { axis: 0 }),
        Node::if_then(Expr::lt(w.clone(), Expr::u32(words)), lanes),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U8),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(words),
        ],
        [256, 1, 1],
        body,
    )
}

fn dynamic_four_byte_clamped_pack_writer_program(words: u32) -> Program {
    let w = Expr::var("w");
    fn source_byte(k: u32) -> Expr {
        let addr = Expr::add(Expr::mul(Expr::var("w"), Expr::u32(4)), Expr::u32(k));
        let len = Expr::buf_len("input");
        let safe_addr = Expr::select(
            Expr::lt(addr.clone(), len.clone()),
            addr,
            Expr::saturating_sub(len, Expr::u32(1)),
        );
        Expr::bitand(
            Expr::cast(DataType::U32, Expr::load("input", safe_addr)),
            Expr::u32(0xFF),
        )
    }
    fn lane_nodes(k: u32) -> Vec<Node> {
        vec![
            Node::let_bind(format!("in_byte_{k}"), Expr::u32(0)),
            Node::if_then_else(
                Expr::eq(Expr::u32(0), Expr::u32(1)),
                vec![Node::assign(
                    &format!("in_byte_{k}"),
                    Expr::u32(b' ' as u32),
                )],
                vec![Node::assign(&format!("in_byte_{k}"), source_byte(k))],
            ),
            Node::let_bind(
                format!("prev_{k}"),
                Expr::atomic_or(
                    "out",
                    Expr::var("w"),
                    Expr::shl(Expr::var(format!("in_byte_{k}")), Expr::u32(k * 8)),
                ),
            ),
        ]
    }
    let mut lanes = Vec::new();
    for k in 0..4 {
        lanes.extend(lane_nodes(k));
    }
    let body = vec![
        Node::let_bind("w", Expr::InvocationId { axis: 0 }),
        Node::if_then(Expr::lt(w.clone(), Expr::u32(words)), lanes),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U8),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(words),
        ],
        [256, 1, 1],
        body,
    )
}

fn dynamic_offset_scatter_pack_writer_program(words: u32) -> Program {
    let w = Expr::var("w");
    fn i_expr(k: u32) -> Expr {
        Expr::add(Expr::mul(Expr::var("w"), Expr::u32(4)), Expr::u32(k))
    }
    fn source_byte(k: u32) -> Expr {
        let addr = i_expr(k);
        let len = Expr::buf_len("input");
        let safe_addr = Expr::select(
            Expr::lt(addr.clone(), len.clone()),
            addr,
            Expr::saturating_sub(len, Expr::u32(1)),
        );
        Expr::bitand(
            Expr::cast(DataType::U32, Expr::load("input", safe_addr)),
            Expr::u32(0xFF),
        )
    }
    fn lane_nodes(k: u32) -> Vec<Node> {
        let i = i_expr(k);
        vec![
            Node::let_bind(format!("off_{k}"), Expr::load("offsets", i.clone())),
            Node::let_bind(
                format!("out_pos_{k}"),
                Expr::saturating_sub(Expr::var(format!("off_{k}")), Expr::u32(1)),
            ),
            Node::let_bind(
                format!("out_word_idx_{k}"),
                Expr::div(Expr::var(format!("out_pos_{k}")), Expr::u32(4)),
            ),
            Node::let_bind(
                format!("out_shift_{k}"),
                Expr::mul(
                    Expr::rem(Expr::var(format!("out_pos_{k}")), Expr::u32(4)),
                    Expr::u32(8),
                ),
            ),
            Node::let_bind(format!("in_byte_{k}"), source_byte(k)),
            Node::let_bind(
                format!("prev_{k}"),
                Expr::atomic_or(
                    "out",
                    Expr::var(format!("out_word_idx_{k}")),
                    Expr::shl(
                        Expr::var(format!("in_byte_{k}")),
                        Expr::var(format!("out_shift_{k}")),
                    ),
                ),
            ),
        ]
    }
    let mut lanes = Vec::new();
    for k in 0..4 {
        lanes.extend(lane_nodes(k));
    }
    let body = vec![
        Node::let_bind("w", Expr::InvocationId { axis: 0 }),
        Node::if_then(Expr::lt(w.clone(), Expr::u32(words)), lanes),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U8),
            BufferDecl::storage("offsets", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words * 4),
            BufferDecl::storage("out", 2, BufferAccess::ReadWrite, DataType::U32).with_count(words),
        ],
        [256, 1, 1],
        body,
    )
}

fn dynamic_masked_comment_scatter_pack_writer_program(words: u32) -> Program {
    let w = Expr::var("w");
    fn i_expr(k: u32) -> Expr {
        Expr::add(Expr::mul(Expr::var("w"), Expr::u32(4)), Expr::u32(k))
    }
    fn source_byte(i: Expr) -> Expr {
        let len = Expr::buf_len("input");
        let safe_addr = Expr::select(
            Expr::lt(i.clone(), len.clone()),
            i,
            Expr::saturating_sub(len, Expr::u32(1)),
        );
        Expr::bitand(
            Expr::cast(DataType::U32, Expr::load("input", safe_addr)),
            Expr::u32(0xFF),
        )
    }
    fn lane_nodes(k: u32, total_bytes: u32) -> Vec<Node> {
        let i = i_expr(k);
        vec![Node::if_then(
            Expr::lt(i.clone(), Expr::u32(total_bytes)),
            vec![
                Node::let_bind(format!("m_{k}"), Expr::load("mask", i.clone())),
                Node::let_bind(format!("off_{k}"), Expr::load("offsets", i.clone())),
                Node::if_then(
                    Expr::eq(Expr::var(format!("m_{k}")), Expr::u32(1)),
                    vec![
                        Node::let_bind(format!("cm_{k}"), Expr::load("comment_mask", i.clone())),
                        Node::let_bind(format!("in_byte_{k}"), Expr::u32(0)),
                        Node::if_then_else(
                            Expr::eq(Expr::var(format!("cm_{k}")), Expr::u32(2)),
                            vec![Node::assign(
                                &format!("in_byte_{k}"),
                                Expr::u32(b' ' as u32),
                            )],
                            vec![Node::assign(&format!("in_byte_{k}"), source_byte(i))],
                        ),
                        Node::let_bind(
                            format!("out_pos_{k}"),
                            Expr::saturating_sub(Expr::var(format!("off_{k}")), Expr::u32(1)),
                        ),
                        Node::let_bind(
                            format!("out_word_idx_{k}"),
                            Expr::div(Expr::var(format!("out_pos_{k}")), Expr::u32(4)),
                        ),
                        Node::let_bind(
                            format!("out_shift_{k}"),
                            Expr::mul(
                                Expr::rem(Expr::var(format!("out_pos_{k}")), Expr::u32(4)),
                                Expr::u32(8),
                            ),
                        ),
                        Node::let_bind(
                            format!("prev_{k}"),
                            Expr::atomic_or(
                                "out",
                                Expr::var(format!("out_word_idx_{k}")),
                                Expr::shl(
                                    Expr::var(format!("in_byte_{k}")),
                                    Expr::var(format!("out_shift_{k}")),
                                ),
                            ),
                        ),
                    ],
                ),
            ],
        )]
    }
    let total_bytes = words * 4;
    let mut lanes = Vec::new();
    for k in 0..4 {
        lanes.extend(lane_nodes(k, total_bytes));
    }
    let body = vec![
        Node::let_bind("w", Expr::InvocationId { axis: 0 }),
        Node::if_then(Expr::lt(w.clone(), Expr::u32(words)), lanes),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U8),
            BufferDecl::storage("mask", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_bytes),
            BufferDecl::storage("comment_mask", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_bytes),
            BufferDecl::storage("offsets", 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_bytes),
            BufferDecl::storage("out", 4, BufferAccess::ReadWrite, DataType::U32).with_count(words),
        ],
        [256, 1, 1],
        body,
    )
}

fn dispatch_and_read_first_word(program: &Program, input_bytes: Vec<u8>) -> u32 {
    dispatch_and_read_first_word_with_lowering(program, input_bytes, false)
}

/// Like [`dispatch_and_read_first_word`] but routes the program through
/// the same `vyre_foundation::optimizer::pre_lowering::optimize` pass
/// that `cat_a_gpu_differential::lower_for_gpu` uses. The catalog
/// failure cases hit that path; pure direct dispatch does not.
fn dispatch_and_read_first_word_lowered(program: &Program, input_bytes: Vec<u8>) -> u32 {
    dispatch_and_read_first_word_with_lowering(program, input_bytes, true)
}

fn dispatch_and_read_first_word_with_lowering(
    program: &Program,
    input_bytes: Vec<u8>,
    lower: bool,
) -> u32 {
    let lowered;
    let prog = if lower {
        lowered = vyre_foundation::optimizer::pre_lowering::optimize(program.clone());
        &lowered
    } else {
        program
    };
    let inputs = vec![input_bytes, vec![0u8; 4]];
    let outputs = backend()
        .dispatch(prog, &inputs, &DispatchConfig::default())
        .expect("Fix: backend.dispatch must succeed for the buf_len writer program");
    let raw = &outputs[0];
    assert!(
        raw.len() >= 4,
        "Fix: output buffer too small to read a u32 result"
    );
    u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
}

fn dispatch_and_read_words(program: &Program, input_bytes: Vec<u8>) -> Vec<u32> {
    let inputs = vec![input_bytes, vec![0u8; 16]];
    let outputs = backend()
        .dispatch(program, &inputs, &DispatchConfig::default())
        .expect("Fix: backend.dispatch must succeed for the word writer program");
    outputs[0]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn dispatch_and_read_words_with_inputs(program: &Program, inputs: Vec<Vec<u8>>) -> Vec<u32> {
    let outputs = backend()
        .dispatch(program, &inputs, &DispatchConfig::default())
        .expect("Fix: backend.dispatch must succeed for the word writer program");
    outputs[0]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn buf_len_returns_one_element_for_four_byte_input() {
    // The fixture binds a 4-byte input → 1 u32 element. ArrayLength
    // must report 1; the IR Store writes that to out[0]. Before the
    // Q3 fix, this returned 0 on wgpu/Vulkan.
    let program = buf_len_writer_program();
    let observed = dispatch_and_read_first_word(&program, vec![0xAB, 0, 0, 0]);
    assert_eq!(
        observed, 1,
        "Q3: arrayLength on a 4-byte (1×u32) read-only storage buffer must return 1, got {observed}. \
         If this is 0, the wgpu/Vulkan path is computing the binding range wrong for small storage buffers  -  \
         see docs/optimization/ROADMAP.md Q3."
    );
}

#[test]
fn buf_len_returns_three_elements_for_twelve_byte_input() {
    // Three u32 elements. Same reasoning as the single-element case
    // but covers a non-minimal size to rule out a `max(1)` saturation
    // or similar implementation accident.
    let program = buf_len_writer_program();
    let observed =
        dispatch_and_read_first_word(&program, vec![0x01, 0, 0, 0, 0x02, 0, 0, 0, 0x03, 0, 0, 0]);
    assert_eq!(
        observed, 3,
        "Q3: arrayLength on a 12-byte (3×u32) read-only storage buffer must return 3, got {observed}."
    );
}

#[test]
fn buf_len_returns_eight_elements_for_thirty_two_byte_input() {
    // 32 bytes  -  past the 16-byte/32-byte minimum-binding-size
    // thresholds some Vulkan stacks impose. If ArrayLength is broken
    // only below a threshold, this case should pass while the smaller
    // ones fail; documenting the boundary helps Q3's root-cause search.
    let program = buf_len_writer_program();
    let bytes: Vec<u8> = (0..32).map(|i| i as u8).collect();
    let observed = dispatch_and_read_first_word(&program, bytes);
    assert_eq!(
        observed, 8,
        "Q3: arrayLength on a 32-byte (8×u32) read-only storage buffer must return 8, got {observed}."
    );
}

#[test]
fn byte_buf_len_reports_padded_byte_capacity_for_dynamic_u8_input() {
    let program = byte_buf_len_writer_program();
    let observed = dispatch_and_read_first_word(&program, vec![b'a', b'b', b'c', b'd', b'e']);
    assert_eq!(
        observed, 8,
        "U8 storage is packed into WGSL u32 words, so dynamic buf_len must expose byte capacity \
         (arrayLength * 4) to byte-addressed IR helpers; got {observed}."
    );
}

#[test]
fn byte_load_extracts_lane_from_dynamic_u8_input() {
    let program = byte_load_writer_program(11);
    let observed = dispatch_and_read_first_word(&program, b"int x = 1; // trailing\n".to_vec());
    assert_eq!(
        observed,
        u32::from(b'/'),
        "U8 load at byte index 11 must extract lane 3 from the packed WGPU u32 word; got {observed}."
    );
}

#[test]
fn byte_loads_pack_adjacent_lanes_from_dynamic_u8_input() {
    let program = four_byte_pack_writer_program(8);
    let observed = dispatch_and_read_first_word(&program, b"int x = 1; // trailing\n".to_vec());
    assert_eq!(
        observed.to_le_bytes(),
        [b'1', b';', b' ', b'/'],
        "four adjacent U8 loads must preserve byte-addressed lanes before byte compaction."
    );
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

/// Wrap the buf_len writer body in three nested Region nodes to mirror
/// the shape `primitive_catalog::primitive_program` builds for
/// `catalog::hash::fnv1a64::consumer_a/b`. If `arrayLength` works on
/// the flat program but not on the deeply-wrapped one, the bug is in
/// region inlining or pre-lowering rather than in the wgpu binding
/// layer that Q3 was supposed to fix.
fn deep_region_wrapped_buf_len_program() -> Program {
    let inner = vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        vec![Node::store("out", Expr::u32(0), Expr::buf_len("input"))],
    )];
    let mid = Node::Region {
        generator: Ident::from("vyre-primitives::test::buf_len_inner"),
        source_region: None,
        body: Arc::new(inner),
    };
    let outer = Node::Region {
        generator: Ident::from("vyre-primitives::test::buf_len_mid"),
        source_region: Some(GeneratorRef {
            name: "vyre-libs::catalog::test::buf_len_outer".to_string(),
        }),
        body: Arc::new(vec![mid]),
    };
    let body = Node::Region {
        generator: Ident::from("vyre-libs::catalog::test::buf_len_outer"),
        source_region: None,
        body: Arc::new(vec![outer]),
    };
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![body],
    )
}

fn loop_counting_buf_len_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::if_then(
            Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
            vec![
                Node::let_bind("seen", Expr::u32(0)),
                Node::loop_for(
                    "i",
                    Expr::u32(0),
                    Expr::buf_len("input"),
                    vec![Node::assign(
                        "seen",
                        Expr::add(Expr::var("seen"), Expr::u32(1)),
                    )],
                ),
                Node::store("out", Expr::u32(0), Expr::var("seen")),
            ],
        )],
    )
}

#[test]
fn buf_len_through_three_region_wraps_for_one_element() {
    let program = deep_region_wrapped_buf_len_program();
    let observed = dispatch_and_read_first_word(&program, vec![0x99, 0, 0, 0]);
    assert_eq!(
        observed, 1,
        "Q3: arrayLength on a triple-Region-wrapped Program must report 1 for a 4-byte input, got {observed}. \
         If this fails while the flat-program tests pass, region inlining or pre-lowering is breaking the BufLen path \
         in catalog wrappers  -  see ROADMAP.md Q3."
    );
}

#[test]
fn buf_len_through_three_region_wraps_for_three_elements() {
    let program = deep_region_wrapped_buf_len_program();
    let observed = dispatch_and_read_first_word(&program, vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
    assert_eq!(
        observed, 3,
        "Q3: arrayLength on a triple-Region-wrapped Program must report 3 for a 12-byte input, got {observed}."
    );
}

#[test]
fn buf_len_through_three_region_wraps_through_pre_lowering_for_one_element() {
    // The cat_a_gpu_differential test path runs every program through
    // `vyre_foundation::optimizer::pre_lowering::optimize` before
    // dispatch. If buf_len works on the flat or shallow-wrapped path
    // but breaks here, the regression lives in the optimizer pipeline
    // (canonicalize → region_inline → const_fold → loop_unroll →
    // strength_reduce → normalize_atomics → CSE+DCE → ...).
    let program = deep_region_wrapped_buf_len_program();
    let observed = dispatch_and_read_first_word_lowered(&program, vec![0x99, 0, 0, 0]);
    assert_eq!(
        observed, 1,
        "Q3: arrayLength after pre_lowering::optimize on a triple-Region-wrapped Program must report 1 for a 4-byte input, got {observed}. \
         If this fails while the pre-lowering-skipping tests pass, an optimizer pass is folding `Expr::buf_len` to a constant  -  see ROADMAP.md Q3."
    );
}

#[test]
fn buf_len_through_three_region_wraps_through_pre_lowering_for_three_elements() {
    let program = deep_region_wrapped_buf_len_program();
    let observed =
        dispatch_and_read_first_word_lowered(&program, vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
    assert_eq!(
        observed, 3,
        "Q3: arrayLength after pre_lowering::optimize on a triple-Region-wrapped Program must report 3 for a 12-byte input, got {observed}."
    );
}

#[test]
fn buf_len_loop_bound_survives_pre_lowering() {
    let program = loop_counting_buf_len_program();
    let observed =
        dispatch_and_read_first_word_lowered(&program, vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
    assert_eq!(
        observed, 3,
        "Q3: a loop bounded by dynamic buf_len(input) must execute once per bound element after pre_lowering, got {observed}."
    );
}

/// Reproducer that mirrors fnv1a64's structure exactly: triple-Region
/// wrap → if-then(gid==0) → let-bind state → Loop bounded by
/// BufLen(input) → body assigns to outer state → after-loop Store.
/// fnv1a64's catalog form returns the unchanged FNV1A64_OFFSET on
/// GPU (loop body never runs), but the Q3 wgpu fix made arrayLength
/// correct for the simpler tests above. This test isolates whether
/// the bug is the loop+assign+outer-state pattern itself.
fn fnv1a64_shaped_count_program() -> Program {
    // Pattern: outer state `n` initialised to 0, loop runs `buf_len(input)`
    // iterations, each iteration does `n = n + 1`. Final n stored to out[0].
    // For a 4-byte input, expect out[0] = 1.
    let inner = vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        vec![
            Node::let_bind("n", Expr::u32(0)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::buf_len("input"),
                vec![
                    // Mirror fnv1a64's pattern: read input[i], use it,
                    // assign back to outer state via Var.
                    Node::let_bind(
                        "byte",
                        Expr::bitand(Expr::load("input", Expr::var("i")), Expr::u32(0xFF)),
                    ),
                    Node::let_bind("next", Expr::add(Expr::var("n"), Expr::u32(1))),
                    Node::assign("n", Expr::var("next")),
                    // The byte let must survive even if unused by `n`.
                    Node::let_bind("_swallow", Expr::var("byte")),
                ],
            ),
            Node::store("out", Expr::u32(0), Expr::var("n")),
        ],
    )];
    let mid = Node::Region {
        generator: Ident::from("vyre-primitives::test::fnv_shape_inner"),
        source_region: None,
        body: Arc::new(inner),
    };
    let outer = Node::Region {
        generator: Ident::from("vyre-primitives::test::fnv_shape_mid"),
        source_region: Some(GeneratorRef {
            name: "vyre-libs::catalog::test::fnv_shape_outer".to_string(),
        }),
        body: Arc::new(vec![mid]),
    };
    let body = Node::Region {
        generator: Ident::from("vyre-libs::catalog::test::fnv_shape_outer"),
        source_region: None,
        body: Arc::new(vec![outer]),
    };
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![body],
    )
}

#[test]
fn fnv1a64_shaped_loop_runs_once_for_one_byte_input() {
    let program = fnv1a64_shaped_count_program();
    let observed = dispatch_and_read_first_word_lowered(&program, vec![0xAB, 0, 0, 0]);
    assert_eq!(
        observed, 1,
        "Q3: a fnv1a64-shaped loop (BufLen-bounded, with outer-state assign) must iterate once for a 4-byte input, got {observed}. \
         If this fails while the simpler buf_len tests pass, the bug is in how the loop body's outer-scope assigns interact with BufLen lowering."
    );
}

#[test]
fn fnv1a64_shaped_loop_runs_three_times_for_twelve_byte_input() {
    let program = fnv1a64_shaped_count_program();
    let observed =
        dispatch_and_read_first_word_lowered(&program, vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
    assert_eq!(
        observed, 3,
        "Q3: a fnv1a64-shaped loop must iterate three times for a 12-byte input, got {observed}."
    );
}
