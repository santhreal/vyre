//! A host grid-sync segment declares every buffer it still references.
//!
//! # Why a missing declaration is a wrong answer, not a slow one
//!
//! When a backend has no native grid-sync, `dispatch_with_grid_sync_split*`
//! rebuilds the program into one segment per barrier and gives each segment its
//! own buffer table: a buffer the segment neither reads nor writes is dropped so
//! the host does not upload or read back bytes that segment cannot touch. The
//! scan that answers "reads or writes" therefore decides which declarations
//! survive. Miss a reference and the node stays while its declaration goes, and
//! the segment fails descriptor lowering with `buffer not declared but
//! referenced` at dispatch time, on the GPU path only, for whichever program
//! happened to pair a barrier with that node.
//!
//! # Which references previously escaped
//!
//! The scan restated the question as a per-variant `match` over `Store` and the
//! four collectives, ending in `_ => {}`. Every buffer a statement names in an
//! `Ident` field outside those arms was invisible: `AsyncLoad.source`,
//! `AsyncLoad.destination`, `AsyncStore.source`, `AsyncStore.destination`,
//! `IndirectDispatch.count_buffer`, `TileLoad.buffer`, and `TileStore.buffer`.
//! `async_load_source_survives_the_split` is the observed case: a DMA resolver
//! fused behind a Jacobi smoother lost `global_dma_pool` and refused to launch.
//! The scan now delegates to `vyre_foundation::visit::node_buffer_refs`, which
//! is exhaustive in the crate that owns `Node`.
//!
//! # What this suite does NOT claim
//!
//! It does not claim a segment's access roles are minimal, only that they are
//! sufficient: a buffer kept as `ReadWrite` when `ReadOnly` would do costs an
//! upload, not a wrong answer. Nor does it prove the split point is legal; that
//! is `grid_sync_nested_fence_survives_split.rs`.

use vyre_driver::grid_sync::plan_host_grid_sync_segment_programs;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Ident, MemoryOrdering, Node, Program,
};
use vyre_foundation::visit::referenced_buffers;
use vyre_test_support::ir_variants::{assert_covers_every_node_variant, node_variant_samples};

fn grid_fence() -> Node {
    Node::barrier_with_ordering(MemoryOrdering::GridSync)
}

/// A program that runs `nodes`, fences, then stores, declaring every buffer the
/// body references.
///
/// The declaration set is derived from the assembled program rather than listed
/// per fixture: a fixture that names a buffer this helper does not declare would
/// be rejected before the split, and the escape would read as a fixture bug.
///
/// Every derived buffer is declared `ReadOnly` on purpose, and that is not the
/// role the node plays on it. `ReadWrite` is the one access the segment rewrite
/// keeps without any recorded role (the passthrough case, so a fused arm's
/// scratch survives a segment that does not touch it), which makes a
/// `ReadWrite` fixture insensitive to exactly the defect under test. `ReadOnly`
/// has no such exemption: the declaration survives only if the scan recorded a
/// read or a write. The fixture is not asked to be valid IR, only to make the
/// retention decision observable.
fn program_declaring_every_referenced_buffer(nodes: Vec<Node>) -> Program {
    let mut body = nodes;
    body.push(grid_fence());
    body.push(Node::Store {
        buffer: Ident::from("out"),
        index: Expr::u32(0),
        value: Expr::u32(1),
    });
    let probe = Program::wrapped(Vec::new(), [1, 1, 1], body.clone());
    let mut names: Vec<Ident> = referenced_buffers(&probe).into_iter().collect();
    names.sort_by(|left, right| left.as_str().cmp(right.as_str()));
    let buffers: Vec<BufferDecl> = names
        .iter()
        .enumerate()
        .map(|(binding, name)| {
            let binding = u32::try_from(binding).expect("fixture binding index fits u32");
            if name.as_str() == "out" {
                return BufferDecl::output(name.as_str(), binding, DataType::U32).with_count(4);
            }
            BufferDecl::storage(
                name.as_str(),
                binding,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(4)
        })
        .collect();
    Program::wrapped(buffers, [1, 1, 1], body)
}

/// Every buffer a segment references is declared by that segment.
fn assert_segments_declare_their_references(label: &str, program: &Program) {
    let segments = plan_host_grid_sync_segment_programs(program)
        .unwrap_or_else(|error| panic!("{label}: planning the host split failed: {error}"));
    assert!(
        !segments.is_empty(),
        "{label}: a fenced program must split into at least one segment"
    );
    for (index, segment) in segments.iter().enumerate() {
        let declared: Vec<&str> = segment
            .buffers()
            .iter()
            .map(vyre_foundation::ir::BufferDecl::name)
            .collect();
        for referenced in referenced_buffers(segment) {
            assert!(
                declared.contains(&referenced.as_str()),
                "{label}: segment {index} references `{}` but declares only {declared:?}. \
                 Fix: derive the segment read/write sets from `node_buffer_refs` so every \
                 buffer a statement names by ident keeps its declaration.",
                referenced.as_str()
            );
        }
    }
}

/// A new `Node` variant fails here before it can reach the split untested.
#[test]
fn every_declared_node_variant_has_a_fixture() {
    assert_covers_every_node_variant(&node_variant_samples());
}

/// No node variant can carry a buffer reference past the segment buffer scan.
#[test]
fn every_node_variant_keeps_its_buffer_declarations_across_the_split() {
    for sample in node_variant_samples() {
        let program = program_declaring_every_referenced_buffer(vec![sample.node.clone()]);
        assert_segments_declare_their_references(&sample.label(), &program);
    }
}

/// The reported defect: an async copy's source buffer.
///
/// `AsyncLoad` names both of its buffers in `Ident` fields and carries only
/// `offset` and `size` as operands, so an operand-only scan sees neither. The
/// source is the half that used to disappear: the destination survived by being
/// a `ReadWrite` passthrough, which is why the failure surfaced as an undeclared
/// read rather than a lost write.
#[test]
fn async_load_source_survives_the_split() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("pool", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1024),
            BufferDecl::storage("staged", 1, BufferAccess::ReadWrite, DataType::U32).with_count(4),
            BufferDecl::output("out", 2, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::AsyncLoad {
                source: Ident::from("pool"),
                destination: Ident::from("staged"),
                offset: Box::new(Expr::u32(0)),
                size: Box::new(Expr::u32(16)),
                tag: Ident::from("dma"),
            },
            Node::AsyncWait {
                tag: Ident::from("dma"),
            },
            grid_fence(),
            Node::Store {
                buffer: Ident::from("out"),
                index: Expr::u32(0),
                value: Expr::load("staged", Expr::u32(0)),
            },
        ],
    );

    assert_segments_declare_their_references("async DMA behind a fence", &program);

    let segments =
        plan_host_grid_sync_segment_programs(&program).expect("plan the async DMA host split");
    let carries_the_load = segments.iter().any(|segment| {
        segment
            .buffers()
            .iter()
            .any(|buffer| buffer.name() == "pool")
    });
    assert!(
        carries_the_load,
        "Fix: the segment holding the AsyncLoad must declare its source buffer `pool`."
    );
}

/// An async copy's destination is a write, not a passthrough.
///
/// Declaring the destination is not enough: a segment that only forwards it as a
/// `ReadWrite` passthrough is not recorded as its writer, so a later segment
/// that also writes it overwrites the copied bytes instead of merging them.
#[test]
fn async_load_destination_is_recorded_as_a_segment_write() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("pool", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1024),
            BufferDecl::output("staged", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::AsyncLoad {
                source: Ident::from("pool"),
                destination: Ident::from("staged"),
                offset: Box::new(Expr::u32(0)),
                size: Box::new(Expr::u32(16)),
                tag: Ident::from("dma"),
            },
            Node::AsyncWait {
                tag: Ident::from("dma"),
            },
            grid_fence(),
            Node::Store {
                buffer: Ident::from("staged"),
                index: Expr::u32(1),
                value: Expr::u32(7),
            },
        ],
    );

    let segments =
        plan_host_grid_sync_segment_programs(&program).expect("plan the async DMA host split");
    assert_eq!(segments.len(), 2, "one fence splits into two segments");
    let later = segments[1]
        .buffer("staged")
        .expect("the later segment declares the accumulator it writes");
    assert_eq!(
        later.access(),
        BufferAccess::ReadWrite,
        "Fix: a buffer an earlier segment produced must be read back and merged, \
         never overwritten by a WriteOnly declaration."
    );
}

/// An indirect dispatch count buffer survives the split.
#[test]
fn indirect_dispatch_count_buffer_survives_the_split() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("counts", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::IndirectDispatch {
                count_buffer: Ident::from("counts"),
                count_offset: 0,
            },
            grid_fence(),
            Node::Store {
                buffer: Ident::from("out"),
                index: Expr::u32(0),
                value: Expr::u32(1),
            },
        ],
    );

    assert_segments_declare_their_references("indirect dispatch behind a fence", &program);
}
