use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op-id under which this VFS resolver registers itself in the
/// inventory.
///
/// The resolver is owned and registered by `vyre-libs`.
pub const VFS_RESOLVE_OP_ID: &str = "vyre-libs::vfs::resolve";

/// GPU-Native Virtual File System (VFS) Asynchronous DMA Resolver
///
/// Resolves one `#include` directive string identifier into an asynchronous
/// block load from High-Bandwidth Memory / Persistent Storage directly into the
/// L1 Warp-Arena.
///
/// One dispatch resolves one request. An async transfer names a source offset
/// and a length, and lands at the head of its destination, so a second
/// workgroup running the same body would write the same destination region with
/// another file's bytes and the result would depend on which workgroup finished
/// last. `block_words` sizes the destination and the transfer together, so the
/// copy is exactly the block the caller asked for rather than a fixed length
/// silently clamped to whatever the destination happened to hold.
#[must_use]
pub fn vfs_resolve_dma(include_hashes: &str, out_file_buffers: &str, block_words: u32) -> Program {
    let words = block_words.max(1);
    let resolve_body = vec![
        // Async transfers pair on stable stream tags and execute as
        // workgroup-collective transfers where the leader issues the DMA and
        // AsyncWait synchronizes the workgroup. The offset names the load
        // itself rather than a binding over it: a load from a read-only buffer
        // at a literal index is workgroup-uniform in the operand it is written
        // in, and `Let` records uniformity through the ordinary analysis, which
        // refuses every load.
        Node::AsyncLoad {
            source: Ident::from("global_dma_pool"),
            destination: Ident::from(out_file_buffers),
            offset: Box::new(Expr::load(include_hashes, Expr::u32(0))),
            size: Box::new(Expr::u32(words * 4)),
            tag: Ident::from("vfs_req"),
        },
        Node::AsyncWait {
            tag: Ident::from("vfs_req"),
        },
    ];

    let body = vec![wrap_anonymous_region(VFS_RESOLVE_OP_ID, resolve_body)];

    Program::wrapped(
        vec![
            BufferDecl::storage(include_hashes, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage(out_file_buffers, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(words),
            BufferDecl::storage("global_dma_pool", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1024),
        ],
        [256, 1, 1], // Warp aligned
        body,
    )
}

const EXPECTED_VFS_RESOLVE_OUTPUT_BYTES: [u8; 4] = [1, 2, 3, 4];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        VFS_RESOLVE_OP_ID,
        || vfs_resolve_dma("include_hashes", "out_file_buffers", 1),
        Some(|| {
            let mut dma_pool = vec![0u8; 4096];
            dma_pool[..4].copy_from_slice(&[1, 2, 3, 4]);
            vec![vec![
                0u32.to_le_bytes().to_vec(),
                vec![0u8; 4],
                dma_pool,
            ]]
        }),
        Some(|| {
            vec![vec![
                EXPECTED_VFS_RESOLVE_OUTPUT_BYTES.to_vec(),
            ]]
        }),
    )
}
