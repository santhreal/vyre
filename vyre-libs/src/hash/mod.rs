//! Hash and checksum compositions.
//!
//! The path is the interface. Callers write `vyre_libs::hash::fnv1a::fnv1a32(..)`
//! and name the operation they reached, so this module exposes its sub-modules
//! rather than a flat namespace.

/// Multi-intrinsic BLAKE3 compression composition.
pub(crate) mod blake3_compress;

/// FNV-1a 32-bit + 64-bit hash primitives.
pub mod fnv1a;

/// Shared BLAKE3 mix/round helpers.
pub mod blake3;

/// CRC-32 (IEEE 802.3 polynomial 0xEDB88320) hash primitive.
pub mod crc32;

/// Adler-32 checksum primitive.
pub mod adler32;

/// Fused CRC-32 + FNV-1a32 + Adler-32 one-pass primitive.
pub mod multi_hash;

/// Hash table primitives.
pub mod table;

/// Vector Symbolic Architecture (VSA) primitives  -  bind + bundle on
/// 10K-dim binary hypervectors. The same Programs serve retrieval,
/// reasoning, and content-addressable Program fingerprint compositions.
pub mod hypervector;

/// Count-Sketch  -  Charikar 2002 frequency-moment estimator. Same
/// Program serves streaming, observability, and profiler latency-distribution
/// sketching.
pub mod sketch;

/// Number-Theoretic Transform  -  exact-integer FFT over GF(p) for
/// FHE / zk / lattice crypto. CPU + per-stage butterfly Program.
/// 32-bit prime variant; 64-bit Goldilocks ships with U64 buffers.
pub mod ntt;

/// Hassanieh-Indyk-Katabi-Price sparse FFT bin-hash primitive.
/// Sparse audio, radio, and imaging analysis composition block.
pub mod sparse_fft;

pub use blake3_compress::blake3_compress;

#[must_use]
pub(crate) fn wrap_unary_scalar_hash_program(
    op_id: &'static str,
    input: &str,
    out: &str,
    n: u32,
    body: Vec<vyre_foundation::ir::Node>,
) -> vyre_foundation::ir::Program {
    use vyre_foundation::composition::wrap_anonymous_region;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::output(out, 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(op_id, body)],
    )
}
