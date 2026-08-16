//! Decode and decompression kernels for GPU-resident pipelines.
//!
//! Each codec is one module holding its IR builder, its CPU reference oracle,
//! its conformance fixtures and its single registered op id. Encoded bytes stay
//! in the same IR surface used by the matching kernels, so a decode-to-scan
//! chain never leaves the device.

/// Base64 decode.
pub mod base64;
mod buffers;
/// Encoding classification from a byte histogram.
pub mod encodex;
/// ASCII hex decode.
pub mod hex;
/// DEFLATE stored-block inflate.
pub mod inflate;
/// RLE-segment-length scan and start-position prefix sum.
///
/// Foundational stage for block-oriented compression decoders: LZ4 literal and
/// match runs, zstd FSE literal counts, PNG IDAT chunks, snappy raw runs. It
/// unpacks `(length, value)` from packed u32 segment headers. The prefix sum
/// that turns those lengths into per-segment output start offsets is
/// `math::prefix_scan`.
pub mod rle_segment_lengths;
mod scan;
/// Indexed LZ4 literal-copy stage for parallel block decoders.
pub mod ziftsieve;

/// Streaming decode to scan adapter. Fuses a decoder Program with a scanner
/// Program so decoded bytes hand off through workgroup-shared memory instead of
/// a DRAM round trip.
pub mod streaming;

pub use base64::{base64_decode, base64_decode_then_aho_corasick, BASE64_DECODE_TABLE_BUFFER};
pub use encodex::{
    classify_from_histogram, encoding_classify_child, ENC_ASCII, ENC_BINARY, ENC_ISO8859_1,
    ENC_UTF16BE, ENC_UTF16LE, ENC_UTF8,
};
pub use encodex::{encodex_gpu, encodex_reference};
pub use hex::hex_decode_table_ref;
pub use hex::{
    hex_decode, hex_decode_table, hex_decode_then_aho_corasick, HEX_DECODE_TABLE_BUFFER,
};
pub use inflate::{inflate_stored_block, inflate_stored_block_then_aho_corasick};
pub use ziftsieve::{ziftsieve_gpu, ziftsieve_literal_copy, ZiftsieveBuffers, ZiftsieveExtents};
#[cfg(any(test, feature = "cpu-parity"))]
pub use ziftsieve::{ziftsieve_reference_extract_literals, ZiftsieveExtract};
