//! Decode and decompression kernels for GPU-resident pipelines.
//!
//! Each codec is one module holding its IR builder, its CPU reference oracle,
//! its conformance fixtures and its single registered op id. Encoded bytes stay
//! in the same IR surface used by the matching kernels, so a decode-to-scan
//! chain never leaves the device.
//!
//! The codec module is the public path. Callers write
//! `vyre_libs::decode::hex::hex_decode(...)`; this module re-exports nothing,
//! so an item has one path and the path names the codec that owns it.

/// Base64 decode.
pub mod base64;
mod buffers;
/// Encoding classification from a byte histogram.
pub mod encodex;
/// ASCII hex decode.
pub mod hex;
/// DEFLATE stored-block inflate.
pub mod inflate;
#[cfg(test)]
mod inflate_tests;
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
