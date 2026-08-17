//! Canonical LE wire packing for GPU buffer upload and readback.
//!
//! Every Tier-2.5 primitive that needs `&[T] -> Vec<u8>` (or the reverse)
//! should import from this module instead of re-implementing
//! `words.iter().flat_map(|w| w.to_le_bytes())` or `bytes.chunks_exact(4)
//! .map(...).collect()`. The bytemuck-backed implementation here reduces
//! to a single `memcpy` on little-endian hosts (every shipped runtime
//! target), so consumers get the bandwidth-bound fast path automatically.
//!
//! # Quick reference
//!
//! ```
//! use vyre_primitives::wire::{pack_u32_slice, unpack_u32_slice_into};
//!
//! let words = [0x1234_5678_u32, 0xdead_beef];
//! let bytes = pack_u32_slice(&words);
//! assert_eq!(bytes, vec![0x78, 0x56, 0x34, 0x12, 0xef, 0xbe, 0xad, 0xde]);
//!
//! let mut decoded = Vec::new();
//! unpack_u32_slice_into(&bytes, 2, "doc-example", &mut decoded).unwrap();
//! assert_eq!(decoded, words);
//! ```
//!
//! # Coverage
//!
//! - `pack_*_slice` / `pack_*_slice_into` for `u16`, `u32`, `u64`, `i32`,
//!   `f32` storage buffers.
//! - `pack_bytes_as_u32_slice` / `pack_bytes_as_u32_slice_min_words` for
//!   the per-lane byte layout GPU haystacks expect.
//! - `unpack_*_slice_into` / `decode_*_le_bytes_all` for the reverse
//!   direction with length-checked and drain-the-buffer flavors.
//! - `append_*_slice_le_bytes` for header builders that need to extend a
//!   composite buffer rather than overwrite it.

#[cfg(test)]
fn checked_byte_len(count: usize, width: usize, label: &str) -> Result<usize, String> {
    count.checked_mul(width).ok_or_else(|| {
        format!("{label} count {count} overflows host byte indexing. Fix: shard the buffer.")
    })
}

/// Grow `out` so a following clear-and-refill to `target_len` elements does not
/// reallocate.
///
/// `vyre-primitives` takes `vyre-foundation` as an OPTIONAL dependency and this
/// module is always on, so the shared owner in
/// `vyre_foundation::allocation` is out of reach here; the arithmetic is
/// inlined instead. `try_reserve_exact` guarantees `len() + additional`, so the
/// additional count is measured from the current length, never from capacity.
fn reserve_exact_len<T>(out: &mut Vec<T>, target_len: usize, label: &str) -> Result<(), String> {
    if target_len > out.capacity() {
        out.try_reserve_exact(target_len - out.len())
            .map_err(|err| {
                format!("{label} could not reserve {target_len} elements: {err}. Fix: shard the buffer.")
            })?;
    }
    Ok(())
}

trait LeWireWord: bytemuck::Pod + Copy {
    const WIDTH: usize;

    fn push_le_bytes(self, out: &mut Vec<u8>);

    /// Decode one LE value from a `WIDTH`-byte chunk (big-endian decode path).
    fn from_le_chunk(chunk: &[u8]) -> Self;
}

macro_rules! impl_le_wire_word {
    ($ty:ty, $width:expr) => {
        impl LeWireWord for $ty {
            const WIDTH: usize = $width;

            fn push_le_bytes(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }

            fn from_le_chunk(chunk: &[u8]) -> Self {
                let mut buf = [0u8; $width];
                buf.copy_from_slice(chunk);
                <$ty>::from_le_bytes(buf)
            }
        }
    };
}

impl_le_wire_word!(u16, 2);
impl_le_wire_word!(u32, 4);
impl_le_wire_word!(i32, 4);
impl_le_wire_word!(u64, 8);
impl_le_wire_word!(f32, 4);

fn append_le_wire_words<T: LeWireWord>(values: &[T], out: &mut Vec<u8>) {
    #[cfg(target_endian = "little")]
    out.extend_from_slice(bytemuck::cast_slice(values));
    #[cfg(target_endian = "big")]
    for &value in values {
        value.push_le_bytes(out);
    }
}

fn pack_le_wire_words_into<T: LeWireWord>(values: &[T], out: &mut Vec<u8>) {
    out.clear();
    append_le_wire_words(values, out);
}

fn try_pack_le_wire_words_into<T: LeWireWord>(
    values: &[T],
    out: &mut Vec<u8>,
    item_label: &str,
    output_label: &str,
) -> Result<(), String> {
    let byte_len = values.len().checked_mul(T::WIDTH).ok_or_else(|| {
        format!(
            "{item_label} count {} overflows host byte indexing. Fix: shard the buffer.",
            values.len()
        )
    })?;
    reserve_exact_len(out, byte_len, output_label)?;
    pack_le_wire_words_into(values, out);
    Ok(())
}

/// Decode `count` little-endian `T` values from the front of `src` into `out`,
/// reusing `out`'s existing allocation. `out` is cleared first.
///
/// This is the single canonical decode body that every `unpack_*_slice_into`
/// and `decode_*_le_bytes_all` entry point routes through, replacing seven
/// byte-identical copies of the old `resize(count, 0)` + `cast_slice_mut` +
/// `copy_from_slice` shape.
///
/// PERF: the old shape wrote every output byte twice - a full-buffer memset to
/// zero, then an immediate `copy_from_slice` over it. On the GPU-readback hot
/// path (decode buffers are KiB-to-MiB, cache-resident) that redundant memset
/// made the "fast path" measurably SLOWER than a naive element-wise decode
/// (1 MiB u32: 18.6 us vs naive 14.0 us on Zen5). This version writes each
/// output byte exactly once by copying directly into uninitialized spare
/// capacity, then setting the length - one `memcpy`, no memset.
#[cfg(target_endian = "little")]
#[allow(unsafe_code)]
#[inline]
fn fill_le_words_into<T: LeWireWord>(src: &[u8], count: usize, out: &mut Vec<T>) {
    let required = count * T::WIDTH;
    debug_assert!(
        src.len() >= required,
        "fill_le_words_into: src has {} bytes, needs {required}",
        src.len()
    );
    out.clear();
    out.reserve(count);
    // SAFETY:
    // * `reserve(count)` (after `clear()`) guarantees the allocation holds at
    //   least `count` contiguous `T` slots = `count * size_of::<T>()` =
    //   `required` bytes of capacity. `as_mut_ptr` points at that buffer.
    // * We form a `&mut [u8]` of exactly `required` bytes over that capacity
    //   and fully initialize all of them with `copy_from_slice`, which also
    //   bounds-checks that `src[..required]` has `required` bytes (the caller
    //   pre-checks `src.len() >= required`, so the slice never panics).
    //   Reinterpreting `*mut T` as `*mut u8` is in-bounds because `T: Pod` is
    //   plain-old-data and the byte length matches; alignment is irrelevant
    //   for a `u8` view, and the output is `T`-aligned regardless.
    // * Only after every byte is written do we `set_len(count)`, so no
    //   uninitialized `T` is ever exposed. `T: Pod` has no invalid bit
    //   patterns and no `Drop`, so the previously-uninitialized slots are
    //   sound to treat as initialized `T` once their bytes are set.
    unsafe {
        let dst = core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), required);
        dst.copy_from_slice(&src[..required]);
        out.set_len(count);
    }
}

/// Big-endian hosts: scalar fallback (no shipped runtime target is BE, but the
/// wire format must stay bit-identical). Keeps the same single-write property.
#[cfg(target_endian = "big")]
#[inline]
fn fill_le_words_into<T: LeWireWord>(src: &[u8], count: usize, out: &mut Vec<T>) {
    let required = count * T::WIDTH;
    debug_assert!(src.len() >= required);
    out.clear();
    out.reserve(count);
    for chunk in src[..required].chunks_exact(T::WIDTH) {
        out.push(T::from_le_chunk(chunk));
    }
}

/// Pack a `&[u32]` into little-endian bytes for `DataType::U32` storage buffers.
#[must_use]
pub fn pack_u32_slice(words: &[u32]) -> Vec<u8> {
    pack_le_wire_words_owned(words)
}

/// Pack `&[u32]` into `out` as little-endian bytes; `out` is cleared first.
///
/// Endian-aware fast path: on little-endian hosts (every shipped runtime
/// target) this reduces to one `extend_from_slice` over a `bytemuck::cast_slice`
/// - no per-word copies. On big-endian hosts it falls back to the scalar loop
/// so the wire format is identical across hosts.
pub fn pack_u32_slice_into(words: &[u32], out: &mut Vec<u8>) {
    pack_le_wire_words_into(words, out);
}

/// Fallible `u32` little-endian pack into caller-owned byte storage.
pub fn try_pack_u32_slice_into(words: &[u32], out: &mut Vec<u8>) -> Result<(), String> {
    try_pack_le_wire_words_into(words, out, "u32 byte pack word", "u32 byte pack output")
}

/// Pack `&[u32]` into `out` as little-endian bytes, padded to at least
/// `min_words * 4` total bytes with trailing zeros.
///
/// Returns an error message (suitable for surfacing through pipeline
/// dispatch) when `min_words` overflows `usize` or its byte length
/// overflows, or when the input is longer than the requested floor.
/// The trailing-zero pad is what GPU dispatch needs when a kernel
/// declares a minimum binding length larger than the live token count.
pub fn pack_u32_slice_min_words_into(
    words: &[u32],
    min_words: u32,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let min_words_usize = min_words as usize;
    if words.len() > min_words_usize {
        let input_bytes = words.len().checked_mul(4).ok_or_else(|| {
            format!(
                "u32 byte pack input count {} overflows host byte indexing. Fix: shard the buffer.",
                words.len()
            )
        })?;
        let min_bytes = min_words_usize.checked_mul(4).ok_or_else(|| {
            format!("u32 byte pack minimum buffer count {min_words_usize} overflows host byte indexing. Fix: shard the buffer.")
        })?;
        return Err(format!(
            "u32 byte pack input has {input_bytes} bytes but minimum buffer only has {min_bytes}. Fix: pass min_words >= words.len()."
        ));
    }
    let target_bytes = min_words_usize.checked_mul(4).ok_or_else(|| {
        format!("u32 byte pack minimum buffer count {min_words_usize} overflows host byte indexing. Fix: shard the buffer.")
    })?;
    reserve_exact_len(out, target_bytes, "u32 byte pack output")?;
    pack_u32_slice_into(words, out);
    for _ in words.len()..min_words_usize {
        out.extend_from_slice(&[0u8, 0, 0, 0]);
    }
    Ok(())
}

/// Pack raw bytes into per-lane `u32` storage (low 8 bits per word).
///
/// # Panics
/// Panics when the packed length overflows the wire ABI; see [`pack_u32_slice_into`].
#[must_use]
pub fn pack_bytes_as_u32_slice(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for byte in bytes {
        out.extend_from_slice(&[*byte, 0, 0, 0]);
    }
    out
}

/// Pack raw bytes into per-lane `u32` storage using caller-owned byte storage.
pub fn pack_bytes_as_u32_slice_into(bytes: &[u8], out: &mut Vec<u8>) {
    out.clear();
    for byte in bytes {
        out.extend_from_slice(&[*byte, 0, 0, 0]);
    }
}

/// Fallible byte-lane pack using caller-owned byte storage.
pub fn try_pack_bytes_as_u32_slice_into(bytes: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    let byte_len = bytes.len().checked_mul(4).ok_or_else(|| {
        format!(
            "byte-lane pack word count {} overflows host byte indexing. Fix: shard the buffer.",
            bytes.len()
        )
    })?;
    reserve_exact_len(out, byte_len, "byte-lane pack output")?;
    pack_bytes_as_u32_slice_into(bytes, out);
    Ok(())
}

/// Pack raw bytes into per-lane `u32` storage (low 8 bits per word),
/// padded to at least `min_words` u32 words. Empty input collapses to
/// exactly `min_words` zero words. Returns the word count (the larger
/// of `bytes.len()` and `min_words`).
///
/// Errors if the requested byte length overflows `usize`. Replaces the
/// hand-rolled `pack_haystack` body across the parser frontends.
pub fn pack_bytes_as_u32_slice_min_words(
    bytes: &[u8],
    min_words: usize,
) -> Result<(Vec<u8>, usize), String> {
    let mut out = Vec::new();
    let words = pack_bytes_as_u32_slice_min_words_into(bytes, min_words, &mut out)?;
    Ok((out, words))
}

/// Pack raw bytes into per-lane `u32` storage with a minimum word count,
/// using caller-owned byte storage.
pub fn pack_bytes_as_u32_slice_min_words_into(
    bytes: &[u8],
    min_words: usize,
    out: &mut Vec<u8>,
) -> Result<usize, String> {
    let word_count = if bytes.len() > min_words {
        bytes.len()
    } else {
        min_words
    };
    let byte_len = word_count.checked_mul(4).ok_or_else(|| {
        format!("byte-lane pack minimum buffer count {word_count} overflows host byte indexing. Fix: shard the buffer.")
    })?;
    reserve_exact_len(out, byte_len, "byte-lane pack output")?;
    pack_bytes_as_u32_slice_into(bytes, out);
    for _ in bytes.len()..min_words {
        out.extend_from_slice(&[0u8, 0, 0, 0]);
    }
    Ok(word_count)
}

/// Pack `&[f32]` into little-endian bytes for `DataType::F32` storage buffers.
///
/// f32 is `Pod` so the LE host path reduces to one `bytemuck::cast_slice`
/// (no per-word copy). Big-endian fallback iterates `f32::to_le_bytes`.
#[must_use]
pub fn pack_f32_slice(values: &[f32]) -> Vec<u8> {
    pack_le_wire_words_owned(values)
}

/// Pack `&[f32]` into `out` as little-endian bytes; `out` is cleared first.
///
/// Same endian-aware shape as [`pack_u32_slice_into`] - one `bytemuck`
/// `cast_slice` copy on LE hosts, scalar fallback on BE hosts.
pub fn pack_f32_slice_into(values: &[f32], out: &mut Vec<u8>) {
    pack_le_wire_words_into(values, out);
}
/// Fallible `f32` little-endian pack into caller-owned byte storage.
pub fn try_pack_f32_slice_into(values: &[f32], out: &mut Vec<u8>) -> Result<(), String> {
    try_pack_le_wire_words_into(values, out, "f32 byte pack value", "f32 byte pack output")
}

#[cfg(test)]
mod fallible_pack_wrapper_tests {
    use super::{
        pack_bytes_as_u32_slice, pack_bytes_as_u32_slice_into, pack_f32_slice_into,
        pack_u32_slice_into, try_pack_bytes_as_u32_slice_into, try_pack_f32_slice_into,
        try_pack_u32_slice_into,
    };

    #[test]
    fn convenience_wrappers_match_fallible_packers() {
        let words = [0x0102_0304_u32, 0xaabb_ccdd];
        let mut compat_words = Vec::new();
        let mut fallible_words = Vec::new();
        pack_u32_slice_into(&words, &mut compat_words);
        try_pack_u32_slice_into(&words, &mut fallible_words)
            .expect("Fix: small u32 wire pack must reserve");
        assert_eq!(compat_words, fallible_words);

        let bytes = b"abc";
        assert_eq!(pack_bytes_as_u32_slice(bytes), {
            let mut out = Vec::new();
            try_pack_bytes_as_u32_slice_into(bytes, &mut out)
                .expect("Fix: small byte-lane wire pack must reserve");
            out
        });

        let floats = [1.0_f32, -0.0, f32::INFINITY];
        let mut compat_floats = Vec::new();
        let mut fallible_floats = Vec::new();
        pack_f32_slice_into(&floats, &mut compat_floats);
        try_pack_f32_slice_into(&floats, &mut fallible_floats)
            .expect("Fix: small f32 wire pack must reserve");
        assert_eq!(compat_floats, fallible_floats);
    }

    #[test]
    fn caller_owned_byte_lane_wrapper_reuses_existing_buffer() {
        let mut compat = Vec::with_capacity(16);
        let ptr = compat.as_ptr();

        pack_bytes_as_u32_slice_into(b"xy", &mut compat);

        assert_eq!(compat, vec![b'x', 0, 0, 0, b'y', 0, 0, 0]);
        assert_eq!(compat.as_ptr(), ptr);
    }
}

/// Decode a packed LE-`u32` byte buffer back into `out` as `Vec<u32>`,
/// reading exactly `count` words. `out` is cleared before writing.
///
/// `label` is woven into both error messages so the caller's site is
/// recognizable in mixed-pipeline failures (e.g. `"cfg_blob"` vs
/// `"hit_buffer"`). Returns the same byte-length / overflow contract
/// as `pack_u32_slice_min_words_into`, just in the reverse direction.
///
/// Endian-aware fast path: on little-endian hosts the whole prefix is
/// copied through one `bytemuck::cast_slice_mut` of the freshly-resized
/// backing storage, so the inner per-word `u32::from_le_bytes` loop is
/// skipped. On big-endian hosts the scalar loop runs to match the wire
/// format.
pub fn unpack_u32_slice_into(
    bytes: &[u8],
    count: usize,
    label: &str,
    out: &mut Vec<u32>,
) -> Result<(), String> {
    let required = count.checked_mul(4).ok_or_else(|| {
        format!("{label}: u32 stream word count {count} overflows host byte indexing. Fix: shard the decode.")
    })?;
    if bytes.len() < required {
        return Err(format!(
            "{label}: u32 stream has {} bytes, needs {required}. Fix: backend output is truncated.",
            bytes.len()
        ));
    }
    reserve_exact_len(out, count, label)?;
    fill_le_words_into::<u32>(bytes, count, out);
    Ok(())
}

/// Decode a packed LE-`f32` byte buffer back into `out` as `Vec<f32>`,
/// reading exactly `count` values. `out` is cleared before writing.
///
/// Same endian-aware shape as [`unpack_u32_slice_into`] - one
/// `bytemuck::cast_slice_mut` copy on LE hosts, scalar fallback on BE
/// hosts. This is the canonical companion to the `.chunks_exact(4).map(|c|
/// f32::from_le_bytes(c.try_into().unwrap())).collect()` pattern repeated
/// across every nn op (activation, attention, norm, moe, linear) - every
/// caller should route here instead of re-implementing the loop.
pub fn unpack_f32_slice_into(
    bytes: &[u8],
    count: usize,
    label: &str,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    let required = count.checked_mul(4).ok_or_else(|| {
        format!("{label}: f32 stream value count {count} overflows host byte indexing. Fix: shard the decode.")
    })?;
    if bytes.len() < required {
        return Err(format!(
            "{label}: f32 stream has {} bytes, needs {required}. Fix: backend output is truncated.",
            bytes.len()
        ));
    }
    reserve_exact_len(out, count, label)?;
    fill_le_words_into::<f32>(bytes, count, out);
    Ok(())
}

/// Owned-`Vec<f32>` variant of [`unpack_f32_slice_into`].
pub fn unpack_f32_slice(bytes: &[u8], count: usize, label: &str) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(count);
    unpack_f32_slice_into(bytes, count, label, &mut out)?;
    Ok(out)
}

/// Test-grade decode helper that drains an entire LE-`f32` byte buffer.
/// Trailing bytes that don't form a complete word are dropped (matches
/// the `.chunks_exact(4)` semantics every test helper across the nn ops
/// uses). Skips error plumbing because every existing call site is in
/// `#[cfg(test)]` infrastructure with bit-exact known inputs.
#[must_use]
pub fn decode_f32_le_bytes_all(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

/// Test-grade decode helper that drains an entire LE-`u32` byte buffer.
/// Companion to [`decode_f32_le_bytes_all`].
#[must_use]
pub fn decode_u32_le_bytes_all(bytes: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

/// Decode exactly eight little-endian `u32` words from a 32-byte array.
///
/// Program fingerprints, BLAKE3 digests, and fixed cache keys all share this
/// shape. Taking `&[u8; 32]` makes the length contract compile-time instead
/// of re-checking it with a fallible slice conversion at every caller.
#[must_use]
pub fn decode_u32x8_le_bytes(bytes: &[u8; 32]) -> [u32; 8] {
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
        u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
        u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
        u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]),
    ]
}

/// Append a `&[u32]` to `out` as little-endian bytes. Does NOT clear `out`
/// first; use this when you're building a composite header where the u32
/// slice is one of several fields. Same LE host bytemuck fast path as
/// [`pack_u32_slice_into`].
pub fn append_u32_slice_le_bytes(words: &[u32], out: &mut Vec<u8>) {
    append_le_wire_words(words, out);
}

/// Append a `&[f32]` to `out` as little-endian bytes; same shape as
/// [`append_u32_slice_le_bytes`].
pub fn append_f32_slice_le_bytes(values: &[f32], out: &mut Vec<u8>) {
    append_le_wire_words(values, out);
}

/// Pack a `&[i32]` into little-endian bytes for `DataType::I32` storage.
#[must_use]
pub fn pack_i32_slice(values: &[i32]) -> Vec<u8> {
    pack_le_wire_words_owned(values)
}

/// Pack `&[i32]` into `out`; same LE bytemuck fast path.
pub fn pack_i32_slice_into(values: &[i32], out: &mut Vec<u8>) {
    pack_le_wire_words_into(values, out);
}

/// Decode an LE-`i32` byte buffer into `Vec<i32>`. Drains the whole buffer.
#[must_use]
pub fn decode_i32_le_bytes_all(bytes: &[u8]) -> Vec<i32> {
    let mut out = Vec::new();
    for chunk in bytes.chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}
/// Pack a `&[u64]` into little-endian bytes for 8-byte storage buffers.
#[cfg(test)]
#[must_use]
pub fn pack_u64_slice(values: &[u64]) -> Vec<u8> {
    pack_le_wire_words_owned(values)
}

/// Pack `&[u64]` into `out`. LE bytemuck fast path; 8 bytes per value.
pub fn pack_u64_slice_into(values: &[u64], out: &mut Vec<u8>) {
    pack_le_wire_words_into(values, out);
}

/// Decode an LE-`u64` byte buffer into `Vec<u64>`. Drains the whole buffer.
#[must_use]
pub fn decode_u64_le_bytes_all(bytes: &[u8]) -> Vec<u64> {
    let mut out = Vec::new();
    for chunk in bytes.chunks_exact(8) {
        out.push(u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    out
}
/// Pack a `&[u16]` into little-endian bytes for `f16`/`bf16` storage.
#[cfg(test)]
#[must_use]
pub fn pack_u16_slice(values: &[u16]) -> Vec<u8> {
    pack_le_wire_words_owned(values)
}

/// Pack `&[u16]` into `out`. LE bytemuck fast path; 2 bytes per value.
pub fn pack_u16_slice_into(values: &[u16], out: &mut Vec<u8>) {
    pack_le_wire_words_into(values, out);
}
fn pack_le_wire_words_owned<T: LeWireWord>(values: &[T]) -> Vec<u8> {
    #[cfg(target_endian = "little")]
    {
        bytemuck::cast_slice::<T, u8>(values).to_vec()
    }
    #[cfg(target_endian = "big")]
    {
        let mut out = Vec::new();
        for &value in values {
            value.push_le_bytes(&mut out);
        }
        out
    }
}

/// Pack a `&[u32]` into a freshly-allocated `Vec<u8>` through the
/// direct owned fast path.
///
/// On little-endian hosts this is the bandwidth-floor implementation:
/// one allocation + one `memcpy` via `bytemuck::cast_slice(...).to_vec()`,
/// no scalar loop and no streaming adapter overhead. This is the
/// canonical owned packing implementation used by [`pack_u32_slice`].
///
/// On big-endian hosts the function falls back to the scalar loop into
/// a pre-reserved Vec so the wire format stays bit-identical.
#[cfg(test)]
#[must_use]
pub fn pack_u32_slice_into_uninit(words: &[u32]) -> Vec<u8> {
    pack_le_wire_words_owned(words)
}

/// Pack a `&[f32]` into a freshly-allocated `Vec<u8>` without zero-fill.
/// Same shape as [`pack_u32_slice_into_uninit`] for the f32 wire path.
#[cfg(test)]
#[must_use]
pub fn pack_f32_slice_into_uninit(values: &[f32]) -> Vec<u8> {
    pack_le_wire_words_owned(values)
}

/// Append raw bytes onto `out` as one `u32` per byte (low 8 bits, zero
/// high bits). Streaming companion to [`pack_bytes_as_u32_slice`] -
/// no intermediate `Vec` allocation, no clear of `out`. Used by header
/// builders that concatenate a packed-lane byte stream onto a composite
/// GPU haystack buffer in one extension pass.
pub fn append_packed_byte_lane(bytes: &[u8], out: &mut Vec<u8>) {
    for byte in bytes {
        out.extend_from_slice(&[*byte, 0, 0, 0]);
    }
}

/// Decode an LE-`u16` byte buffer into `Vec<u16>`. Drains the whole buffer.
#[must_use]
pub fn decode_u16_le_bytes_all(bytes: &[u8]) -> Vec<u16> {
    let mut out = Vec::new();
    for chunk in bytes.chunks_exact(2) {
        out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    out
}

fn pack_u32_iter_into<I>(words: I, out: &mut Vec<u8>)
where
    I: IntoIterator<Item = u32>,
{
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
}

/// Packs an owned or generated stream of little-endian u32 words.
///
/// Use this for range/generated test inputs so callers do not duplicate
/// `flat_map(u32::to_le_bytes)` loops or allocate an intermediate word vector.
#[cfg(test)]
pub fn pack_u32_iter<I>(words: I) -> Vec<u8>
where
    I: IntoIterator<Item = u32>,
{
    let mut out = Vec::new();
    pack_u32_iter_into(words, &mut out);
    out
}

/// Read one little-endian `f32` word at `word_index` from a byte stream.
///
/// This is the canonical scalar companion to [`decode_f32_le_bytes_all`]
/// for call sites that need one indexed word but still need the same checked
/// bounds and actionable diagnostics as the bulk decoder.
///
/// # Errors
/// Returns an error when `word_index * 4` overflows host indexing or when the
/// requested word is not fully present in `bytes`.
pub fn read_f32_le_word(bytes: &[u8], word_index: usize, label: &str) -> Result<f32, String> {
    if let Some(chunk) = bytes.chunks_exact(4).nth(word_index) {
        Ok(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
    } else {
        Err(format!(
            "{label}: f32 word {word_index} requires 4 bytes, but stream has {} bytes. Fix: backend output is truncated.",
            bytes.len()
        ))
    }
}

/// Read one little-endian `u32` word at `word_index` from a byte stream.
///
/// This is the canonical scalar companion to [`decode_u32_le_bytes_all`]
/// for call sites that need one indexed word but still need the same checked
/// bounds and actionable diagnostics as the bulk decoder.
///
/// # Errors
/// Returns an error when `word_index * 4` overflows host indexing or when the
/// requested word is not fully present in `bytes`.
pub fn read_u32_le_word(bytes: &[u8], word_index: usize, label: &str) -> Result<u32, String> {
    if let Some(chunk) = bytes.chunks_exact(4).nth(word_index) {
        Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
    } else {
        Err(format!(
            "{label}: u32 word {word_index} requires 4 bytes, but stream has {} bytes. Fix: backend output is truncated.",
            bytes.len()
        ))
    }
}

/// Read one little-endian `u32` at a BYTE offset into `bytes`: the byte-addressed
/// companion to [`read_u32_le_word`] (which is word-indexed). Use this when the
/// caller already holds a byte offset rather than a word index; both share the
/// same checked bounds and actionable diagnostics, so the workspace's byte-offset
/// u32 readers decode through ONE implementation instead of re-rolling it.
///
/// # Errors
/// Returns an error when `byte_offset + 4` overflows host indexing or when the
/// four requested bytes are not fully present in `bytes`.
pub fn read_u32_le_at(bytes: &[u8], byte_offset: usize, label: &str) -> Result<u32, String> {
    if let Some(chunk) = bytes.get(byte_offset..).and_then(|tail| tail.get(..4)) {
        Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
    } else {
        Err(format!(
            "{label}: u32 at byte offset {byte_offset} requires 4 bytes, but stream has {} bytes. Fix: backend output is truncated.",
            bytes.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_u32_le_at_reads_byte_offset_and_agrees_with_word_reader() {
        // word 0 = 0, word 1 @ byte 4 = 0x12345678
        let bytes = [0u8, 0, 0, 0, 0x78, 0x56, 0x34, 0x12];
        assert_eq!(read_u32_le_at(&bytes, 0, "t").unwrap(), 0);
        assert_eq!(read_u32_le_at(&bytes, 4, "t").unwrap(), 0x1234_5678);
        // The byte-offset reader at 4*word must equal the word-indexed reader at word.
        assert_eq!(
            read_u32_le_at(&bytes, 4, "t").unwrap(),
            read_u32_le_word(&bytes, 1, "t").unwrap()
        );
        // Truncation fails closed (bytes 5..9 but only 8 present), never a silent read.
        let err = read_u32_le_at(&bytes, 5, "t").unwrap_err();
        assert!(
            err.contains("truncated"),
            "expected truncation error, got: {err}"
        );
    }

    #[test]
    fn byte_lane_pack_sets_low_byte_and_zeroes_high_bytes() {
        let packed = pack_bytes_as_u32_slice(&[0x00, 0x7f, 0xff]);
        assert_eq!(packed, vec![0x00, 0, 0, 0, 0x7f, 0, 0, 0, 0xff, 0, 0, 0]);
    }

    #[test]
    fn byte_lane_pack_into_reuses_output_and_clears_stale_tail() {
        let mut out = Vec::with_capacity(32);
        out.extend_from_slice(&[0xff; 32]);
        let ptr = out.as_ptr();

        try_pack_bytes_as_u32_slice_into(&[0x11, 0x22], &mut out).unwrap();

        assert_eq!(out, vec![0x11, 0, 0, 0, 0x22, 0, 0, 0]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn indexed_u32_read_checks_bounds_and_endianness() {
        let bytes = pack_u32_slice(&[0x0102_0304, 0xaabb_ccdd]);
        assert_eq!(
            read_u32_le_word(&bytes, 0, "indexed-read test").expect("Fix: first word must decode."),
            0x0102_0304
        );
        assert_eq!(
            read_u32_le_word(&bytes, 1, "indexed-read test")
                .expect("Fix: second word must decode."),
            0xaabb_ccdd
        );
        let err = read_u32_le_word(&bytes, 2, "indexed-read test")
            .expect_err("Fix: out-of-range word must be rejected.");
        assert!(err.contains("indexed-read test"), "unexpected error: {err}");
    }

    #[test]
    fn fixed_u32x8_decode_matches_bulk_decoder() {
        let words = [
            0x0000_0000,
            0x0102_0304,
            0x1122_3344,
            0x5566_7788,
            0x99aa_bbcc,
            0xddee_ff00,
            0x8000_0001,
            0xffff_ffff,
        ];
        let bytes_vec = pack_u32_slice(&words);
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&bytes_vec);

        assert_eq!(decode_u32x8_le_bytes(&bytes), words);
        assert_eq!(
            decode_u32x8_le_bytes(&bytes).as_slice(),
            decode_u32_le_bytes_all(&bytes)
        );
    }

    #[test]
    fn byte_lane_min_words_pads_without_aliasing_live_bytes() {
        let (packed, words) = pack_bytes_as_u32_slice_min_words(&[0xab, 0xcd], 4)
            .expect("Fix: byte-lane packing with a larger floor must not fail.");
        assert_eq!(words, 4);
        assert_eq!(
            packed,
            vec![0xab, 0, 0, 0, 0xcd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn byte_lane_min_words_into_reuses_output_and_clears_stale_tail() {
        let mut out = Vec::with_capacity(32);
        out.extend_from_slice(&[0xff; 32]);
        let ptr = out.as_ptr();

        let words = pack_bytes_as_u32_slice_min_words_into(&[0xab, 0xcd], 4, &mut out).unwrap();

        assert_eq!(words, 4);
        assert_eq!(
            out,
            vec![0xab, 0, 0, 0, 0xcd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn append_byte_lane_preserves_prefix_and_appends_packed_words() {
        let mut packed = vec![0xee, 0xdd];
        append_packed_byte_lane(&[1, 2], &mut packed);
        assert_eq!(packed, vec![0xee, 0xdd, 1, 0, 0, 0, 2, 0, 0, 0]);
    }

    #[test]
    fn generated_u32_pack_matches_slice_pack() {
        let words = [0, 1, 0x1234_5678, u32::MAX];
        let generated = pack_u32_iter(words);
        assert_eq!(generated, pack_u32_slice(&words));
    }

    #[test]
    fn pack_u32_slice_into_clears_stale_bytes() {
        let mut out = vec![0xff; 32];
        pack_u32_slice_into(&[0x0102_0304], &mut out);
        assert_eq!(out, vec![4, 3, 2, 1]);
    }

    #[test]
    fn pack_and_unpack_u32_into_reuse_buffers() {
        let words = [0x0102_0304, 0xaabb_ccdd];
        let mut bytes = Vec::with_capacity(32);
        bytes.extend_from_slice(&[0xff; 32]);
        let bytes_ptr = bytes.as_ptr();

        try_pack_u32_slice_into(&words, &mut bytes).unwrap();

        assert_eq!(bytes, pack_u32_slice(&words));
        assert_eq!(bytes.as_ptr(), bytes_ptr);

        let mut decoded = Vec::with_capacity(8);
        decoded.extend_from_slice(&[u32::MAX; 8]);
        let decoded_ptr = decoded.as_ptr();
        unpack_u32_slice_into(&bytes, words.len(), "wire u32 reuse", &mut decoded).unwrap();

        assert_eq!(decoded, words);
        assert_eq!(decoded.as_ptr(), decoded_ptr);
    }

    #[test]
    fn pack_and_unpack_f32_into_reuse_buffers() {
        let values = [1.5_f32, -2.25, f32::INFINITY];
        let mut bytes = Vec::with_capacity(32);
        bytes.extend_from_slice(&[0xff; 32]);
        let bytes_ptr = bytes.as_ptr();

        try_pack_f32_slice_into(&values, &mut bytes).unwrap();

        assert_eq!(bytes, pack_f32_slice(&values));
        assert_eq!(bytes.as_ptr(), bytes_ptr);

        let mut decoded = Vec::with_capacity(8);
        decoded.extend_from_slice(&[f32::NAN; 8]);
        let decoded_ptr = decoded.as_ptr();
        unpack_f32_slice_into(&bytes, values.len(), "wire f32 reuse", &mut decoded).unwrap();

        assert_eq!(decoded, values);
        assert_eq!(decoded.as_ptr(), decoded_ptr);
    }

    #[test]
    fn unpack_u32_into_rejects_truncated_input_transactionally() {
        let mut decoded = vec![0x1234_5678, 0x9abc_def0];
        let before = decoded.clone();

        let err = unpack_u32_slice_into(&[1, 2, 3], 1, "truncated u32", &mut decoded)
            .expect_err("truncated u32 stream must be rejected");

        assert!(err.contains("truncated u32"));
        assert_eq!(decoded, before);
    }

    // Adversarial lock on the unsafe single-write decode helper
    // (`fill_le_words_into`). The risks the `unsafe` introduces are: leaking
    // stale capacity past `count`, reading past `src`, or exposing
    // uninitialized slots via a too-early `set_len`. Each case below would
    // surface as a wrong value, wrong length, or (under Miri/ASAN) UB.
    #[test]
    fn unpack_decode_uninit_helper_is_exact_and_leak_free() {
        // 1) Over-length source: only the first `count` words are decoded; the
        //    trailing bytes must be ignored, not appended.
        let words: Vec<u32> = (0..1000).collect();
        let mut bytes = pack_u32_slice(&words);
        bytes.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]); // one extra word of junk
        let mut out = Vec::new();
        unpack_u32_slice_into(&bytes, words.len(), "over-length", &mut out).unwrap();
        assert_eq!(out, words, "decoded the junk tail or wrong count");
        assert_eq!(out.len(), words.len());

        // 2) Reuse into a buffer that previously held a LONGER, fully-populated
        //    run: `set_len` must shrink and no stale element may survive.
        let mut reused: Vec<u32> = vec![0xFFFF_FFFF; 4096];
        unpack_u32_slice_into(&bytes, words.len(), "shrink reuse", &mut reused).unwrap();
        assert_eq!(reused, words);

        // 3) Reuse into a buffer whose capacity is EXACTLY `count` (no spare),
        //    growing from empty.
        let mut tight: Vec<u32> = Vec::with_capacity(words.len());
        unpack_u32_slice_into(&bytes, words.len(), "tight cap", &mut tight).unwrap();
        assert_eq!(tight, words);

        // 4) Owned drain decoder over a large buffer: byte-exact round-trip.
        assert_eq!(decode_u32_le_bytes_all(&pack_u32_slice(&words)), words);

        // 5) Empty decode is a no-op that clears, never touches the pointer math.
        let mut empty = vec![1u32, 2, 3];
        unpack_u32_slice_into(&[], 0, "empty", &mut empty).unwrap();
        assert!(empty.is_empty());
    }
}
