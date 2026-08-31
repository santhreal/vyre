//! Packed haystack layout the scan kernels in this crate consume.

/// Zero-pad `bytes` up to a 4-byte boundary so the result uploads as a dense
/// `u32` lane buffer (four haystack bytes per word).
///
/// This is the packed haystack layout the scan kernels in this crate consume.
/// It is distinct from `vyre_primitives::wire::pack_bytes_as_u32_slice`, which
/// expands one byte per `u32` word.
#[must_use]
pub fn pack_haystack_u32(bytes: &[u8]) -> Vec<u8> {
    let mut packed = bytes.to_vec();
    packed.extend(std::iter::repeat_n(0, (4 - bytes.len() % 4) % 4));
    packed
}
