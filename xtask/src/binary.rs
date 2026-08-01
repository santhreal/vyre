//! Bounded binary parsing helpers shared by xtask evidence readers.

/// Find the first exact occurrence of `needle` in `haystack`.
pub(crate) fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Read one little-endian `u32` and advance `cursor` only on success.
pub(crate) fn read_u32_le(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
    let end = cursor.checked_add(4)?;
    let raw: [u8; 4] = bytes.get(*cursor..end)?.try_into().ok()?;
    *cursor = end;
    Some(u32::from_le_bytes(raw))
}
