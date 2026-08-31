//! Little-endian word access into a byte buffer a device shares with a host.
//!
//! A protocol suite states a ring, control, or debug buffer word by word, and
//! eight suites carried the same four-line writer. A copy that transposed the
//! offset arithmetic would still compile and would state a different buffer
//! than the one the runtime reads.

/// Write `value` into the word at `word_idx`.
///
/// # Panics
///
/// Panics when the word lies outside `bytes`, which is a fixture addressing a
/// buffer smaller than the layout it states.
pub fn write_word(bytes: &mut [u8], word_idx: usize, value: u32) {
    let offset = word_idx * 4;
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// The word at `word_idx`.
///
/// # Panics
///
/// Panics when the word lies outside `bytes`.
#[must_use]
pub fn read_word(bytes: &[u8], word_idx: usize) -> u32 {
    let offset = word_idx * 4;
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("Fix: a four-byte slice is a u32 word; state a buffer that holds the word."),
    )
}
