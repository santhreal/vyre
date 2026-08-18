//! Sequential mathematical witnesses for text processing, UTF-8 classification, and information metrics.

/// Sequential mathematical witness for byte histogram (256 bins).
#[must_use]
pub fn byte_histogram_witness(input: &[u8]) -> [u32; 256] {
    let mut counts = [0u32; 256];
    for &b in input {
        counts[b as usize] += 1;
    }
    counts
}

/// Sequential mathematical witness for character class mapping.
#[must_use]
pub fn char_class_witness(input: &[u8], table: &[u32; 256]) -> Vec<u32> {
    input.iter().map(|&b| table[b as usize]).collect()
}

/// Sequential mathematical witness for each byte's zero-based line index.
#[must_use]
pub fn line_index_witness(input: &[u8]) -> Vec<u32> {
    let mut line = 0u32;
    input
        .iter()
        .enumerate()
        .map(|(offset, &byte)| {
            let index = line;
            if byte == b'\n'
                || (byte == b'\r' && input.get(offset + 1).is_some_and(|&next| next != b'\n'))
            {
                line = line.wrapping_add(1);
            }
            index
        })
        .collect()
}

/// Sequential mathematical witness for UTF-8 shape counts (ASCII, 2-byte, 3-byte, 4-byte).
#[must_use]
pub fn utf8_shape_counts_witness(input: &[u8]) -> [u32; 4] {
    let mut counts = [0u32; 4];
    let mut i = 0;
    while i < input.len() {
        let b = input[i];
        if b < 0x80 {
            counts[0] += 1;
            i += 1;
        } else if (b & 0xE0) == 0xC0 {
            counts[1] += 1;
            i += 2;
        } else if (b & 0xF0) == 0xE0 {
            counts[2] += 1;
            i += 3;
        } else if (b & 0xF8) == 0xF0 {
            counts[3] += 1;
            i += 4;
        } else {
            i += 1;
        }
    }
    counts
}

/// Count observed and expected UTF-8 continuation bytes from a byte histogram.
#[must_use]
pub fn utf8_histogram_shape_counts_witness(histogram: &[u32; 256]) -> (u32, u32) {
    let continuation = histogram[0x80..0xC0]
        .iter()
        .fold(0_u32, |sum, &count| sum.saturating_add(count));
    let expected = histogram[0xC2..0xE0]
        .iter()
        .fold(0_u32, |sum, &count| sum.saturating_add(count));
    let expected = histogram[0xE0..0xF0].iter().fold(expected, |sum, &count| {
        sum.saturating_add(count.saturating_mul(2))
    });
    let expected = histogram[0xF0..0xF5].iter().fold(expected, |sum, &count| {
        sum.saturating_add(count.saturating_mul(3))
    });
    (continuation, expected)
}

/// Classify an encoding from a 256-bin byte histogram.
#[must_use]
pub fn encoding_classify_histogram_witness(histogram: &[u32; 256], count: u32) -> u32 {
    const ASCII: u32 = 0;
    const UTF8: u32 = 1;
    const UTF16_LE: u32 = 2;
    const ISO_8859_1: u32 = 4;
    if count == 0 {
        return ASCII;
    }
    let null_count = histogram[0];
    let ascii_count = histogram[0..128]
        .iter()
        .fold(0_u32, |sum, &value| sum.saturating_add(value));
    let high_count = count.saturating_sub(ascii_count);
    if null_count > count / 8 {
        return UTF16_LE;
    }
    if high_count == 0 {
        return ASCII;
    }
    let (continuation, expected) = utf8_histogram_shape_counts_witness(histogram);
    let tolerance = count.saturating_add(19) / 20;
    if continuation.abs_diff(expected) < tolerance {
        UTF8
    } else {
        ISO_8859_1
    }
}

/// Sequential UTF-8 classification witness for the text validator ABI.
#[must_use]
pub fn utf8_validate_witness(source: &[u8]) -> Vec<u32> {
    const ASCII: u32 = 0;
    const LEAD_2: u32 = 1;
    const LEAD_3: u32 = 2;
    const LEAD_4: u32 = 3;
    const CONT: u32 = 4;
    const INVALID: u32 = 5;
    let is_continuation = |byte: u8| (0x80..=0xbf).contains(&byte);
    let mut output = vec![INVALID; source.len()];
    let mut index = 0;
    while index < source.len() {
        let first = source[index];
        if first <= 0x7f {
            output[index] = ASCII;
            index += 1;
            continue;
        }
        if (0xc2..=0xdf).contains(&first)
            && source.get(index + 1).copied().is_some_and(is_continuation)
        {
            output[index] = LEAD_2;
            output[index + 1] = CONT;
            index += 2;
            continue;
        }
        if index + 2 < source.len() {
            let second = source[index + 1];
            let third = source[index + 2];
            let second_valid = match first {
                0xe0 => (0xa0..=0xbf).contains(&second),
                0xe1..=0xec | 0xee..=0xef => is_continuation(second),
                0xed => (0x80..=0x9f).contains(&second),
                _ => false,
            };
            if second_valid && is_continuation(third) {
                output[index] = LEAD_3;
                output[index + 1] = CONT;
                output[index + 2] = CONT;
                index += 3;
                continue;
            }
        }
        if index + 3 < source.len() {
            let second = source[index + 1];
            let third = source[index + 2];
            let fourth = source[index + 3];
            let second_valid = match first {
                0xf0 => (0x90..=0xbf).contains(&second),
                0xf1..=0xf3 => is_continuation(second),
                0xf4 => (0x80..=0x8f).contains(&second),
                _ => false,
            };
            if second_valid && is_continuation(third) && is_continuation(fourth) {
                output[index] = LEAD_4;
                output[index + 1] = CONT;
                output[index + 2] = CONT;
                output[index + 3] = CONT;
                index += 4;
                continue;
            }
        }
        index += 1;
    }
    output
}

/// Shannon entropy in bits per byte.
#[must_use]
pub fn shannon_entropy_bits_per_byte_witness(bytes: &[u8]) -> f32 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0_u32; 256];
    for &byte in bytes {
        counts[byte as usize] += 1;
    }
    let length = bytes.len() as f64;
    -counts
        .iter()
        .filter(|&&count| count != 0)
        .map(|&count| {
            let probability = f64::from(count) / length;
            probability * probability.log2()
        })
        .sum::<f64>() as f32
}
