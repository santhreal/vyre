pub(crate) fn u32_bytes(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

pub(crate) const MATMUL_2X2_EXPECTED_BYTES: [u8; 16] = [
    0x13, 0x00, 0x00, 0x00, // 19
    0x16, 0x00, 0x00, 0x00, // 22
    0x2b, 0x00, 0x00, 0x00, // 43
    0x32, 0x00, 0x00, 0x00, // 50
];

#[cfg(test)]
#[must_use]
pub(crate) fn matmul_2x2_expected() -> Vec<Vec<Vec<u8>>> {
    vec![vec![MATMUL_2X2_EXPECTED_BYTES.to_vec()]]
}

pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    vyre_primitives::wire::pack_f32_slice(values)
}

#[cfg(test)]
pub(crate) fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    vyre_primitives::wire::decode_f32_le_bytes_all(bytes)
}

#[cfg(test)]
pub(crate) fn decode_f32_one(bytes: &[u8]) -> f32 {
    match try_decode_f32_one(bytes) {
        Ok(value) => value,
        Err(_) => f32::NAN,
    }
}

#[cfg(test)]
pub(crate) fn try_decode_f32_one(bytes: &[u8]) -> Result<f32, String> {
    vyre_primitives::wire::read_f32_le_word(bytes, 0, "f32 scalar fixture output")
}

#[cfg(test)]
pub(crate) fn decode_u32_one(bytes: &[u8]) -> u32 {
    match try_decode_u32_one(bytes) {
        Ok(value) => value,
        Err(_) => u32::MAX,
    }
}

#[cfg(test)]
pub(crate) fn try_decode_u32_one(bytes: &[u8]) -> Result<u32, String> {
    vyre_primitives::wire::read_u32_le_word(bytes, 0, "u32 scalar fixture output")
}

#[cfg(test)]
pub(crate) fn bytes_to_u32(slice: &[u8]) -> Vec<u32> {
    vyre_primitives::wire::decode_u32_le_bytes_all(slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2x2_expected_bytes_identity() {
        let constructed = u32_bytes(&[19, 22, 43, 50]);
        assert_eq!(constructed, MATMUL_2X2_EXPECTED_BYTES);
    }

    #[test]
    fn test_round_trip() {
        let original = vec![1, 2, 3, 0xFFFFFFFF, 0x12345678];
        let bytes = u32_bytes(&original);
        let back = bytes_to_u32(&bytes);
        assert_eq!(original, back);
    }

    #[test]
    fn test_empty_input() {
        let original: Vec<u32> = vec![];
        let bytes = u32_bytes(&original);
        assert!(bytes.is_empty());
        let back = bytes_to_u32(&bytes);
        assert!(back.is_empty());
    }

    #[test]
    fn test_f32_bit_exact_pack() {
        let bytes = f32_bytes(&[1.0, -0.0, f32::INFINITY, f32::NAN]);
        let unpacked =
            vyre_primitives::wire::unpack_f32_slice(&bytes, 4, "test_f32_bit_exact_pack")
                .expect("Fix: f32 test fixture pack must round-trip.");
        assert_eq!(unpacked[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(unpacked[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(unpacked[2].to_bits(), f32::INFINITY.to_bits());
        assert!(unpacked[3].is_nan());
    }
}
