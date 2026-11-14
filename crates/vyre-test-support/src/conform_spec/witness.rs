//! Witness set enumeration.

use vyre_spec::DataType;

/// Deterministic witness inventory for one frozen semantic data type.
pub trait WitnessSet {
    /// Host value represented by this witness inventory.
    type Value;

    /// Stable semantic data type certified by this inventory.
    const DATA_TYPE: DataType;

    /// Enumerate witnesses in canonical order.
    fn enumerate() -> Vec<Self::Value>;

    /// Fingerprint the canonical witness byte stream.
    fn fingerprint_canonical() -> [u8; 32];
}

/// Canonical u32 witness set: boundary values + deterministic pseudo-random samples.
pub struct U32Witness;

impl U32Witness {
    /// Enumerate the canonical u32 witness set.
    pub fn enumerate() -> Vec<u32> {
        const BOUNDARY: [u32; 12] = [
            0,
            1,
            2,
            3,
            u32::MAX,
            u32::MAX - 1,
            0x8000_0000,
            0x7FFF_FFFF,
            0xAAAA_AAAA,
            0x5555_5555,
            0xDEAD_BEEF,
            0xCAFE_F00D,
        ];
        let mut out = BOUNDARY.to_vec();
        let seed = *blake3::hash(b"u32-witness-v1").as_bytes();
        let [s0, s1, s2, s3, s4, s5, s6, s7, ..] = seed;
        let mut state = u64::from_le_bytes([s0, s1, s2, s3, s4, s5, s6, s7]);
        for _ in 0..24 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let z = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let z = z ^ (z >> 31);
            out.push((z ^ (z >> 32)) as u32);
        }
        out
    }

    /// Canonical blake3 fingerprint for this witness set (little-endian encoding).
    #[must_use]
    pub fn fingerprint_canonical() -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for v in Self::enumerate() {
            hasher.update(&v.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

impl WitnessSet for U32Witness {
    type Value = u32;

    const DATA_TYPE: DataType = DataType::U32;

    fn enumerate() -> Vec<Self::Value> {
        U32Witness::enumerate()
    }

    fn fingerprint_canonical() -> [u8; 32] {
        U32Witness::fingerprint_canonical()
    }
}
