//! Fixtures the adversarial gates in this crate share.
//!
//! One case loop and one packer, so no suite writes its own.

/// Little-endian u32 packing, the same shipped packer every other suite uses.
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

macro_rules! adversarial_vec_u32_cases {
    ($($name:ident: $input:expr, $param:expr => $expected:expr, $message:expr;)+) => {
        $(
            #[test]
            fn $name() {
                let input = $input;
                let param = $param;
                let expected = $expected;
                let actual = cpu_ref(&input, param);
                assert_eq!(actual, expected, "{}", $message);
            }
        )+
    };
}
