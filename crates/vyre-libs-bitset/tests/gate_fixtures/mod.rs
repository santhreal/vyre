//! Fixtures the adversarial gates in this crate share.
//!
//! One packer, so no suite writes its own. The case loop is
//! `vyre_test_support::adversarial_cpu_ref_cases!`.

/// Little-endian u32 packing, the same shipped packer every other suite uses.
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;
