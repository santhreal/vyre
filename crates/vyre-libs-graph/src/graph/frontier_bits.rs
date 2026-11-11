//! Packed-bitset addressing skeleton for graph kernels, delegating downward to
//! [`vyre_libs_builder::builder::csr`].

pub(in crate::graph) use vyre_libs_builder::builder::csr::{
    active_source_lane, bind_bit_address, bind_word, bit_is_set, set_bit, when_bit_set, BitAccess,
};
