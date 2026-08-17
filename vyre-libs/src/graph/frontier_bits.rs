//! Packed-bitset addressing skeleton for graph kernels, delegating downward to
//! [`crate::builder::csr`].

pub(in crate::graph) use crate::builder::csr::{
    active_source_lane, bind_bit_address, bind_word, bit_is_set, set_bit, when_bit_set, BitAccess,
};
