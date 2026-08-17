//! Property gates for `vyre_reference::composition_witness::bitset_and_witness`.

#![cfg(feature = "bitset")]

use proptest::prelude::*;
use vyre_reference::composition_witness::bitset_and_witness;

#[macro_use]
mod bitset_law_properties;

bitset_and_law_tests!(cpu_ref);
