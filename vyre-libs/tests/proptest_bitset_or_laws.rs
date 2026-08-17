//! Property gates for `vyre_reference::composition_witness::bitset_or_witness`.

#![cfg(feature = "bitset")]

use proptest::prelude::*;
use vyre_reference::composition_witness::bitset_or_witness;

#[macro_use]
mod bitset_law_properties;

bitset_or_law_tests!(cpu_ref);
