//! What an operation registration carries, and how the registry reads back.
//!
//! A catalog entry declares a type signature and a behavioural contract. Both
//! are shared presets rather than per-op values, so they have one owner here
//! instead of one copy per dialect. The catalog projection alongside them is
//! the read direction of the same fact: the library tier of the canonical
//! registry `vyre-foundation` owns, filtered for the consumers that need it.

pub mod contracts;
pub mod operation_catalog;
pub(crate) mod signatures;
