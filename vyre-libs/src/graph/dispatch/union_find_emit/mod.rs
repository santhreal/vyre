//! Union-find substrate consumer.
//!
//! The self-substrate consumes the same backend-neutral IR primitive as any
//! other caller. Concrete drivers are responsible for target emission.

mod dispatch;

#[cfg(test)]
#[path = "../../../../tests/internal/graph/dispatch/union_find_emit/mod.rs"]
mod tests;

pub use dispatch::{
    union_find_alias_program, union_find_alias_via, union_find_alias_via_into,
    union_find_alias_via_with_scratch_into, UnionFindGpuScratch,
};
