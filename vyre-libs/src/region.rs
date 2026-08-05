//! Substrate-neutral composition-region helpers.
//!
//! The implementation is owned by `vyre-foundation`; this module gives
//! library builders concise names without depending on a test harness.

pub use vyre_foundation::composition::{
    reparent_program_children, tag_program, wrap_anonymous_region as wrap_anonymous,
    wrap_child_region as wrap_child, wrap_region as wrap,
};
