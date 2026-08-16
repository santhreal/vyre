//! The over-fire dispatch floor the hardware registration and out-of-bounds
//! suites share.
//!
//! ONE home so no two over-fire gates can drift. The adversarial case macros
//! and the dominator-tree oracle helper that used to sit here moved with the
//! composition domains they serve, to `vyre-libs/tests/common/mod.rs`.

pub(crate) use vyre_test_support::overfire_grid;
