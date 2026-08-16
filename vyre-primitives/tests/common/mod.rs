//! The over-fire dispatch floor the hardware registration and out-of-bounds
//! suites share.
//!
//! ONE home so no two over-fire gates can drift. The adversarial case macros
//! and the dominator-tree oracle helper that used to sit here moved with the
//! composition domains they serve, to `vyre-libs/tests/common/mod.rs`.

use vyre_foundation::ir::Program;

/// The one-workgroup over-fire dispatch floor shared by every over-fire gate: the
/// largest declared buffer element count plus one whole workgroup of lanes, the
/// realistic worst case a whole-workgroup GPU dispatch produces past the logical
/// element count. ONE home so no two over-fire gates can drift.
pub(crate) fn overfire_grid(program: &Program) -> u32 {
    let workgroup_lanes = program.workgroup_size()[0].max(1);
    let max_count = program
        .buffers()
        .iter()
        .map(vyre_foundation::ir::BufferDecl::count)
        .max()
        .unwrap_or(0);
    max_count.saturating_add(workgroup_lanes)
}
