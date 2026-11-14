//! Megakernel work-queue item layout.

/// One work-queue item the megakernel worker consumes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResidentWorkItem {
    /// Stable op id index into the dialect registry.
    pub op_handle: u32,
    /// Input-buffer handle.
    pub input_handle: u32,
    /// Output-buffer handle.
    pub output_handle: u32,
    /// Optional per-item parameter word.
    pub param: u32,
}
