//! Vulkan compute dispatch for the SPIR-V backend.
//!
//! Uses `ash` to drive a minimal Vulkan 1.0 compute pipeline:
//! instance → physical device (with compute queue) → logical device →
//! shader module → descriptor set → compute pipeline → command buffer →
//! fence-wait submit.
//!
//! The loader and process-wide context are in [`device`], the descriptor set
//! and readback in [`bindings`], and the dispatch itself in [`dispatch`].

mod bindings;
mod device;
mod dispatch;

pub(crate) use device::{shared_device, VulkanDevice};
pub(crate) use dispatch::{dispatch_program, InputOrder};

#[cfg(test)]
mod tests {
    use super::dispatch::infer_grid;
    use vyre_foundation::ir::{BufferDecl, DataType, Program};

    /// Before the fix, a program with count=0 output and count=512 input launched exactly
    /// 1 workgroup (max_output_count==0 → div_ceil(lanes).max(1)==1). After the fix it
    /// falls back to the input count and launches ceil(512/lanes) workgroups.
    #[test]
    fn test_infer_grid_runtime_sized_output_falls_back_to_input_count() {
        let program = Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(512),
                BufferDecl::output("out", 1, DataType::U32), // count=0: runtime-sized
            ],
            [64, 1, 1],
            Vec::new(),
        );
        let grid = infer_grid(&program, [64, 1, 1])
            .expect("Fix: infer_grid must succeed when input count is non-zero");
        assert!(
            grid[0] >= 512 / 64,
            "Fix: grid x must be at least ceil(512/64)=8, got {}",
            grid[0]
        );
    }

    /// A program where all buffers have count=0 must return an error requiring
    /// grid_override rather than silently launching 1 workgroup.
    #[test]
    fn test_infer_grid_all_runtime_sized_requires_grid_override() {
        let program = Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32), // count=0
                BufferDecl::output("out", 1, DataType::U32), // count=0
            ],
            [64, 1, 1],
            Vec::new(),
        );
        let result = infer_grid(&program, [64, 1, 1]);
        assert!(
            result.is_err(),
            "Fix: all-runtime-sized program must require grid_override, not silently launch 1 workgroup"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Fix:"),
            "Fix: error must carry Fix: hint, got: {err}"
        );
    }

    /// Static-count output still uses output count (unchanged behavior after fix).
    #[test]
    fn test_infer_grid_static_output_count_unchanged() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(256)],
            [64, 1, 1],
            Vec::new(),
        );
        let grid = infer_grid(&program, [64, 1, 1]).expect("Fix: static output count must succeed");
        assert_eq!(
            grid[0],
            4, // ceil(256/64) = 4
            "Fix: grid x must be ceil(256/64)=4, got {}",
            grid[0]
        );
    }
}
