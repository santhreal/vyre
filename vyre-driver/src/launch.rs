//! Backend-neutral dispatch launch preparation.

use vyre_foundation::ir::Program;

use crate::binding::Binding;
use crate::program_walks::{
    dispatch_element_count_for_program, infer_dispatch_grid_for_count,
    try_dispatch_param_words_into,
};
use crate::tuner::Mode;
use crate::validation::{validate_launch_geometry, LaunchGeometryLimits};
use crate::{BackendError, DispatchConfig};

pub use crate::launch_natural::*;

/// Fully prepared launch metadata shared by concrete drivers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchPlan {
    /// Logical element count passed to the lowered kernel.
    pub element_count: u32,
    /// Effective workgroup/block shape after dispatch overrides.
    pub workgroup: [u32; 3],
    /// Effective grid shape after dispatch overrides or inference.
    pub grid: [u32; 3],
    /// Per-buffer element-count metadata uploaded as the shared params buffer.
    pub param_words: Vec<u32>,
    /// Maximum preferred alignment across all launch bindings.
    ///
    /// Concrete drivers use this to pick upload staging and device-buffer
    /// allocation paths without re-inspecting Program buffer declarations.
    pub max_binding_alignment: usize,
}

impl LaunchPlan {
    /// Empty launch plan with reusable parameter-word storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            element_count: 1,
            workgroup: [1, 1, 1],
            grid: [1, 1, 1],
            param_words: Vec::new(),
            max_binding_alignment: 1,
        }
    }

    /// Prepare dispatch geometry and parameter words from a validated binding plan.
    ///
    /// # Errors
    ///
    /// Returns when caller overrides produce zero dimensions, overflow the
    /// logical launch element count, or exceed backend-reported launch limits.
    pub fn from_bindings(
        program: &Program,
        bindings: &[Binding],
        config: &DispatchConfig,
        limits: LaunchGeometryLimits,
    ) -> Result<Self, BackendError> {
        let mut plan = Self::new();
        plan.prepare_into(program, bindings, config, limits)?;
        Ok(plan)
    }

    /// Prepare dispatch geometry and parameter words, reusing this plan's buffers.
    ///
    /// # Errors
    ///
    /// Returns when caller overrides produce zero dimensions, overflow the
    /// logical launch element count, or exceed backend-reported launch limits.
    pub fn prepare_into(
        &mut self,
        program: &Program,
        bindings: &[Binding],
        config: &DispatchConfig,
        limits: LaunchGeometryLimits,
    ) -> Result<(), BackendError> {
        self.prepare_into_for_mode(program, bindings, config, limits, Mode::from_env())
    }

    pub(crate) fn prepare_into_for_mode(
        &mut self,
        program: &Program,
        bindings: &[Binding],
        config: &DispatchConfig,
        limits: LaunchGeometryLimits,
        mode: Mode,
    ) -> Result<(), BackendError> {
        let workgroup =
            effective_launch_workgroup_for_mode(program, bindings, config, limits, mode);
        validate_launch_geometry(workgroup, [1, 1, 1], limits)?;
        let element_count = launch_element_count(program, bindings, workgroup, config, limits)?;
        let grid = match config.grid_override {
            Some(grid) => grid,
            None => {
                // Non-1D workgroups need an explicit grid_override  -
                // there's no single right way to map an unknown
                // element_count across N×M (or N×M×K) thread tiles,
                // and silently picking one produces silently-wrong
                // results. Force the caller to make the choice.
                if workgroup[1] != 1 || workgroup[2] != 1 {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: backend `{}` requires DispatchConfig::grid_override for non-1D workgroups. \
                             workgroup={:?} has no unambiguous default grid; set grid_override to the logical [x, y, z] you want.",
                            limits.backend, workgroup,
                        ),
                    });
                }
                infer_dispatch_grid_for_count(element_count, workgroup)?
            }
        };
        validate_launch_geometry(workgroup, grid, limits)?;
        self.element_count = element_count;
        self.workgroup = workgroup;
        self.grid = grid;
        self.max_binding_alignment = bindings
            .iter()
            .map(|binding| binding.preferred_alignment)
            .max()
            .unwrap_or(1);
        try_dispatch_param_words_into(bindings, element_count, &mut self.param_words).map_err(
            |error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {}: dispatch ABI parameter staging failed: {error}",
                    limits.backend
                ),
            },
        )?;
        Ok(())
    }
}

impl Default for LaunchPlan {
    fn default() -> Self {
        Self::new()
    }
}

fn launch_element_count(
    program: &Program,
    bindings: &[Binding],
    workgroup: [u32; 3],
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
) -> Result<u32, BackendError> {
    let inferred = dispatch_element_count_for_program(program, bindings);
    let Some(grid) = config.grid_override else {
        return Ok(inferred);
    };
    if workgroup.contains(&0) || grid.contains(&0) {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: {} grid_override and workgroup dimensions must all be non-zero.",
                limits.backend
            ),
        });
    }
    grid[0]
        .checked_mul(workgroup[0])
        .filter(|count| *count != 0)
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: {} grid_override.x * workgroup_size.x must fit in u32.",
                limits.backend
            ),
        })
}

pub(crate) fn effective_launch_workgroup_for_mode(
    program: &Program,
    bindings: &[Binding],
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    mode: Mode,
) -> [u32; 3] {
    let element_count = dispatch_element_count_for_program(program, bindings);
    resolve_launch_workgroup_for_mode(program, config, limits, element_count, mode)
}

/// Where a launch's workgroup shape comes from.
///
/// A program compiled through the whole-program compiler carries a geometry the
/// compiler searched for and recorded in the artifact, and the emitted module
/// declares that shape. Launching such a module at another width runs a kernel
/// nobody compiled, so the recorded geometry is authoritative and the launch
/// tuner never sees the launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaunchGeometry {
    /// No compiled artifact governs this launch, so the tuner may choose a width.
    Untracked,
    /// The artifact recorded this workgroup for the node being launched.
    Compiled([u32; 3]),
}

impl LaunchGeometry {
    /// Read the geometry a target descriptor recorded for one artifact node.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidProgram`] when the record is absent. A
    /// descriptor with a zero extent recorded nothing, and falling back to a
    /// declared or tuned width there would launch a shape the artifact never
    /// authenticated.
    pub fn from_recorded(workgroup: [u32; 3], backend: &str) -> Result<Self, BackendError> {
        if workgroup.contains(&0) {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: backend `{backend}` received an authenticated target whose descriptor records no workgroup geometry ({workgroup:?}). \
                     Recompile the artifact with a compiler that records the selected geometry for every node; the driver must not choose one."
                ),
            });
        }
        Ok(Self::Compiled(workgroup))
    }
}

/// Compute the shared VSA program fingerprint used by backend caches.
#[must_use]
pub fn program_vsa_fingerprint(program: &Program) -> Vec<u32> {
    program_vsa_fingerprint_words(program).to_vec()
}

/// Compute the shared VSA program fingerprint without heap allocation.
#[must_use]
pub fn program_vsa_fingerprint_words(program: &Program) -> [u32; 8] {
    let fingerprint = program.fingerprint();
    let mut words = [0u32; 8];
    for (word, chunk) in words.iter_mut().zip(fingerprint.chunks_exact(4)) {
        *word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::BindingRole;
    use crate::launch_fixtures::wide_limits;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

    #[test]
    fn program_vsa_fingerprint_words_match_wire_decoder() {
        let program = Program::wrapped(vec![], [64, 1, 1], vec![]);
        let words = program_vsa_fingerprint_words(&program);
        let fingerprint = program.fingerprint();

        for (index, chunk) in fingerprint.chunks_exact(4).enumerate() {
            assert_eq!(
                words[index],
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            );
        }
        assert_eq!(program_vsa_fingerprint(&program), words.to_vec());
    }

    #[test]
    fn launch_plan_prepare_into_reuses_param_words() {
        // An unguarded store: the launch covers the whole binding span, which is
        // what the parameter words this test watches are derived from.
        let program = Program::wrapped(
            vec![BufferDecl::output("input", 0, DataType::U32).with_count(7)],
            [64, 1, 1],
            vec![Node::store("input", Expr::logical_index(0), Expr::u32(1))],
        );
        let bindings = vec![Binding {
            name: std::sync::Arc::from("input"),
            binding: 0,
            buffer_index: 0,
            role: BindingRole::Input,
            element_size: 4,
            preferred_alignment: 64,
            element_count: 7,
            static_byte_len: Some(28),
            input_index: Some(0),
            output_index: None,
        }];
        let limits = wide_limits("test", 1536);
        let mut plan = LaunchPlan {
            param_words: Vec::with_capacity(8),
            ..LaunchPlan::new()
        };
        let ptr = plan.param_words.as_ptr();
        plan.prepare_into(&program, &bindings, &DispatchConfig::default(), limits)
            .unwrap();
        assert_eq!(plan.element_count, 7);
        assert_eq!(plan.grid, [1, 1, 1]);
        assert_eq!(plan.param_words, vec![7, 7]);
        assert_eq!(plan.max_binding_alignment, 64);
        assert_eq!(plan.param_words.as_ptr(), ptr);
    }
}
