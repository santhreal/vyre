//! Wgpu adapter for the self-hosted optimizer, and the two IR shapes its
//! end-to-end suites assert against.
//!
//! `vyre_self_substrate::optimizer` drives every GPU pass through
//! [`ProgramDispatcher`]. Satisfying that trait from a live `WgpuBackend` is one
//! implementation, not one per suite.

use vyre::ir::{Expr, Node, Program};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Adapts a live `WgpuBackend` to the dispatcher the self-hosted optimizer
/// expects.
pub(crate) struct WgpuProgramDispatcher<'a> {
    backend: &'a WgpuBackend,
}

impl<'a> WgpuProgramDispatcher<'a> {
    pub(crate) fn new(backend: &'a WgpuBackend) -> Self {
        Self { backend }
    }
}

impl ProgramDispatcher for WgpuProgramDispatcher<'_> {
    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut config = DispatchConfig::default();
        config.grid_override = grid_override;
        VyreBackend::dispatch(self.backend, program, inputs, &config)
            .map_err(|err| DispatchError::BackendError(err.to_string()))
    }
}

/// A buffer-free program holding just `entry`, the shape every optimizer pass
/// suite feeds in.
pub(crate) fn wrapped(entry: Vec<Node>) -> Program {
    Program::wrapped(Vec::new(), [1, 1, 1], entry)
}

/// Peel the `Region` wrapper `Program::wrapped` adds and read the single
/// let-bound value.
///
/// # Panics
///
/// Panics when the program is not a single `Region` holding a single `Let`,
/// which means the pass under test changed the node shape rather than the
/// value.
pub(crate) fn first_let_value(p: &Program) -> Expr {
    match p.entry() {
        [Node::Region { body, .. }] => match body.as_slice() {
            [Node::Let { value, .. }] => value.clone(),
            _ => panic!("expected single Let in body, got {body:?}"),
        },
        _ => panic!("expected wrapped Program with single Region"),
    }
}
