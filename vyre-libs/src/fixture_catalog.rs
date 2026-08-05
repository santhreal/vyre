//! Library composition fixture metadata.
//!
//! Execution belongs to upper conformance harnesses. This module contains only
//! neutral program builders and deterministic byte fixtures; it has no backend
//! or harness-crate dependency.

use vyre_foundation::ir::Program;
/// Floating-point parity policy for upper execution harnesses.
pub mod fp_contract;


pub use crate::region::{
    reparent_program_children, tag_program, wrap, wrap_anonymous, wrap_child,
};

/// Deterministic fixture input cases.
pub type InputsFn = fn() -> Vec<Vec<Vec<u8>>>;
/// Deterministic expected-output fixtures.
pub type ExpectedFn = fn() -> Vec<Vec<Vec<u8>>>;

/// Neutral fixture descriptor for a library composition.
pub struct OpEntry {
    /// Stable operation identifier.
    pub id: &'static str,
    /// Construct the neutral program under test.
    pub build: fn() -> Program,
    /// Deterministic input byte fixtures.
    pub test_inputs: Option<InputsFn>,
    /// Deterministic reference output byte fixtures.
    pub expected_output: Option<ExpectedFn>,
    /// Coarse library category.
    pub category: Option<&'static str>,
}

impl OpEntry {
    /// Construct a fixture descriptor.
    #[must_use]
    pub const fn new(
        id: &'static str,
        build: fn() -> Program,
        test_inputs: Option<InputsFn>,
        expected_output: Option<ExpectedFn>,
    ) -> Self {
        Self { id, build, test_inputs, expected_output, category: None }
    }

    /// Attach a category.
    #[must_use]
    pub const fn with_category(mut self, category: &'static str) -> Self {
        self.category = Some(category);
        self
    }

    /// Return the category.
    #[must_use]
    pub const fn category(&self) -> Option<&'static str> {
        self.category
    }

    /// Return the permitted f32 ULP drift for this composition.
    #[must_use]
    pub fn tolerance(&self) -> u32 {
        Self::tolerance_for_id(self.id)
    }

    /// Resolve the permitted f32 ULP drift for an operation id.
    #[must_use]
    pub fn tolerance_for_id(id: &str) -> u32 {
        match id {
            "vyre-libs::nn::softmax" => 1,
            "vyre-libs::nn::attention" | "vyre-libs::nn::gqa_attention" => 4,
            "vyre-libs::nn::layer_norm" | "vyre-libs::nn::silu" => 1,
            "vyre-libs::nn::logit_softcap" | "vyre-libs::nn::rms_norm" | "vyre-libs::nn::rms_norm_linear" => 2,
            "vyre-libs::math::fft::fft_convolve_circular_complex" => 4,
            "vyre-libs::math::linalg::matmul_strassen_2x2" => 32,
            "vyre-libs::optim::newton_schulz_5step" => 64,
            "vyre-libs::optim::ema_apply" => 1,
            "vyre-libs::optim::muoneq_r" => 8,
            "vyre-primitives::math::newton_schulz_poly5_f32" => 32,
            _ => 0,
        }
    }
}

inventory::collect!(OpEntry);

/// Iterate over neutral library composition fixtures.
pub fn all_entries() -> impl Iterator<Item = &'static OpEntry> {
    inventory::iter::<OpEntry>()
}

/// Fixpoint metadata consumed by upper execution harnesses.
#[derive(Clone, Debug)]
pub struct FixpointContract {
    /// Changed-flag buffer name.
    pub converged_flag_buffer: &'static str,
    /// Explicit iteration ceiling.
    pub max_iterations: u32,
}

/// Associates fixpoint metadata with a neutral composition.
pub struct FixpointRegistration {
    /// Stable operation id.
    pub op_id: &'static str,
    /// Fixpoint contract.
    pub contract: FixpointContract,
}

inventory::collect!(FixpointRegistration);

/// Look up fixpoint metadata.
#[must_use]
pub fn fixpoint_contract(op_id: &str) -> Option<&'static FixpointContract> {
    inventory::iter::<FixpointRegistration>()
        .find(|registration| registration.op_id == op_id)
        .map(|registration| &registration.contract)
}

/// Convergence metadata consumed by upper execution harnesses.
#[derive(Clone, Debug)]
pub struct ConvergenceContract {
    /// Stable operation id.
    pub op_id: &'static str,
    /// Explicit iteration ceiling.
    pub max_iterations: u32,
}

inventory::collect!(ConvergenceContract);

/// Look up convergence metadata.
#[must_use]
pub fn convergence_contract(op_id: &str) -> Option<&'static ConvergenceContract> {
    inventory::iter::<ConvergenceContract>().find(|contract| contract.op_id == op_id)
}
