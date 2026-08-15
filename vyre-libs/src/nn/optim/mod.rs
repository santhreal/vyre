//! Optimizer sub-dialect for Parameter Golf recipe (all F32).
//!
//! MuonEq-R, AdamW, EMA, Newton-Schulz orthogonalization.
pub(crate) mod adamw_step;
pub(crate) mod ema_apply;
pub(crate) mod muon_core;
pub(crate) mod muon_update;
pub(crate) mod muoneq_r;
pub(crate) mod newton_schulz;

pub use adamw_step::adamw_step;
pub use ema_apply::ema_apply;
pub use muon_update::muon_update;
pub use muoneq_r::muoneq_r;
pub use newton_schulz::newton_schulz_5step;
