//! `cargo xtask error-codes` renders the canonical driver error catalog.

use xtask::gate::{GateCtx, GateError, Report};

/// Holds the committed driver error catalog to the live error-code inventory.
pub struct ErrorCodes;

impl xtask::gate::GateBehavior for ErrorCodes {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut inspection = xtask::artifact_gate::Inspection::new();
        inspection.generates_text(ERROR_CODES_PATH, render());
        Ok(xtask::artifact_gate::settle_inspection(
            ctx,
            ctx.gate_name()?,
            inspection,
        ))
    }
}

const ERROR_CODES_PATH: &str = "docs/generated/driver-error-codes.toml";

fn render() -> String {
    vyre_driver::error_catalog::render_catalog_toml()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the gate must render the driver's exhaustive inventory directly.
    /// A copied catalog or second renderer can drift while still producing TOML.
    #[test]
    fn driver_error_catalog_uses_canonical_renderer() {
        assert_eq!(render(), vyre_driver::error_catalog::render_catalog_toml());
        assert!(render().contains(
            "# Regenerate: ./cargo_full run -p xtask --bin xtask -- error-codes --write"
        ));
    }
}
