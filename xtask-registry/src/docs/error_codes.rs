//! `cargo xtask error-codes` renders the canonical error-code catalogs.

use xtask::gate::{GateCtx, GateError, Report};

/// Holds both committed error catalogs to their live inventories.
///
/// The driver catalog names the `BackendError` variants a run can report; the
/// validation catalog names the rules the validator can refuse a program with.
/// Both are rendered from the crate that owns the inventory, and one gate writes
/// them because a generated document with no writer goes stale the first time its
/// source changes.
pub struct ErrorCodes;

impl xtask::gate::GateBehavior for ErrorCodes {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut inspection = xtask::artifact_gate::Inspection::new();
        inspection.generates_text(DRIVER_CODES_PATH, render_driver());
        inspection.generates_text(VALIDATION_CODES_PATH, render_validation());
        Ok(xtask::artifact_gate::settle_inspection(
            ctx,
            ctx.gate_name()?,
            inspection,
        ))
    }
}

const DRIVER_CODES_PATH: &str = "docs/generated/driver-error-codes.toml";

const VALIDATION_CODES_PATH: &str = "docs/generated/error-codes.toml";

fn render_driver() -> String {
    vyre_driver::error_catalog::render_catalog_toml()
}

fn render_validation() -> String {
    vyre_foundation::validate::render_catalog_toml()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the gate must render each inventory through the crate that owns it.
    /// A copied catalog or second renderer can drift while still producing TOML,
    /// and a catalog this gate does not write is a document whose stated
    /// regeneration command does nothing.
    #[test]
    fn each_catalog_uses_the_renderer_of_the_crate_that_owns_it() {
        assert_eq!(
            render_driver(),
            vyre_driver::error_catalog::render_catalog_toml()
        );
        assert!(render_driver().contains(
            "# Regenerate: ./cargo_full run -p xtask --bin xtask -- error-codes --write"
        ));
        assert_eq!(
            render_validation(),
            vyre_foundation::validate::render_catalog_toml()
        );
        assert!(
            render_validation().contains("V139"),
            "the validation catalog must carry the node rules"
        );
    }
}
