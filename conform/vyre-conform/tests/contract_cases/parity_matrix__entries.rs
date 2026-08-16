// The operations this matrix covers: every live registration, plus the
// synthetic bundle that carries the IR variants no registered op builds.

use vyre::ir::{DataType, ExprNode, Program};

use super::parity_matrix_synthetic_entries::synthetic_entries;

pub(crate) type FixtureCases = Vec<Vec<Vec<u8>>>;
pub(crate) type FixtureFn = fn() -> FixtureCases;

#[derive(Clone, Copy)]
pub(crate) struct UnifiedEntry {
    pub(crate) id: &'static str,
    pub(crate) build: Option<fn() -> Program>,
    pub(crate) test_inputs: Option<FixtureFn>,
    pub(crate) expected_output: Option<FixtureFn>,
}

impl UnifiedEntry {
    pub(crate) fn program(&self) -> Option<Program> {
        self.build.map(|build| build().with_entry_op_id(self.id))
    }
}

#[derive(Debug)]
pub(crate) struct SyntheticOpaqueExpr;

impl ExprNode for SyntheticOpaqueExpr {
    fn extension_kind(&self) -> &'static str {
        "vyre.conform.synthetic.opaque"
    }

    fn debug_identity(&self) -> &str {
        "synthetic-opaque-expr"
    }

    fn result_type(&self) -> Option<DataType> {
        Some(DataType::U32)
    }

    fn cse_safe(&self) -> bool {
        true
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        [0x5a; 32]
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn wire_payload(&self) -> Vec<u8> {
        vec![0x5a]
    }
}

pub(crate) fn unified_entries() -> Vec<UnifiedEntry> {
    let canonical = vyre_registry_link::operation::live_operation_registry()
        .iter()
        .map(|entry| UnifiedEntry {
            id: entry.id,
            build: entry.build,
            test_inputs: entry.test_inputs,
            expected_output: entry.expected_output,
        });
    canonical.chain(synthetic_entries()).collect()
}
