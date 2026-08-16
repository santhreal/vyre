// The operations this matrix covers: every live registration, plus the
// synthetic bundle that carries the IR variants no registered op builds.

use vyre::ir::{DataType, ExprNode, Program};

use super::parity_matrix_synthetic_entries::synthetic_entries;

pub(crate) type FixtureCases = Vec<Vec<Vec<u8>>>;
pub(crate) type FixtureFn = fn() -> FixtureCases;

#[derive(Clone, Copy)]
pub(crate) struct UnifiedEntry {
    pub(crate) id: &'static str,
    pub(crate) build: fn() -> Program,
    pub(crate) test_inputs: Option<FixtureFn>,
    pub(crate) expected_output: Option<FixtureFn>,
}

impl UnifiedEntry {
    pub(crate) fn program(&self) -> Program {
        (self.build)().with_entry_op_id(self.id)
    }
}

/// Extension kind of the synthetic opaque expression, named once so the
/// `ExprNode` impl and its wire resolver cannot drift apart.
const SYNTHETIC_OPAQUE_KIND: &str = "vyre.conform.synthetic.opaque";

#[derive(Debug)]
pub(crate) struct SyntheticOpaqueExpr;

impl ExprNode for SyntheticOpaqueExpr {
    fn extension_kind(&self) -> &'static str {
        SYNTHETIC_OPAQUE_KIND
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

/// The wire decoder for the synthetic extension.
///
/// `Expr::Opaque` round-trips through the neutral artifact, so an extension with
/// no registered resolver cannot be encoded for a backend: the target compiler
/// rejects the artifact at decode. The bundle carries the variant, so the test
/// binary owns the resolver for it.
fn decode_synthetic_opaque(
    payload: &[u8],
) -> Result<std::sync::Arc<dyn ExprNode>, String> {
    if payload != [0x5a] {
        return Err(format!(
            "synthetic opaque payload is {payload:?}. Fix: emit the single marker byte `wire_payload` writes."
        ));
    }
    Ok(std::sync::Arc::new(SyntheticOpaqueExpr))
}

inventory::submit! {
    vyre_foundation::extension::OpaqueExprResolver {
        kind: SYNTHETIC_OPAQUE_KIND,
        deserialize: decode_synthetic_opaque,
    }
}

/// Every registration that builds a neutral program, plus the synthetic bundle.
///
/// A registration with no builder is a callee identity: it exists so
/// `Expr::Call` resolves through the registry, and its signature is the whole
/// contract. `vyre_foundation::operation::OperationRegistry` owns that rule and
/// refuses a registration supplying neither a program nor a signature, so this
/// matrix takes the buildable ops as its subjects and does not re-judge the
/// ones the registry already validated.
pub(crate) fn unified_entries() -> Vec<UnifiedEntry> {
    let canonical = vyre_registry_link::operation::live_operation_registry()
        .iter()
        .filter_map(|entry| {
            Some(UnifiedEntry {
                id: entry.id,
                build: entry.build?,
                test_inputs: entry.test_inputs,
                expected_output: entry.expected_output,
            })
        });
    canonical.chain(synthetic_entries()).collect()
}
