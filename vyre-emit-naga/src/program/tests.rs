#![allow(missing_docs)]

// Tests for `mod.rs`. Split out per audit item #85 to keep the
// parent file focused on production code.

use super::atomic_scanner::{scan_buffer_targets, BufferTargets};
use super::{extension_ops, LoweringError};
use std::sync::Arc;
use vyre_foundation::ir::{BufferDecl, DataType, Ident};
use vyre_foundation::ir::{Expr, ExprNode, Node, Program};

macro_rules! opaque_expr_fixture {
    ($name:ident, $kind:literal, $fingerprint_byte:literal) => {
        #[derive(Debug)]
        struct $name;

        impl ExprNode for $name {
            fn extension_kind(&self) -> &'static str {
                $kind
            }

            fn debug_identity(&self) -> &str {
                $kind
            }

            fn result_type(&self) -> Option<DataType> {
                Some(DataType::U32)
            }

            fn cse_safe(&self) -> bool {
                true
            }

            fn stable_fingerprint(&self) -> [u8; 32] {
                [$fingerprint_byte; 32]
            }

            fn validate_extension(&self) -> Result<(), String> {
                Ok(())
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
    };
}

opaque_expr_fixture!(OpaqueAtomicExpr, "test::scan::opaque-atomic", 0x42);

struct OpaqueAtomicExprScanner;

impl extension_ops::NagaProgramScanAtomicExpr for OpaqueAtomicExprScanner {
    fn naga_scan_atomic_expr(
        &self,
        ext: &dyn ExprNode,
        out: &mut rustc_hash::FxHashSet<Ident>,
    ) -> Result<(), LoweringError> {
        ext.as_any()
                .downcast_ref::<OpaqueAtomicExpr>()
                .ok_or_else(|| {
                    LoweringError::invalid(
                        "opaque atomic scanner received the wrong expression payload. Fix: register scanner kinds with matching payload types.",
                    )
                })?;
        out.insert(Ident::from("opaque_target"));
        Ok(())
    }
}

static OPAQUE_ATOMIC_EXPR_SCANNER: OpaqueAtomicExprScanner = OpaqueAtomicExprScanner;

inventory::submit! {
    extension_ops::NagaProgramScanAtomicExprRegistration {
        kind: "test::scan::opaque-atomic",
        scanner: &OPAQUE_ATOMIC_EXPR_SCANNER,
    }
}

opaque_expr_fixture!(OpaqueUnknownExpr, "test::scan::opaque-unknown", 0x99);

#[test]
fn atomic_scan_collects_targets_from_opaque_expr_extensions() {
    let mut targets = BufferTargets::default();
    let expr = Expr::Opaque(Arc::new(OpaqueAtomicExpr));
    let node = Node::let_bind("x", expr);
    scan_buffer_targets(std::slice::from_ref(&node), &mut targets)
        .expect("Fix: atomic scanner should honor extension scan traits.");
    assert!(targets.atomic.contains(&Ident::from("opaque_target")));
}

#[test]
fn atomic_scan_rejects_unknown_opaque_expr_extensions() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 1, DataType::U32)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::Opaque(Arc::new(OpaqueUnknownExpr)),
        )],
    );
    let err = scan_buffer_targets(program.entry(), &mut BufferTargets::default())
        .expect_err("Fix: unsupported opaque atomics should fail with actionable error.");
    let message = err.to_string();
    assert!(message.contains("unsupported opaque expression"));
    assert!(message.contains("Fix:"));
}
