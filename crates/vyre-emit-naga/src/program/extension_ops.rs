use std::sync::LazyLock;

use super::LoweringError;
use rustc_hash::{FxHashMap, FxHashSet};
use vyre_foundation::ir::{ExprNode, Ident, NodeExtension};

pub(crate) trait NagaProgramScanAtomicExpr: Send + Sync + 'static {
    fn naga_scan_atomic_expr(
        &self,
        ext: &dyn ExprNode,
        out: &mut FxHashSet<Ident>,
    ) -> Result<(), LoweringError>;
}

pub(crate) trait NagaProgramScanAtomicNode: Send + Sync + 'static {
    fn naga_scan_atomic_node(
        &self,
        ext: &dyn NodeExtension,
        out: &mut FxHashSet<Ident>,
    ) -> Result<(), LoweringError>;
}

pub(crate) struct NagaProgramScanAtomicExprRegistration {
    pub kind: &'static str,
    pub scanner: &'static dyn NagaProgramScanAtomicExpr,
}

pub(crate) struct NagaProgramScanAtomicNodeRegistration {
    pub kind: &'static str,
    pub scanner: &'static dyn NagaProgramScanAtomicNode,
}

inventory::collect!(NagaProgramScanAtomicExprRegistration);
inventory::collect!(NagaProgramScanAtomicNodeRegistration);

/// Expression scanners keyed by extension kind, built on first lookup.
static EXPR_SCANNERS: LazyLock<FxHashMap<&'static str, &'static dyn NagaProgramScanAtomicExpr>> =
    LazyLock::new(|| {
        inventory::iter::<NagaProgramScanAtomicExprRegistration>
            .into_iter()
            .map(|registration| (registration.kind, registration.scanner))
            .collect()
    });

/// Node scanners keyed by extension kind, built on first lookup.
static NODE_SCANNERS: LazyLock<FxHashMap<&'static str, &'static dyn NagaProgramScanAtomicNode>> =
    LazyLock::new(|| {
        inventory::iter::<NagaProgramScanAtomicNodeRegistration>
            .into_iter()
            .map(|registration| (registration.kind, registration.scanner))
            .collect()
    });

pub(crate) fn scan_registered_atomic_expr(
    ext: &dyn ExprNode,
    out: &mut FxHashSet<Ident>,
) -> Result<bool, LoweringError> {
    let Some(scanner) = EXPR_SCANNERS.get(ext.extension_kind()) else {
        return Ok(false);
    };
    scanner.naga_scan_atomic_expr(ext, out)?;
    Ok(true)
}

pub(crate) fn scan_registered_atomic_node(
    ext: &dyn NodeExtension,
    out: &mut FxHashSet<Ident>,
) -> Result<bool, LoweringError> {
    let Some(scanner) = NODE_SCANNERS.get(ext.extension_kind()) else {
        return Ok(false);
    };
    scanner.naga_scan_atomic_node(ext, out)?;
    Ok(true)
}
