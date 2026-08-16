//! Validation of buffer load and store operations.
//!
//! Every memory access in vyre IR must target a declared buffer, and every
//! write must target a writable one. A write is a `Node::Store` or the
//! destination of an async transfer: the target compilers lower an
//! `AsyncLoad`/`AsyncStore` destination to stores through that buffer's
//! binding, so a read-only destination is a write to a read-only binding.
//! Both go through [`admits_store`], because a second copy of that list is how
//! one write kind ends up admitting what the other refuses.

use crate::ir_inner::model::program::BufferDecl;
use crate::ir_inner::model::op_signature::{BufferAccess, DataType};
use crate::validate::{err, ValidationError};
use crate::validate::{ValidationLocation, ValidationPhase};
use rustc_hash::FxHashMap;

/// Whether a program may write into a buffer of this access mode.
///
/// `Workgroup` is writable and is not host-visible, which is why this is not
/// the same predicate as the output set in [`crate::serial::output_set`]. That
/// one answers which buffers a host reads back.
pub(crate) fn admits_store(access: &BufferAccess) -> bool {
    matches!(
        access,
        BufferAccess::ReadWrite | BufferAccess::WriteOnly | BufferAccess::Workgroup
    )
}

/// Validate that a `Node::Store` targets a writable, declared buffer.
///
/// The function checks two invariants: the buffer name must appear in
/// the program's `buffers` list, and its access mode must allow writes.
/// Violations are appended to `errors` with actionable hints.
///
/// # Examples
///
/// `check_store` is `pub(crate)` and runs inside
/// [`crate::validate::rule_pipeline::validate`] for every `Node::Store`. See
/// that function's unit tests for runnable coverage of the writable /
/// unknown-buffer / Bytes-element branches.
///
/// # Errors
///
/// Appends a `ValidationError` when the buffer is unknown or not
/// writable.
#[inline]
pub(crate) fn check_store(
    buffer: &str,
    buffers: &FxHashMap<&str, &BufferDecl>,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(buf) = buffers.get(buffer) {
        if !admits_store(&buf.access) {
            errors.push(err(
    "V063",
    ValidationPhase::Memory,
    ValidationLocation::Program,
    format!(
                "store to non-writable buffer `{buffer}`"
            ),
    "declare it with BufferAccess::ReadWrite, BufferAccess::WriteOnly, or BufferAccess::Workgroup.".to_string()
));
        }
        // L.1.18: V013 was historically enforced only on `Expr::Atomic`,
        // leaving `Node::Store` targeting a `Bytes` buffer to pass
        // validation silently and then fail lower in target-text emission.
        // Extend V013 here so the error surfaces at validate() time.
        if buf.element == DataType::Bytes && !buf.bytes_extraction {
            errors.push(err(
    "V013",
    ValidationPhase::Memory,
    ValidationLocation::Program,
    format!(
                "store to buffer `{buffer}` with element type `bytes` is not supported"
            ),
    "use a typed buffer (U32/I32/F32/…) for stores, or declare the buffer with `.with_bytes_extraction(true)` when this is a bytes-producing op such as decode.base64.".to_string()
));
        }
    } else {
        errors.push(err(
            "V064",
            ValidationPhase::Memory,
            ValidationLocation::Program,
            format!("store to unknown buffer `{buffer}`"),
            "declare it in Program::buffers.".to_string(),
        ));
    }
}

/// Validate that an async transfer's destination is writable when the program
/// declares it.
///
/// An `AsyncLoad`/`AsyncStore` endpoint may name a storage tier the dispatch
/// does not bind, which is why the name is not required to resolve. When it
/// does resolve, the target compilers lower the transfer to a counted store
/// loop through that buffer's binding, so a destination the program declared
/// read-only is a write to a read-only binding.
///
/// The rule is also the alias proof loop-invariant hoisting rests on.
/// `LoopLicm` treats a load from a read-only buffer as invariant because
/// nothing in a valid program writes one. A DMA into a read-only destination
/// would be exactly such a write, and the hoisted load would answer a value
/// from before the transfer.
///
/// # Errors
///
/// Appends a `ValidationError` when the destination resolves to a buffer no
/// program may write.
#[inline]
pub(crate) fn check_async_destination(
    destination: &str,
    buffers: &FxHashMap<&str, &BufferDecl>,
    errors: &mut Vec<ValidationError>,
) {
    let Some(buf) = buffers.get(destination) else {
        return;
    };
    if admits_store(&buf.access) {
        return;
    }
    errors.push(err(
        "V134",
        ValidationPhase::Memory,
        ValidationLocation::Program,
        format!("async transfer writes into non-writable buffer `{destination}`"),
        "declare it with BufferAccess::ReadWrite, BufferAccess::WriteOnly, or \
         BufferAccess::Workgroup, or name a storage tier the dispatch does not bind."
            .to_string(),
    ));
}

/// Validate that an `Expr::Load` targets a declared buffer.
///
/// Loads are less restricted than stores (read-only buffers are fine),
/// but the buffer name must still be declared in the program. This
/// function appends an error when it is not.
///
/// # Examples
///
/// `check_load` is `pub(crate)` and runs inside
/// [`crate::validate::rule_pipeline::validate`] for every `Expr::Load`. See
/// that function's unit tests for runnable coverage of the
/// unknown-buffer and Bytes-element branches.
///
/// # Errors
///
/// Appends a `ValidationError` when the buffer is not declared.
#[inline]
pub(crate) fn check_load(
    buffer: &str,
    buffers: &FxHashMap<&str, &BufferDecl>,
    errors: &mut Vec<ValidationError>,
) {
    match buffers.get(buffer) {
        None => {
            errors.push(err(
                "V065",
                ValidationPhase::Memory,
                ValidationLocation::Program,
                format!("load from unknown buffer `{buffer}`"),
                "declare it in Program::buffers.".to_string(),
            ));
        }
        // L.1.18: V013 coverage extends to `Expr::Load`  -  loading from
        // a `Bytes` buffer gives the caller an opaque multi-byte blob
        // that no scalar arithmetic in the IR knows how to consume.
        // Catch it here rather than letting target-text lowering fail with a
        // generic "unexpected Bytes type" diagnostic.
        Some(buf) if buf.element == DataType::Bytes && !buf.bytes_extraction => {
            errors.push(err(
    "V013",
    ValidationPhase::Memory,
    ValidationLocation::Program,
    format!(
                "load from buffer `{buffer}` with element type `bytes` is not supported"
            ),
    "declare the buffer with a typed element (U32/I32/F32/…) or with `.with_bytes_extraction(true)` when the consuming op is a dedicated bytes-extraction op.".to_string()
));
        }
        Some(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::BufferDecl;

    fn buf_map(decl: &BufferDecl) -> FxHashMap<&str, &BufferDecl> {
        let mut m = FxHashMap::default();
        m.insert(decl.name(), decl);
        m
    }

    #[test]
    fn store_to_unknown_buffer_errors() {
        let buffers: FxHashMap<&str, &BufferDecl> = FxHashMap::default();
        let mut errors = Vec::new();
        check_store("missing", &buffers, &mut errors);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message().contains("unknown buffer"));
    }

    #[test]
    fn store_to_readonly_errors() {
        let decl = BufferDecl::read("buf", 0, DataType::U32).with_count(4);
        let buffers = buf_map(&decl);
        let mut errors = Vec::new();
        check_store("buf", &buffers, &mut errors);
        assert!(errors.iter().any(|e| e.message().contains("non-writable")));
    }

    #[test]
    fn store_to_readwrite_passes() {
        let decl =
            BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4);
        let buffers = buf_map(&decl);
        let mut errors = Vec::new();
        check_store("buf", &buffers, &mut errors);
        assert!(errors.is_empty());
    }

    #[test]
    fn load_from_unknown_buffer_errors() {
        let buffers: FxHashMap<&str, &BufferDecl> = FxHashMap::default();
        let mut errors = Vec::new();
        check_load("missing", &buffers, &mut errors);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message().contains("unknown buffer"));
    }

    #[test]
    fn load_from_declared_buffer_passes() {
        let decl = BufferDecl::read("buf", 0, DataType::U32).with_count(4);
        let buffers = buf_map(&decl);
        let mut errors = Vec::new();
        check_load("buf", &buffers, &mut errors);
        assert!(errors.is_empty());
    }
}
