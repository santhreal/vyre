//! Out-of-bounds rules enforced by the parity engine.
//!
//! GPU drivers differ on what happens when a shader indexes past the end of a
//! buffer: some clamp, some return zero, some crash.
//!
//! Under [`ExecutionStrictness::Strict`](crate::request::ExecutionStrictness)
//! an access outside a declared extent is a structured
//! [`ReferenceErrorClass::OutOfBoundsAccess`](crate::ReferenceErrorClass) at
//! the access site, so the oracle never issues an output derived from an
//! index the program did not gate.
//!
//! Under [`ExecutionStrictness::DiagnosticPermissive`](crate::request::ExecutionStrictness)
//! the access is absorbed deterministically instead: defined-type zero-fill for
//! a scalar load, an empty slice for `Bytes`, and a dropped store. That mode
//! measures how far a program relies on the absorption; it cannot issue an
//! expected output.

use vyre_foundation::ir::DataType as IrDataType;

use crate::error::ReferenceError;
use crate::value::Value;
use vyre_foundation::ir::DataType;

use std::cell::Cell;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Report a poisoned reference buffer byte lock as an invariant violation.
///
/// A poisoned lock means a writer panicked mid-store, leaving the bytes
/// inconsistent. Recovering them would let the oracle emit corrupt golden
/// values that the conform gate then trusts as truth, so the unit of work that
/// reads them ends instead.
pub(crate) fn poisoned_buffer_byte_lock() -> ! {
    vyre_foundation::failure_domain::invariant_violation_poison(
        "the reference oracle",
        "reference Buffer byte lock",
    )
}

/// Count of out-of-bounds accesses the interpreter silently absorbed during one
/// tracked run.
///
/// The reference interpreter DEFINES OOB loads as zero-fill and OOB stores as a
/// no-op (see the module docstring) so its output stays deterministic. That
/// silent absorption is exactly what MASKS a GPU/CPU parity hazard: an IR program
/// with an ungated data-derived index "works" here but a real GPU, which does no
/// bounds-checking, reads garbage / corrupts memory. This report surfaces the
/// masking so a test can assert a program NEVER relies on it, a correctly-gated
/// program handles an out-of-contract index with explicit control flow and thus
/// records ZERO OOB accesses even on hostile input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OobReport {
    /// Scalar/`Bytes` loads whose index fell outside the buffer (zero-filled).
    pub oob_loads: u64,
    /// Stores whose index fell outside the buffer (dropped).
    pub oob_stores: u64,
    /// Atomic loads/stores whose index fell outside the buffer.
    pub oob_atomics: u64,
}

impl OobReport {
    /// Total OOB accesses of every kind. Zero means the run never indexed past a
    /// buffer end (the invariant a bounds-gated program upholds).
    #[must_use]
    pub fn total(&self) -> u64 {
        self.oob_loads
            .saturating_add(self.oob_stores)
            .saturating_add(self.oob_atomics)
    }
}

thread_local! {
    /// Per-thread OOB tally. The interpreter runs single-threaded per call, so a
    /// thread-local cleanly brackets one run without global cross-run contention.
    static OOB_COUNTS: Cell<OobReport> = const { Cell::new(OobReport {
        oob_loads: 0,
        oob_stores: 0,
        oob_atomics: 0,
    }) };
    /// Per-thread strictness. Strict is the default, so an entry point that
    /// states nothing refuses an out-of-bounds access rather than absorbing
    /// one. Diagnostic permissive evaluation opts out for the length of its
    /// run through [`enter_strictness`].
    static STRICT_MODE: Cell<bool> = const { Cell::new(true) };
}

/// Kind of access that fell outside a declared extent.
#[derive(Clone, Copy)]
enum OobAccess {
    Load,
    Store,
    Atomic,
}

impl OobAccess {
    /// Word used in the diagnostic for this access kind.
    ///
    /// The match has no catch-all arm, so a new access kind states its own
    /// diagnostic rather than borrowing another kind's.
    const fn label(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Store => "store",
            Self::Atomic => "atomic access",
        }
    }
}

/// Tally one out-of-bounds access, and refuse it under strict mode.
///
/// Absorbing the access is a diagnostic-mode behavior. Strict mode is the mode
/// whose outputs a backend is graded against, so the access ends the run with
/// the index and the extent it exceeded.
fn record_oob(
    access: OobAccess,
    buffer: &Buffer,
    index: u32,
    extent: u32,
) -> Result<(), ReferenceError> {
    OOB_COUNTS.with(|c| {
        let mut r = c.get();
        match access {
            OobAccess::Load => r.oob_loads = r.oob_loads.saturating_add(1),
            OobAccess::Store => r.oob_stores = r.oob_stores.saturating_add(1),
            OobAccess::Atomic => r.oob_atomics = r.oob_atomics.saturating_add(1),
        }
        c.set(r);
    });
    if !is_strict_mode() {
        return Ok(());
    }
    Err(ReferenceError::out_of_bounds(format!(
        "out-of-bounds {} at element index {index} of a {:?} buffer holding {extent} elements. \
         Fix: gate the index against the declared buffer extent before the access.",
        access.label(),
        buffer.element
    )))
}

/// Set this thread's strictness for the length of the returned guard.
///
/// The guard restores the previous value on drop, so a nested evaluation
/// cannot leave the thread in the mode it borrowed. Strict is the default, and
/// only diagnostic permissive evaluation asks for `false`.
pub(crate) fn enter_strictness(strict: bool) -> StrictnessGuard {
    let previous = STRICT_MODE.with(Cell::get);
    STRICT_MODE.with(|mode| mode.set(strict));
    StrictnessGuard { previous }
}

/// Restores the strictness that was in effect before the run it brackets.
pub(crate) struct StrictnessGuard {
    previous: bool,
}

impl Drop for StrictnessGuard {
    fn drop(&mut self) {
        STRICT_MODE.with(|mode| mode.set(self.previous));
    }
}

/// Return whether strict execution mode is active on this thread.
#[must_use]
pub(crate) fn is_strict_mode() -> bool {
    STRICT_MODE.with(Cell::get)
}

/// Reset this thread's OOB tally to zero. Call before a tracked run.
pub(crate) fn reset_oob_report() {
    OOB_COUNTS.with(|c| c.set(OobReport::default()));
}

/// Read this thread's accumulated OOB tally (does not reset).
#[must_use]
pub(crate) fn oob_report() -> OobReport {
    OOB_COUNTS.with(Cell::get)
}

/// Typed bytes backing one declared IR buffer.
///
/// This struct exists to give the reference interpreter a single place to enforce
/// stride-correct indexing and OOB semantics, independent of how any GPU driver
/// handles buffer bounds.
#[derive(Debug, Clone)]
pub struct Buffer {
    pub(crate) bytes: Arc<RwLock<Vec<u8>>>,
    pub(crate) element: IrDataType,
}

impl Buffer {
    /// Create a buffer from typed bytes.
    #[must_use]
    pub fn new(bytes: Vec<u8>, element: DataType) -> Self {
        Self {
            bytes: Arc::new(RwLock::new(bytes)),
            element,
        }
    }

    /// Acquire the byte buffer for reading, failing closed on poison.
    ///
    /// A poisoned lock means a writer panicked mid-store, leaving the bytes
    /// inconsistent. Silently recovering with `into_inner()` would let the CPU
    /// reference oracle emit corrupt golden values that the conform gate then
    /// trusts as truth (a silent correctness fallback (Law 10). Surface it).
    ///
    /// # Panics
    /// Panics when the lock is poisoned.
    fn read_bytes(&self) -> RwLockReadGuard<'_, Vec<u8>> {
        self.bytes
            .read()
            .unwrap_or_else(|_| poisoned_buffer_byte_lock())
    }

    /// Acquire the byte buffer for writing, failing closed on poison (see
    /// [`Buffer::read_bytes`]).
    ///
    /// # Panics
    /// Panics when the lock is poisoned.
    fn write_bytes(&self) -> RwLockWriteGuard<'_, Vec<u8>> {
        self.bytes
            .write()
            .unwrap_or_else(|_| poisoned_buffer_byte_lock())
    }

    pub(crate) fn len(&self) -> u32 {
        let bytes_guard = self.read_bytes();
        let count = if let Some(bits) = self.element.bit_width() {
            bytes_guard
                .len()
                .checked_mul(8)
                .map(|total_bits| total_bits / bits)
                .unwrap_or(usize::MAX)
        } else if let Some(stride) = self.element.size_bytes() {
            if stride == 0 {
                bytes_guard.len()
            } else {
                bytes_guard.len() / stride
            }
        } else {
            bytes_guard.len()
        };
        match u32::try_from(count) {
            Ok(value) => value,
            Err(_) => {
                debug_assert!(
                    false,
                    "Buffer::len overflowed u32::MAX for byte_len={}; element={:?}. \
                     Fix: split or downsize the buffer so per-element indexing remains representable.",
                    bytes_guard.len(),
                    self.element
                );
                u32::MAX
            }
        }
    }

    pub(crate) fn byte_len(&self) -> usize {
        self.read_bytes().len()
    }

    pub(crate) fn element(&self) -> &IrDataType {
        &self.element
    }

    pub(crate) fn zero_fill(&self) {
        self.write_bytes().fill(0);
    }

    /// Copy `byte_count` bytes starting at `start`, zero-padding a short tail.
    ///
    /// An async transfer names a byte span rather than an element index, so it
    /// reads through here instead of the element-indexed [`load`]. A span that
    /// starts past the end, or runs off the end, is out of bounds: strict mode
    /// refuses it and diagnostic mode zero-pads the part that is not backed by
    /// bytes.
    ///
    /// # Errors
    /// Returns an out-of-bounds error under strict mode when the span is not
    /// fully backed by bytes.
    ///
    /// # Panics
    /// Panics when the byte lock is poisoned; see [`Buffer::read_bytes`].
    pub(crate) fn read_window(
        &self,
        start: usize,
        byte_count: usize,
    ) -> Result<Vec<u8>, ReferenceError> {
        let bytes_guard = self.read_bytes();
        let mut payload = vec![0; byte_count];
        let available = bytes_guard.len().saturating_sub(start).min(byte_count);
        if available < byte_count {
            drop(bytes_guard);
            record_oob(OobAccess::Load, self, span_index(start), self.len())?;
            let bytes_guard = self.read_bytes();
            let available = bytes_guard.len().saturating_sub(start).min(byte_count);
            payload[..available].copy_from_slice(&bytes_guard[start..start + available]);
            return Ok(payload);
        }
        payload[..available].copy_from_slice(&bytes_guard[start..start + available]);
        Ok(payload)
    }

    /// Write `payload` starting at `start`.
    ///
    /// # Errors
    /// Returns an out-of-bounds error under strict mode when the span is not
    /// fully backed by bytes. Diagnostic mode drops the part past the end.
    ///
    /// # Panics
    /// Panics when the byte lock is poisoned; see [`Buffer::read_bytes`].
    pub(crate) fn write_window(&self, start: usize, payload: &[u8]) -> Result<(), ReferenceError> {
        let mut bytes_guard = self.write_bytes();
        let available = bytes_guard.len().saturating_sub(start).min(payload.len());
        if available < payload.len() {
            drop(bytes_guard);
            record_oob(OobAccess::Store, self, span_index(start), self.len())?;
            let mut bytes_guard = self.write_bytes();
            let available = bytes_guard.len().saturating_sub(start).min(payload.len());
            bytes_guard[start..start + available].copy_from_slice(&payload[..available]);
            return Ok(());
        }
        bytes_guard[start..start + available].copy_from_slice(&payload[..available]);
        Ok(())
    }

    /// Consume the buffer and return its bytes.
    ///
    /// # Panics
    /// Panics when the byte lock is poisoned; see [`Buffer::read_bytes`].
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        // Same poison policy as the guard helpers: a poisoned lock is a corrupt
        // reference buffer, never silently laundered.
        std::sync::Arc::try_unwrap(self.bytes)
            .map(|rw| {
                rw.into_inner()
                    .unwrap_or_else(|_| poisoned_buffer_byte_lock())
            })
            .unwrap_or_else(|shared| {
                shared
                    .read()
                    .unwrap_or_else(|_| poisoned_buffer_byte_lock())
                    .clone()
            })
    }

    /// Consume this buffer and return its contents as a value.
    #[must_use]
    pub fn into_value(self) -> crate::value::Value {
        crate::value::Value::from(self.into_bytes())
    }
}

/// Read element `index` of `buffer`.
///
/// # Errors
/// Returns an out-of-bounds error under strict mode when `index` is outside the
/// declared extent, and a type mismatch when the backing bytes do not decode as
/// the declared element type.
pub(crate) fn load(buffer: &Buffer, index: u32) -> Result<Value, ReferenceError> {
    let extent = buffer.len();
    let bytes_guard = buffer.read_bytes();
    let stride = buffer.element.min_bytes();
    let ty = ir_to_conform_type(buffer.element.clone());
    if matches!(buffer.element, IrDataType::Bytes) {
        let offset = index as usize;
        if offset > bytes_guard.len() {
            drop(bytes_guard);
            record_oob(OobAccess::Load, buffer, index, extent)?;
            return Ok(Value::from(Vec::new()));
        }
        return Ok(Value::from(&bytes_guard[offset..]));
    }
    let in_bounds = byte_offset(index, stride)
        .filter(|offset| stride != 0 && offset + stride <= bytes_guard.len());
    let Some(offset) = in_bounds else {
        drop(bytes_guard);
        record_oob(OobAccess::Load, buffer, index, extent)?;
        return absorbed_load(ty);
    };
    read_element(ty.clone(), &bytes_guard[offset..offset + stride]).map_err(|detail| {
        ReferenceError::type_mismatch(format!(
            "element {index} of a {:?} buffer does not decode as {ty:?}: {detail}. \
             Fix: declare the buffer with the element type its bytes carry.",
            buffer.element
        ))
    })
}

/// Deterministic value a diagnostic-mode out-of-bounds load yields.
///
/// # Errors
/// Returns a type mismatch when the element type has no defined zero. A load
/// answered with an empty payload instead would hand back a width the program
/// never declared, which is the failure diagnostic mode exists to record.
fn absorbed_load(ty: DataType) -> Result<Value, ReferenceError> {
    Value::try_zero_for(ty.clone()).ok_or_else(|| {
        ReferenceError::type_mismatch(format!(
            "an out-of-bounds load of a {ty:?} element has no defined zero. \
             Fix: declare the buffer with an element type of fixed storage width."
        ))
    })
}

/// Element index a byte-span diagnostic reports.
///
/// A window names a byte offset rather than an element, and the diagnostic
/// states element indices, so the offset is reported as itself rather than
/// divided by a stride the span does not declare.
fn span_index(start: usize) -> u32 {
    u32::try_from(start).unwrap_or(u32::MAX)
}

/// Write `value` into element `index` of `buffer`.
///
/// # Errors
/// Returns an out-of-bounds error under strict mode when `index` is outside the
/// declared extent.
pub(crate) fn store(buffer: &mut Buffer, index: u32, value: &Value) -> Result<(), ReferenceError> {
    let extent = buffer.len();
    let mut bytes_guard = buffer.write_bytes();
    let stride = buffer.element.min_bytes();
    if matches!(buffer.element, IrDataType::Bytes) {
        let offset = index as usize;
        if offset >= bytes_guard.len() {
            drop(bytes_guard);
            return record_oob(OobAccess::Store, buffer, index, extent);
        }
        let bytes = value.to_bytes();
        let available = bytes_guard.len() - offset;
        let write_len = bytes.len().min(available);
        bytes_guard[offset..offset + write_len].copy_from_slice(&bytes[..write_len]);
        return Ok(());
    }
    let in_bounds = byte_offset(index, stride)
        .filter(|offset| stride != 0 && offset + stride <= bytes_guard.len());
    let Some(offset) = in_bounds else {
        drop(bytes_guard);
        return record_oob(OobAccess::Store, buffer, index, extent);
    };
    write_element(
        buffer.element.clone(),
        &mut bytes_guard[offset..offset + stride],
        value,
    )
}

/// Read the 32-bit atomic word at element `index`.
///
/// `Ok(None)` is the diagnostic-mode absorption of an out-of-bounds atomic.
///
/// # Errors
/// Returns an out-of-bounds error under strict mode when `index` is outside the
/// declared extent.
pub(crate) fn atomic_load(buffer: &Buffer, index: u32) -> Result<Option<u32>, ReferenceError> {
    let extent = buffer.len();
    let bytes_guard = buffer.read_bytes();
    let stride = buffer.element.min_bytes().max(4);
    let in_bounds = byte_offset(index, stride).filter(|offset| offset + 4 <= bytes_guard.len());
    match in_bounds {
        Some(offset) => Ok(Some(read_u32(&bytes_guard[offset..offset + 4]))),
        None => {
            drop(bytes_guard);
            record_oob(OobAccess::Atomic, buffer, index, extent)?;
            Ok(None)
        }
    }
}

/// Write the 32-bit atomic word at element `index`.
///
/// # Errors
/// Returns an out-of-bounds error under strict mode when `index` is outside the
/// declared extent.
pub(crate) fn atomic_store(
    buffer: &mut Buffer,
    index: u32,
    value: u32,
) -> Result<(), ReferenceError> {
    let extent = buffer.len();
    let mut bytes_guard = buffer.write_bytes();
    let stride = buffer.element.min_bytes().max(4);
    let in_bounds = byte_offset(index, stride).filter(|offset| offset + 4 <= bytes_guard.len());
    match in_bounds {
        Some(offset) => {
            write_u32(&mut bytes_guard[offset..offset + 4], value);
            Ok(())
        }
        None => {
            drop(bytes_guard);
            record_oob(OobAccess::Atomic, buffer, index, extent)
        }
    }
}

fn byte_offset(index: u32, stride: usize) -> Option<usize> {
    (index as usize).checked_mul(stride)
}

/// Encode `value` into one element slot of a `element`-typed buffer.
///
/// The match has no catch-all arm. `DataType` is exhaustively matchable, so a
/// new element type is a build failure here rather than a slot that silently
/// receives whatever the last arm happened to write. Every arm below either
/// states the element's own encoding or copies the value's canonical bytes at
/// the slot width, which is the buffer's byte semantics for that type.
///
/// # Errors
/// Returns a type mismatch when a float element receives a value that carries
/// no number. That case used to write `0.0`, so a store of the wrong value
/// type produced a zero the program never computed and the oracle certified it.
fn write_element(
    element: IrDataType,
    target: &mut [u8],
    value: &Value,
) -> Result<(), ReferenceError> {
    // A `Value::Bytes` whose length is exactly the slot width is already the
    // element's storage encoding, so it is copied verbatim on every arm
    // including the float ones. That is a byte-exact store, not a coercion,
    // and it is how a load of a narrow element round-trips back into its
    // buffer.
    if matches!(value, Value::Bytes(bytes) if bytes.len() == target.len()) {
        value.write_bytes_width_into(target);
        return Ok(());
    }
    match element {
        IrDataType::F16 => {
            let narrowed = float_element(&element, value)?;
            target.copy_from_slice(&crate::float16::f32_to_f16(narrowed).to_le_bytes());
        }
        IrDataType::BF16 => {
            let narrowed = float_element(&element, value)?;
            target.copy_from_slice(&crate::float16::f32_to_bf16(narrowed).to_le_bytes());
        }
        IrDataType::F32 => {
            // Value::Float carries an f64; the GPU buffer is four bytes
            // of f32, so narrow via `as f32` before writing. Dropping the
            // upper four bytes of `v.to_le_bytes()` (what the default
            // to_bytes_width path does) would mangle the f32 bit pattern.
            let narrowed =
                crate::execution::typed_ops::canonical_f32(float_element(&element, value)?);
            target.copy_from_slice(&narrowed.to_le_bytes());
        }
        IrDataType::U8
        | IrDataType::U16
        | IrDataType::U32
        | IrDataType::I8
        | IrDataType::I16
        | IrDataType::I32
        | IrDataType::I64
        | IrDataType::U64
        | IrDataType::Bool
        | IrDataType::I4
        | IrDataType::FP4
        | IrDataType::NF4
        | IrDataType::F8E4M3
        | IrDataType::F8E5M2
        | IrDataType::F64
        | IrDataType::Vec2U32
        | IrDataType::Vec4U32
        | IrDataType::Bytes
        | IrDataType::Array { .. }
        | IrDataType::Vec { .. }
        | IrDataType::Tensor
        | IrDataType::TensorShaped { .. }
        | IrDataType::SparseCsr { .. }
        | IrDataType::SparseCoo { .. }
        | IrDataType::SparseBsr { .. }
        | IrDataType::DeviceMesh { .. }
        | IrDataType::Quantized { .. }
        | IrDataType::Handle(_)
        | IrDataType::Opaque(_) => {
            value.write_bytes_width_into(target);
        }
    }
    Ok(())
}

/// The f32 a float element slot receives, or a refusal.
///
/// A `U32` source is the bit pattern of an f32 word, which is how a program
/// that computed a float through integer lanes writes it back. Anything else
/// carries no float, and a float slot has no defined encoding for it.
fn float_element(element: &IrDataType, value: &Value) -> Result<f32, ReferenceError> {
    match value {
        Value::Float(value) => Ok(*value as f32),
        Value::U32(bits) => Ok(f32::from_bits(*bits)),
        Value::I32(_) | Value::U64(_) | Value::Bool(_) | Value::Bytes(_) | Value::Array(_) => {
            Err(ReferenceError::type_mismatch(format!(
                "store of {value:?} into a {element:?} element has no defined float encoding. \
                 Fix: cast the value to f32 before the store."
            )))
        }
    }
}

fn read_element(ty: DataType, bytes: &[u8]) -> Result<Value, String> {
    match ty {
        DataType::F16 => {
            if bytes.len() < 2 {
                return Err("f16 requires 2 bytes".to_string());
            }
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(Value::Float(f64::from(crate::float16::f16_to_f32(bits))))
        }
        DataType::BF16 => {
            if bytes.len() < 2 {
                return Err("bf16 requires 2 bytes".to_string());
            }
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(Value::Float(f64::from(crate::float16::bf16_to_f32(bits))))
        }
        // Every remaining declared `DataType` is named rather than absorbed,
        // so adding a variant to the spec fails to compile here instead of
        // silently decoding through the generic element path.
        other @ (DataType::U8
        | DataType::U16
        | DataType::U32
        | DataType::I8
        | DataType::I16
        | DataType::I32
        | DataType::I64
        | DataType::U64
        | DataType::Vec2U32
        | DataType::Vec4U32
        | DataType::Bool
        | DataType::Bytes
        | DataType::F32
        | DataType::F64
        | DataType::F8E4M3
        | DataType::F8E5M2
        | DataType::I4
        | DataType::FP4
        | DataType::NF4
        | DataType::Tensor
        | DataType::Handle(_)
        | DataType::Array { .. }
        | DataType::Vec { .. }
        | DataType::TensorShaped { .. }
        | DataType::SparseCsr { .. }
        | DataType::SparseCoo { .. }
        | DataType::SparseBsr { .. }
        | DataType::DeviceMesh { .. }
        | DataType::Quantized { .. }
        | DataType::Opaque(_)) => Value::from_element_bytes(other, bytes),
    }
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn write_u32(bytes: &mut [u8], value: u32) {
    bytes.copy_from_slice(&value.to_le_bytes());
}

/// The element type a load decodes as.
///
/// `IrDataType` and `DataType` are the same frozen contract type, so every
/// element type decodes as itself. `Bool` is the one remap: a GPU stores a
/// boolean as a word, so a `Bool` buffer decodes through the `U32` reader and
/// the program sees the word it would read on a device.
fn ir_to_conform_type(ty: IrDataType) -> DataType {
    if matches!(ty, IrDataType::Bool) {
        DataType::U32
    } else {
        ty
    }
}

// Inline: covers the crate-private `Buffer` and `atomic_store`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bits(value: Value) -> u32 {
        match value {
            Value::Float(value) => (value as f32).to_bits(),
            other => {
                let bytes = other.to_bytes();
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            }
        }
    }

    #[test]
    fn f32_load_canonicalizes_subnormal_and_nan_payloads() {
        let positive_subnormal = Buffer::new(1u32.to_le_bytes().to_vec(), DataType::F32);
        assert_eq!(f32_bits(load(&positive_subnormal, 0).unwrap()), 0x0000_0000);

        let negative_subnormal = Buffer::new(0x8000_0001u32.to_le_bytes().to_vec(), DataType::F32);
        assert_eq!(f32_bits(load(&negative_subnormal, 0).unwrap()), 0x8000_0000);

        let payload_nan = Buffer::new(0x7fa0_0001u32.to_le_bytes().to_vec(), DataType::F32);
        assert_eq!(f32_bits(load(&payload_nan, 0).unwrap()), 0x7fc0_0000);
    }

    #[test]
    fn f32_store_canonicalizes_subnormal_and_nan_payloads() {
        let mut subnormal = Buffer::new(vec![0; 4], DataType::F32);
        store(
            &mut subnormal,
            0,
            &Value::Float(f64::from(f32::from_bits(0x8000_0001))),
        )
        .unwrap();
        assert_eq!(f32_bits(subnormal.into_value()), 0x8000_0000);

        let mut payload_nan = Buffer::new(vec![0; 4], DataType::F32);
        store(&mut payload_nan, 0, &Value::U32(0x7fa0_0001)).unwrap();
        assert_eq!(f32_bits(payload_nan.into_value()), 0x7fc0_0000);
    }

    #[test]
    fn oob_accesses_are_counted_and_in_bounds_are_not() {
        // The OOB tally must count exactly the accesses diagnostic mode absorbs
        // (zero-fill loads / dropped stores), and nothing in-bounds, which is the
        // signal that reveals an ungated data-derived index.
        reset_oob_report();
        let _diagnostic = enter_strictness(false);
        let buf = Buffer::new(vec![0u8; 8], DataType::U32); // 2 elements
        let _ = load(&buf, 0);
        let _ = load(&buf, 1);
        assert_eq!(oob_report().total(), 0, "in-bounds loads must not count");

        let _ = load(&buf, 2); // element 2 of 2 → OOB
        let _ = load(&buf, 99); // far OOB
        let after_loads = oob_report();
        assert_eq!(after_loads.oob_loads, 2, "two OOB loads counted");
        assert_eq!(after_loads.oob_stores, 0);

        let mut wbuf = Buffer::new(vec![0u8; 8], DataType::U32);
        store(&mut wbuf, 1, &Value::U32(7)).unwrap(); // in bounds
        store(&mut wbuf, 5, &Value::U32(9)).unwrap(); // OOB, dropped
        let after_store = oob_report();
        assert_eq!(
            after_store.oob_stores, 1,
            "one OOB store counted, in-bounds not"
        );

        let mut abuf = Buffer::new(vec![0u8; 8], DataType::U32);
        atomic_store(&mut abuf, 7, 3).unwrap(); // OOB atomic
        assert_eq!(oob_report().oob_atomics, 1, "OOB atomic store counted");

        reset_oob_report();
        assert_eq!(oob_report().total(), 0, "reset clears the tally");
    }

    #[test]
    fn poisoned_reference_buffer_lock_is_not_silently_recovered() {
        // A writer that panics mid-store poisons the lock. The reference oracle
        // must fail closed on a subsequent access rather than handing back the
        // half-mutated bytes (which would silently produce a corrupt golden
        // value the conform gate then trusts). Law 10.
        let buffer = Buffer::new(vec![0u8; 8], DataType::U32);
        let poisoner = buffer.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.write_bytes();
            panic!("poison reference buffer lock mid-store");
        })
        .join();

        let panic = std::panic::catch_unwind(|| {
            let _ = buffer.len();
        })
        .expect_err("poisoned reference Buffer lock must panic instead of recovering");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&'static str>().copied())
            .unwrap_or("<non-string panic>");
        assert!(
            message.contains("reference Buffer byte lock was poisoned"),
            "panic must name the poisoned reference buffer lock, got: {message}"
        );
    }

    /// Both async-transfer window helpers must fail closed on a poisoned lock,
    /// for the same reason [`poisoned_reference_buffer_lock_is_not_silently_recovered`]
    /// gives: an async copy that reads or writes half-mutated bytes hands the
    /// conform gate a golden value that no correct backend can reproduce.
    ///
    /// Both reference node executors reach the copy through these two methods,
    /// so this covers both arms. The statement executor used to recover the
    /// poisoned guard with `into_inner()` while the hashmap executor panicked,
    /// which is the divergence that made one arm's oracle trustworthy and the
    /// other's not.
    #[test]
    fn poisoned_lock_fails_closed_in_both_async_window_helpers() {
        for (label, access) in [
            (
                "read_window",
                (|buffer: &Buffer| {
                    let _ = buffer.read_window(0, 4);
                }) as fn(&Buffer),
            ),
            ("write_window", |buffer: &Buffer| {
                let _ = buffer.write_window(0, &[1, 2, 3, 4]);
            }),
        ] {
            let buffer = Buffer::new(vec![0u8; 8], DataType::U32);
            let poisoner = buffer.clone();
            let _ = std::thread::spawn(move || {
                let _guard = poisoner.write_bytes();
                panic!("poison reference buffer lock mid-store");
            })
            .join();

            let payload = std::panic::catch_unwind(|| access(&buffer)).expect_err(
                "Fix: an async window helper must panic on a poisoned buffer lock, not recover",
            );
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&'static str>().copied())
                .unwrap_or("<non-string panic>");
            assert!(
                message.contains("reference Buffer byte lock was poisoned"),
                "Fix: {label} must name the poisoned lock contract, got: {message}"
            );
        }
    }
}
