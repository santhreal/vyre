//! A named buffer binding, and the two records it carries.
//!
//! `BufferDecl` is here; the discipline it declares is [`linear_type`] and the
//! refinement it attaches to its element count is [`shape_predicate`], each
//! with the tests that own its grammar.

use std::ops::Range;
use std::sync::Arc;

use crate::ir_inner::model::op_signature::{BufferAccess, DataType};

use super::{MemoryHints, MemoryKind};

/// The substructural discipline a buffer binding declares.
mod linear_type;

/// The refinement predicate a buffer binding attaches to its element count.
mod shape_predicate;

pub use linear_type::LinearType;
pub use shape_predicate::ShapePredicate;

/// A named buffer binding in a program.
///
/// # Examples
///
/// ```
/// use vyre::ir::{BufferDecl, BufferAccess, DataType};
///
/// let buf = BufferDecl::read("input", 0, DataType::U32);
/// assert_eq!(buf.name(), "input");
/// assert_eq!(buf.binding(), 0);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferDecl {
    /// Human-readable name. Referenced by `Expr::Load`, `Node::Store`, etc.
    pub name: Arc<str>,
    /// Binding slot: `@binding(N)`. All buffers are in `@group(0)`.
    /// Ignored for `BufferAccess::Workgroup`.
    pub binding: u32,
    /// Access mode.
    pub access: BufferAccess,
    /// Memory tier.
    pub kind: MemoryKind,
    /// Element data type.
    pub element: DataType,
    /// Number of elements.
    ///
    /// For `Workgroup` memory this is the static array length.
    /// For storage and uniform buffers this is `0` (runtime-sized).
    pub count: u32,
    /// Whether this buffer is the scalar expression output for composition inlining.
    pub is_output: bool,
    /// Whether the end-to-end pipeline reads this buffer after Program execution.
    ///
    /// Passes must treat this as an externally-visible sink even when the IR
    /// itself does not read the buffer again.
    pub pipeline_live_out: bool,
    /// Optional byte range to read back from this output buffer.
    ///
    /// `None` preserves the historical behavior and reads back the full
    /// declared output buffer.
    pub output_byte_range: Option<Range<usize>>,
    /// Non-binding backend optimization hints.
    pub hints: MemoryHints,
    /// When true, admits `DataType::Bytes` load/store despite V013.
    ///
    /// Bytes-producing or bytes-extraction ops (decode.base64,
    /// `compression.lz4_decompress`, `match.dfa_scan` position emission, etc.)
    /// opt into V013 relaxation per-buffer. Default false keeps scalar
    /// arithmetic protected from accidental bytes-blob reinterpretation.
    pub bytes_extraction: bool,
    /// Linear-type discipline for this buffer (P-1.0-V2.1).
    ///
    /// Defaults to `LinearType::Unrestricted` so existing programs
    /// continue to type-check. Authors opt in by calling
    /// [`BufferDecl::with_linear_type`]. The type-checker pass
    /// (`crate::validate::linear_type`) walks the IR and
    /// rejects programs that violate the declared discipline; backends
    /// that hit a violation surface it as a validation error before
    /// lowering.
    pub linear_type: LinearType,
    /// Optional shape-refinement predicate (P-1.0-V3.1).
    ///
    /// `None` is the default (no shape constraint, identical to the
    /// pre-V3.x IR). Authors opt in via
    /// [`BufferDecl::with_shape_predicate`]. The validator
    /// ([`crate::validate::shape_predicate::check_shape_predicates`])
    /// evaluates each predicate against the program's static `count`
    /// at `validate()` time and rejects programs whose static shape
    /// contradicts the declaration.
    pub shape_predicate: Option<ShapePredicate>,
}

impl BufferDecl {
    /// Create a storage buffer declaration.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre::ir::{BufferDecl, BufferAccess, DataType};
    /// let _ = BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::U32);
    /// ```
    #[must_use]
    #[inline]
    pub fn storage(name: &str, binding: u32, access: BufferAccess, element: DataType) -> Self {
        let kind = match &access {
            BufferAccess::ReadOnly => MemoryKind::Readonly,
            BufferAccess::Uniform => MemoryKind::Uniform,
            BufferAccess::Workgroup => MemoryKind::Shared,
            _ => MemoryKind::Global,
        };
        Self {
            name: Arc::from(name),
            binding,
            access,
            kind,
            element,
            count: 0,
            is_output: false,
            pipeline_live_out: false,
            output_byte_range: None,
            hints: MemoryHints::default(),
            bytes_extraction: false,
            linear_type: LinearType::default(),
            shape_predicate: None,
        }
    }

    /// Shorthand for a read-only storage buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre::ir::{BufferDecl, DataType};
    /// let _ = BufferDecl::read("a", 0, DataType::U32);
    /// ```
    #[must_use]
    #[inline]
    pub fn read(name: &str, binding: u32, element: DataType) -> Self {
        Self::storage(name, binding, BufferAccess::ReadOnly, element)
    }

    /// Shorthand for a read-write storage buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre::ir::{BufferDecl, DataType};
    /// let _ = BufferDecl::read_write("a", 0, DataType::U32);
    /// ```
    #[must_use]
    #[inline]
    pub fn read_write(name: &str, binding: u32, element: DataType) -> Self {
        Self::storage(name, binding, BufferAccess::ReadWrite, element)
    }

    /// Shorthand for the read-write result buffer used by call inlining.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre::ir::{BufferDecl, DataType};
    /// let _ = BufferDecl::output("a", 0, DataType::U32);
    /// ```
    #[must_use]
    #[inline]
    pub fn output(name: &str, binding: u32, element: DataType) -> Self {
        Self {
            is_output: true,
            pipeline_live_out: true,
            ..Self::read_write(name, binding, element)
        }
    }

    /// Mark whether a caller/backend observes this buffer after Program execution.
    #[must_use]
    #[inline]
    pub fn with_pipeline_live_out(mut self, flag: bool) -> Self {
        self.pipeline_live_out = flag;
        self
    }

    /// Attach an output byte range for backends that can read back a slice.
    #[must_use]
    #[inline]
    pub fn with_output_byte_range(mut self, range: Range<usize>) -> Self {
        self.output_byte_range = Some(range);
        self
    }

    /// Set the static element count for storage-style buffers.
    ///
    /// Set the element count. A count of `0` retains the IR's
    /// runtime-sized-buffer representation; validators reject zero-sized
    /// workgroup allocations before dispatch.
    #[must_use]
    #[inline]
    pub fn with_count(mut self, count: u32) -> Self {
        self.count = count;
        self
    }

    /// Shorthand for a uniform buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre::ir::{BufferDecl, DataType};
    /// let _ = BufferDecl::uniform("a", 0, DataType::U32);
    /// ```
    #[must_use]
    #[inline]
    pub fn uniform(name: &str, binding: u32, element: DataType) -> Self {
        Self::storage(name, binding, BufferAccess::Uniform, element)
    }

    /// Shorthand for a workgroup-local shared array.
    ///
    /// `count` is the static number of elements visible to all invocations
    /// in the same workgroup.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre::ir::{BufferAccess, BufferDecl, DataType, MemoryKind};
    ///
    /// let scratch = BufferDecl::workgroup("scratch", 64, DataType::U32);
    ///
    /// assert_eq!(scratch.name(), "scratch");
    /// assert_eq!(scratch.access(), BufferAccess::Workgroup);
    /// assert_eq!(scratch.kind(), MemoryKind::Shared);
    /// assert_eq!(scratch.count(), 64);
    /// ```
    #[must_use]
    #[inline]
    pub fn workgroup(name: &str, count: u32, element: DataType) -> Self {
        Self {
            name: Arc::from(name),
            binding: 0,
            access: BufferAccess::Workgroup,
            kind: MemoryKind::Shared,
            element,
            count,
            is_output: false,
            pipeline_live_out: false,
            output_byte_range: None,
            hints: MemoryHints::default(),
            bytes_extraction: false,
            linear_type: LinearType::default(),
            shape_predicate: None,
        }
    }

    /// Mark this buffer as a bytes-extraction context so V013 admits Bytes load/store.
    #[must_use]
    #[inline]
    pub fn with_bytes_extraction(mut self, flag: bool) -> Self {
        self.bytes_extraction = flag;
        self
    }

    /// Set the linear-type discipline (P-1.0-V2.1).
    ///
    /// Defaults to [`LinearType::Unrestricted`] from the constructor;
    /// the type-checker pass enforces stricter disciplines when set.
    #[must_use]
    #[inline]
    pub fn with_linear_type(mut self, linear_type: LinearType) -> Self {
        self.linear_type = linear_type;
        self
    }

    /// Set the shape-refinement predicate (P-1.0-V3.1).
    ///
    /// Defaults to `None` (unconstrained); the validator
    /// ([`crate::validate::shape_predicate::check_shape_predicates`])
    /// rejects programs whose static `count` violates the predicate.
    #[must_use]
    #[inline]
    pub fn with_shape_predicate(mut self, predicate: ShapePredicate) -> Self {
        self.shape_predicate = Some(predicate);
        self
    }

    /// Override the memory tier.
    #[must_use]
    #[inline]
    pub fn with_kind(mut self, kind: MemoryKind) -> Self {
        self.kind = kind;
        self
    }

    /// Override memory optimization hints.
    #[must_use]
    #[inline]
    pub fn with_hints(mut self, hints: MemoryHints) -> Self {
        self.hints = hints;
        self
    }

    /// Buffer name.
    #[must_use]
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Binding slot.
    #[must_use]
    #[inline]
    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Buffer access mode.
    #[must_use]
    #[inline]
    pub fn access(&self) -> BufferAccess {
        self.access.clone()
    }

    /// Memory tier.
    #[must_use]
    #[inline]
    pub fn kind(&self) -> MemoryKind {
        self.kind
    }

    /// Non-binding memory hints.
    #[must_use]
    #[inline]
    pub fn hints(&self) -> MemoryHints {
        self.hints
    }

    /// Element data type.
    #[must_use]
    #[inline]
    pub fn element(&self) -> DataType {
        self.element.clone()
    }

    /// Static element count for workgroup buffers.
    #[must_use]
    #[inline]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Whether [`Self::count`] is a static array length that reaches generated
    /// backend code, rather than a runtime-sized binding length.
    ///
    /// This mirrors, arm for arm, the `MemoryClass::Shared` and
    /// `MemoryClass::Scratch` cases of `vyre_lower::lower::memory_class`, which
    /// is the single Program-to-descriptor boundary every emitter reads. Those
    /// two classes are the ones whose `element_count` becomes a fixed-length
    /// array in emitted code; every other class emits a runtime-sized array
    /// and ignores the count.
    ///
    /// `Persistent` is excluded because it is rejected before classification.
    ///
    /// Compiled-pipeline cache identity uses this to decide whether `count`
    /// belongs in the cache key: including it for a runtime-sized storage
    /// buffer would recompile the shader on every buffer resize, and omitting
    /// it for a workgroup buffer would let two different shared-memory array
    /// lengths share one cache entry.
    #[must_use]
    #[inline]
    pub fn has_static_element_count(&self) -> bool {
        match (self.kind, &self.access) {
            (MemoryKind::Persistent, _) => false,
            (MemoryKind::Shared, _) | (_, BufferAccess::Workgroup) => true,
            (MemoryKind::Local, _) => true,
            _ => false,
        }
    }

    /// Static packed byte length for fixed-size buffers.
    ///
    /// Returns `Ok(None)` for runtime-sized buffer declarations (`count == 0`)
    /// and for fixed-count buffers whose element type is runtime-sized. Sub-byte
    /// element types use their packed bit width, so three `I4` elements occupy
    /// two bytes rather than three conservative one-byte lanes.
    ///
    /// # Errors
    ///
    /// Returns an actionable diagnostic when the packed byte count overflows.
    pub fn static_byte_len(&self) -> Result<Option<usize>, String> {
        let count = usize::try_from(self.count).map_err(|error| {
            format!(
                "buffer `{}` static element count {} cannot fit usize ({error}). Fix: split the buffer or reduce its element count.",
                self.name, self.count
            )
        })?;
        if count == 0 {
            return Ok(None);
        }
        self.element.packed_size_bytes(count).map_err(|error| {
            format!(
                "buffer `{}` static byte length could not be computed: {error}. Fix: use a fixed-width element type or split the buffer.",
                self.name
            )
        })
    }

    /// Return true when this buffer is the unique inlining result buffer.
    #[must_use]
    #[inline]
    pub fn is_output(&self) -> bool {
        self.is_output
    }

    /// Return true when the buffer must survive IR-local deadness analysis.
    #[must_use]
    #[inline]
    pub fn is_pipeline_live_out(&self) -> bool {
        self.pipeline_live_out
    }

    /// True when a backend ALLOCATES this buffer's storage (an output it writes) rather
    /// than reading it from the dispatch inputs: an `is_output` buffer, any `WriteOnly`
    /// buffer, or a `pipeline_live_out` `ReadWrite` intermediate.
    ///
    /// This is the SINGLE cross-backend definition of "backend-allocated output": the
    /// reference interpreter, the CpuRef backend, and the device drivers MUST all agree
    /// on which buffers they allocate-and-write vs. read from inputs, so each calls this
    /// method instead of re-deriving the predicate. Drift here would make the interpreter
    /// and a backend disagree on a program's outputs (a silent readback bug).
    #[must_use]
    #[inline]
    pub fn is_backend_allocated_output(&self) -> bool {
        self.is_output()
            || matches!(self.access, BufferAccess::WriteOnly)
            || (self.is_pipeline_live_out() && matches!(self.access, BufferAccess::ReadWrite))
    }

    /// Refuse this buffer when it is backend-allocated and has no static size.
    ///
    /// A buffer selected by [`Self::is_backend_allocated_output`] never receives
    /// host bytes, so `count == 0` leaves an executor nothing to size its
    /// allocation or its readback from. Every execution path calls this rather
    /// than re-deriving the condition, so the reference interpreter refuses
    /// exactly what the device backends refuse. An oracle that accepts what its
    /// targets reject certifies programs that cannot run.
    ///
    /// A writable buffer that is NOT backend-allocated (a plain `ReadWrite`) is
    /// deliberately accepted: it consumes one host input slot, so its element
    /// count is inferable from the bytes the caller supplies and is resolved per
    /// dispatch.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre::ir::{BufferDecl, DataType};
    ///
    /// // No count and backend-allocated: refused, and the message names the remedy.
    /// let error = BufferDecl::output("out", 0, DataType::U32)
    ///     .require_static_readback_size()
    ///     .expect_err("a countless output has no readback size");
    /// assert!(error.contains(".with_count(n)"));
    ///
    /// // A count makes it well-formed.
    /// assert!(BufferDecl::output("out", 0, DataType::U32)
    ///     .with_count(4)
    ///     .require_static_readback_size()
    ///     .is_ok());
    ///
    /// // A plain read_write takes its size from the caller's bytes, so it is fine.
    /// assert!(BufferDecl::read_write("rw", 1, DataType::U32)
    ///     .require_static_readback_size()
    ///     .is_ok());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the operator-facing message when this buffer is backend-allocated
    /// and its readback size cannot be determined. A `count` of zero means
    /// "runtime-sized" rather than "zero elements", since `count` defaults to
    /// zero and so cannot represent a declared empty buffer. An explicit
    /// `output_byte_range` states the readback size directly, so it satisfies
    /// this check on its own: that is how a legitimately EMPTY output declares
    /// itself, with `.with_output_byte_range(0..0)`. Without that escape an
    /// empty-input program is indistinguishable from a mis-declared one, and the
    /// only way to pass is to inflate the count to a nonzero value the buffer
    /// does not have.
    #[inline]
    pub fn require_static_readback_size(&self) -> Result<(), String> {
        if self.is_backend_allocated_output() && self.count == 0 && self.output_byte_range.is_none()
        {
            return Err(format!(
                "backend-allocated output buffer `{}` has no static element count and no output byte range, so its readback size is unknown. Fix: declare it with .with_count(n), or with .with_output_byte_range(0..0) if it is genuinely empty.",
                self.name()
            ));
        }
        Ok(())
    }

    /// Byte range the consumer needs from this output buffer, if declared.
    #[must_use]
    #[inline]
    pub fn output_byte_range(&self) -> Option<Range<usize>> {
        self.output_byte_range.clone()
    }

    /// Linear-type discipline (P-1.0-V2.1).
    #[must_use]
    #[inline]
    pub fn linear_type(&self) -> LinearType {
        self.linear_type
    }

    /// Shape-refinement predicate (P-1.0-V3.1).
    #[must_use]
    #[inline]
    pub fn shape_predicate(&self) -> Option<&ShapePredicate> {
        self.shape_predicate.as_ref()
    }
}
