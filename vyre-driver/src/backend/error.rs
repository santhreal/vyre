//! Actionable backend error taxonomy.

/// Machine-readable classification of a backend failure kind.
///
/// Use this to drive retry logic, circuit breakers, and alerting rules
/// without parsing human-readable message strings.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode {
    /// Backend device reported insufficient memory.
    DeviceOutOfMemory,
    /// Acquired device generation was lost or invalidated.
    DeviceLost,
    /// The backend does not support a required feature.
    UnsupportedFeature,
    /// A lock used by the backend failed to unlock safely.
    ///
    /// This is generally caused by a panic while a write guard was held and
    /// indicates an internal synchronization bug in process state.
    PoisonedLock,
    /// GPU kernel-source compilation failed. "Shader" in the variant
    /// name is historical; the code covers any kernel-source compile
    /// failure for any backend kernel-source or binary validation.
    /// A 2.0 rename to `KernelCompileFailed` is tracked in the
    /// semver-policy doc; the variant stays stable in 0.x.
    KernelCompileFailed,
    /// Command dispatch or queue submission failed.
    DispatchFailed,
    /// The program itself is invalid for this backend.
    InvalidProgram,
    /// A cooperative (whole-grid-sync) launch could not fit every block
    /// co-resident on the device. This is a routable performance condition,
    /// not a hard failure: the orchestrator should fall back (loudly) to a
    /// recall-identical non-cooperative path (resident fixpoint or host split).
    CooperativeResidencyExceeded,
    /// Unclassified error (produced by [`BackendError::new`]).
    Unknown,
}

impl ErrorCode {
    /// Stable integer identifier for API consumers and diagnostic catalogs.
    ///
    /// These ids are append-only. Existing assignments must not be reused or
    /// renumbered because downstream systems may persist them in telemetry,
    /// alert rules, and retry policies.
    #[must_use]
    pub const fn stable_id(self) -> u32 {
        match self {
            Self::DeviceOutOfMemory => 1001,
            Self::UnsupportedFeature => 1002,
            Self::PoisonedLock => 1003,
            Self::KernelCompileFailed => 1004,
            Self::DispatchFailed => 1005,
            Self::InvalidProgram => 1006,
            Self::CooperativeResidencyExceeded => 1007,
            Self::DeviceLost => 1008,
            Self::Unknown => 1999,
        }
    }

    /// Every variant, ordered by [`Self::stable_id`].
    ///
    /// Catalog renderers and conformance tests walk this instead of a
    /// hand-maintained list. [`Self::catalog_index`] and the const assertion
    /// below make a variant that is missing here a compile error rather than
    /// a silently uncatalogued code.
    pub const ALL: &'static [Self] = &[
        Self::DeviceOutOfMemory,
        Self::UnsupportedFeature,
        Self::PoisonedLock,
        Self::KernelCompileFailed,
        Self::DispatchFailed,
        Self::InvalidProgram,
        Self::CooperativeResidencyExceeded,
        Self::DeviceLost,
        Self::Unknown,
    ];

    /// Position of this code in [`Self::ALL`].
    ///
    /// Exhaustive on purpose: a new variant must add an arm, and the arm must
    /// name a position that exists and is not already taken, or the const
    /// assertion below fails to evaluate.
    const fn catalog_index(self) -> usize {
        match self {
            Self::DeviceOutOfMemory => 0,
            Self::UnsupportedFeature => 1,
            Self::PoisonedLock => 2,
            Self::KernelCompileFailed => 3,
            Self::DispatchFailed => 4,
            Self::InvalidProgram => 5,
            Self::CooperativeResidencyExceeded => 6,
            Self::DeviceLost => 7,
            Self::Unknown => 8,
        }
    }

    /// One-line description carried into the generated catalog.
    ///
    /// Exhaustive for the same reason as [`Self::catalog_index`]: a new
    /// variant cannot reach the catalog without a description.
    #[must_use]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::DeviceOutOfMemory => "Backend device reported insufficient memory.",
            Self::UnsupportedFeature => "The backend does not support a required feature.",
            Self::PoisonedLock => {
                "A lock used by the backend failed to unlock safely, indicating an \
                 internal synchronization bug in process state."
            }
            Self::KernelCompileFailed => "Kernel-source or kernel-binary compilation failed.",
            Self::DispatchFailed => "Command dispatch or queue submission failed.",
            Self::InvalidProgram => "The program itself is invalid for this backend.",
            Self::CooperativeResidencyExceeded => {
                "A whole-grid-sync launch could not fit every block co-resident on the \
                 device; the orchestrator falls back loudly to a recall-identical \
                 non-cooperative path."
            }
            Self::DeviceLost => "Acquired device generation was lost or invalidated.",
            Self::Unknown => {
                "The backend reported a failure it could not classify, produced by \
                 BackendError::new. A code that stays Unknown across releases is a \
                 missing variant, not a category."
            }
        }
    }
}

const _: () = {
    let mut index = 0;
    while index < ErrorCode::ALL.len() {
        assert!(
            ErrorCode::ALL[index].catalog_index() == index,
            "ErrorCode::ALL and ErrorCode::catalog_index disagree"
        );
        index += 1;
    }
};

/// Actionable backend dispatch failure.
///
/// Every error that flows through the frozen `VyreBackend` contract must
/// include remediation text beginning with `Fix: `. This guarantees that
/// conform reports are directly actionable for backend authors and that
/// consumers never receive an opaque failure string.
///
/// Use specific variants (`DeviceOutOfMemory`, `KernelCompileFailed`, etc.) when
/// the failure class is known. [`BackendError::Other`] carries actionable failures
/// that do not fit a structured variant.
///
/// # Examples
///
/// ```
/// use vyre_driver::BackendError;
///
/// let err = BackendError::new("adapter not found. Fix: install a compatible device driver.");
/// assert!(err.message().contains("Fix:"));
/// ```
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum BackendError {
    /// Device ran out of memory during buffer allocation or dispatch.
    #[error(
        "device out of memory: requested {requested} bytes, {available} available.          Fix: reduce buffer sizes or split the dispatch into smaller chunks."
    )]
    DeviceOutOfMemory {
        /// Bytes requested that triggered the OOM condition.
        requested: u64,
        /// Bytes reported available at the time of the failure.
        available: u64,
    },

    /// The acquired device generation was lost and all native handles are stale.
    #[error(
        "device generation {generation} was lost on backend `{backend}` device `{device}`: {message}. Fix: reacquire the registered materializer and rematerialize the authenticated artifact before retrying."
    )]
    DeviceLost {
        /// Registered backend identifier.
        backend: String,
        /// Backend-local physical or logical device identifier.
        device: String,
        /// Invalidated device generation.
        generation: u64,
        /// Concrete device-loss detail.
        message: String,
    },

    /// The backend does not support a required feature.
    #[error(
        "unsupported feature `{name}` on backend `{backend}`.          Fix: check backend capability before using this feature, or select a backend that supports it."
    )]
    UnsupportedFeature {
        /// Feature name (e.g. `"subgroup_ops"`, `"f16"`).
        name: String,
        /// Backend identifier (matches [`crate::backend::VyreBackend::id`]).
        backend: String,
    },

    /// Internal lock poisoning was detected during backend synchronization.
    #[error(
        "backend lock poisoned: {lock_error}. Fix: report the panic origin, prevent panics on lock guards, and retry the backend operation."
    )]
    PoisonedLock {
        /// Diagnostic details from the poison error.
        lock_error: String,
    },

    /// GPU kernel-source compilation failed.
    ///
    /// "Shader" in the variant name is historical and generalised
    ///  -  the code applies to any kernel-source compile failure across
    /// backends. A 2.0 rename to
    /// `KernelCompileFailed` is tracked in the semver-policy doc.
    #[error(
        "kernel-source compile failed on backend `{backend}`: {compiler_message} Fix: validate the vyre IR before lowering and check the lowered kernel source for type errors."
    )]
    KernelCompileFailed {
        /// Backend identifier.
        backend: String,
        /// Compiler error text or lowered shader / IR excerpt.
        compiler_message: String,
    },

    /// Command dispatch or GPU queue submission failed.
    #[error(
        "dispatch failed (code {code:?}): {message}. Fix: inspect the backend error code and queue state, reduce dispatch pressure, or reacquire the backend before retrying."
    )]
    DispatchFailed {
        /// Optional backend-specific numeric error code.
        code: Option<i32>,
        /// Human-readable failure detail.
        message: String,
    },

    /// Foundation validation rejected the program with a structured issue.
    #[error("{source}")]
    Validation {
        /// Structured foundation-owned validation issue.
        #[source]
        source: vyre_foundation::validate::ValidationError,
    },

    /// The program is structurally invalid for this backend.
    #[error("{fix}")]
    InvalidProgram {
        /// Actionable description, should begin with `Fix: `.
        fix: String,
    },

    /// A cooperative whole-grid launch could not be made fully resident: the
    /// grid has more blocks than the device can co-schedule for a grid-sync
    /// barrier. The orchestrator must fall back (loudly) to a recall-identical
    /// non-cooperative path rather than launch a kernel that would deadlock.
    #[error(
        "cooperative grid-sync launch needs {grid_blocks} co-resident block(s) but the device can fit at most {resident_limit}. Fix: route this dispatch to the resident-fixpoint or host-split grid-sync path, reduce the grid/workgroup size, or lower kernel register/shared-memory pressure. Detail: {detail}"
    )]
    CooperativeResidencyExceeded {
        /// Blocks the launch geometry requires.
        grid_blocks: u64,
        /// Blocks the device can keep co-resident for this kernel.
        resident_limit: u64,
        /// Which residency bound tripped (thread vs occupancy) and the geometry.
        detail: String,
    },

    /// Actionable backend failure without a more specific structured class.
    #[error("{0}")]
    Other(String),
}

impl BackendError {
    /// Build an unclassified backend error from a complete actionable message.
    ///
    /// The message is preserved verbatim. Callers that can identify the
    /// failure class use a structured variant instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use vyre_driver::BackendError;
    ///
    /// let err = BackendError::new("queue full. Fix: retry with a smaller dispatch size.");
    /// assert_eq!(err.to_string(), "queue full. Fix: retry with a smaller dispatch size.");
    /// ```
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }

    /// Build an actionable unsupported-extension error for opaque IR payloads.
    #[must_use]
    pub fn unsupported_extension(
        backend: impl Into<String>,
        extension_kind: &str,
        debug_identity: &str,
    ) -> Self {
        Self::UnsupportedFeature {
            name: format!("opaque IR extension `{extension_kind}`/`{debug_identity}`"),
            backend: backend.into(),
        }
    }

    /// Build a structured lock-poisoning error.
    ///
    /// This constructor accepts any `PoisonError` from `RwLock` operations
    /// and returns an actionable error carrying the root poison metadata.
    pub fn poisoned_lock<T>(error: std::sync::PoisonError<T>) -> Self {
        Self::PoisonedLock {
            lock_error: error.to_string(),
        }
    }

    /// Human-readable failure message, equivalent to [`ToString::to_string`].
    ///
    /// Prefer explicit `match` on variants or [`ErrorCode`] for programmatic
    /// error handling; avoid string-parsing this output.
    #[must_use]
    pub fn message(&self) -> String {
        self.to_string()
    }

    /// Consume this error and return its message string.
    ///
    /// Useful in `map_err` chains that expect `String`.
    #[must_use]
    pub fn into_message(self) -> String {
        self.to_string()
    }

    /// Machine-readable error code for programmatic error handling.
    ///
    /// Use this to drive retry logic, circuit breakers, and alerting
    /// without parsing human-readable message strings.
    #[must_use]
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::DeviceOutOfMemory { .. } => ErrorCode::DeviceOutOfMemory,
            Self::DeviceLost { .. } => ErrorCode::DeviceLost,
            Self::UnsupportedFeature { .. } => ErrorCode::UnsupportedFeature,
            Self::PoisonedLock { .. } => ErrorCode::PoisonedLock,
            Self::KernelCompileFailed { .. } => ErrorCode::KernelCompileFailed,
            Self::DispatchFailed { .. } => ErrorCode::DispatchFailed,
            Self::Validation { .. } => ErrorCode::InvalidProgram,
            Self::InvalidProgram { .. } => ErrorCode::InvalidProgram,
            Self::CooperativeResidencyExceeded { .. } => ErrorCode::CooperativeResidencyExceeded,
            Self::Other(_) => ErrorCode::Unknown,
        }
    }
}
