//! Resident dispatch step, buffer sets, read ranges, and dispatch error types.

use crate::ir::Program;

/// One resident-buffer kernel launch in an ordered optimizer sequence.
pub struct ResidentDispatchStep<'a> {
    /// Program to launch.
    pub program: &'a Program,
    /// Resident handle ids in canonical buffer binding order.
    pub handle_ids: &'a [u64],
    /// Optional launch grid override.
    pub grid_override: Option<[u32; 3]>,
}

/// One byte range to read from a resident buffer after an ordered sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentReadRange {
    /// Resident handle id.
    pub handle_id: u64,
    /// First byte to read from the device buffer.
    pub byte_offset: usize,
    /// Number of meaningful bytes to transfer.
    pub byte_len: usize,
}

/// Resident handles for immutable payloads that may stay device-resident
/// across optimizer calls.
///
/// `retained_by_dispatcher` means the dispatcher owns the handles after the
/// caller is done with the current launch sequence. Call
/// [`super::ProgramDispatcher::release_resident_static_uploads`] instead of
/// `free_resident` so a dispatcher with a device-side cache can keep read-only
/// graph and arena buffers hot while one without frees them immediately.
#[derive(Debug)]
pub struct ResidentStaticBufferSet {
    /// Resident handle ids in the same order as the payload slice passed to
    /// `acquire_resident_static_uploads`.
    pub handles: Vec<u64>,
    /// True when the handles were already resident and no host upload was paid.
    pub cache_hit: bool,
    /// True when the dispatcher retained ownership for future reuse.
    pub retained_by_dispatcher: bool,
}

/// Errors a dispatcher may surface. Concrete backends compose their
/// own error types into this; the orchestrator only needs the
/// boundary message.
#[derive(Debug)]
pub enum DispatchError {
    /// The dispatcher rejected the Program. The string carries the
    /// backend's actionable message (must contain `Fix:`).
    Rejected(String),
    /// Input arity or shape did not match the Program's declared
    /// buffer set. Hard error  -  not retryable.
    BadInputs(String),
    /// Backend raised an internal error. Same shape as `Rejected` but
    /// the cause is in the backend, not the Program.
    BackendError(String),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(msg) => write!(f, "dispatcher rejected program: {msg}"),
            Self::BadInputs(msg) => write!(f, "dispatcher input mismatch: {msg}"),
            Self::BackendError(msg) => write!(f, "dispatcher backend error: {msg}"),
        }
    }
}

impl std::error::Error for DispatchError {}
