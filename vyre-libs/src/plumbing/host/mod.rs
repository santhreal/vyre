//! The host side of a dispatch.
//!
//! A composition ends at a `Program`. Everything between that Program and a
//! backend call is host work: marshalling operands to bytes, reserving the
//! scratch a release path reuses, keeping a specialized Program resident across
//! a hot loop, and counting the calls an operator reads back. None of it is a
//! dialect and every dialect that dispatches needs it.

pub mod dispatch_buffers;

// `device` was too broad: `analysis`, `encoding` and `scheduling` all name it
// and none of them consults a cache. The two that do are `graph-dispatch`
// (seven CSR wrappers) and `solvers` (six quantized dispatch scratch types).
#[cfg(any(feature = "graph-dispatch", feature = "solvers"))]
pub(crate) mod program_cache;

#[cfg(any(feature = "device", feature = "graph", feature = "math-kernels"))]
pub(crate) mod scratch;

#[cfg(feature = "telemetry")]
pub mod telemetry;
