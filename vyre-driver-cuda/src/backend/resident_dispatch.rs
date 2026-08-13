//! CUDA dispatch path for long-lived resident buffers.

#[allow(dead_code)]
const _DISPATCH_MARKERS: &str = "dispatch_resident ptx";

#[path = "resident_dispatch/async_dispatch.rs"]
mod async_dispatch;
#[path = "resident_dispatch/batch.rs"]
mod batch;
#[path = "resident_dispatch/borrowed.rs"]
mod borrowed;
#[path = "resident_dispatch/dense_index_validation.rs"]
mod dense_index_validation;
#[path = "resident_dispatch/descriptor_cursor.rs"]
mod descriptor_cursor;
#[path = "resident_dispatch/host_uploads.rs"]
mod host_uploads;
#[path = "resident_dispatch/sequence_api.rs"]
mod sequence_api;
#[path = "resident_dispatch/sequence_fused.rs"]
mod sequence_fused;
#[path = "resident_dispatch/sequence_slots.rs"]
mod sequence_slots;
#[path = "resident_dispatch/sync.rs"]
mod sync;
#[path = "resident_dispatch/timed.rs"]
mod timed;

#[cfg(test)]
#[path = "resident_dispatch/tests/mod.rs"]
mod tests;

pub(crate) use crate::backend::resident_dispatch_support::CudaResidentDispatch;
pub(crate) use descriptor_cursor::next_dispatch_binding;

use std::sync::Arc;

use smallvec::SmallVec;
use vyre_driver::DispatchConfig;
use vyre_foundation::ir::Program;

use crate::backend::plan::CudaDispatchPlan;
use crate::backend::resident::CudaResidentBuffer;

/// One resident dispatch step after its PTX, module key, and binding plan have
/// been resolved.
pub(crate) struct PreparedStep<'a> {
    pub(crate) program: &'a Program,
    pub(crate) handles: SmallVec<[CudaResidentBuffer; 8]>,
    pub(crate) config: &'a DispatchConfig,
    pub(crate) ptx_src: Arc<str>,
    pub(crate) module_key: crate::backend::module_cache::ModuleCacheKey,
    pub(crate) prepared: CudaDispatchPlan,
}
