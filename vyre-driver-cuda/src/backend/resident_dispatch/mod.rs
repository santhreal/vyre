//! CUDA dispatch path for long-lived resident buffers.

#[allow(dead_code)]
const _DISPATCH_MARKERS: &str = "dispatch_resident ptx";

mod async_dispatch;
mod batch;
mod borrowed;
mod dense_index_validation;
mod descriptor_cursor;
mod host_uploads;
mod sequence_api;
mod sequence_fused;
mod sequence_slots;
mod sync;
mod timed;

#[cfg(test)]
mod tests;

pub(crate) use crate::backend::resident_dispatch_accounting::CudaResidentDispatch;
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
