//! GPU device abstraction and initialization.

pub use acquire::EnabledFeatures;
pub use acquire::{acquire_gpu, cached_adapter_info, cached_device, init_device};
pub(crate) use acquire::{poll_device_once, poll_device_wait_for, pop_error_scope_now};
pub use selector::{
    acquire_gpu_for_adapter, adapter_for_info, adapter_index_from_env, adapter_probe_report,
    enumerate_adapters, has_real_gpu_adapter, init_device_for_adapter, select_adapter,
    AdapterCriteria, AdapterProbeReport,
};
pub(crate) use selector::{init_device_for_adapter_identity, AdapterIdentity};

mod acquire;
mod selector;

/// Reserve capacity for an adapter-probe vector, reporting the failure as a
/// backend error.
///
/// Both submodules probe adapters and both need the same fallible reserve, so
/// the helper lives here rather than once per submodule.
fn reserve_probe_vec<T>(
    vec: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), vyre_driver::BackendError> {
    crate::staging_reserve::reserve_backend_vec(vec, additional, context)
        .map_err(|error| vyre_driver::BackendError::new(error.to_string()))
}
