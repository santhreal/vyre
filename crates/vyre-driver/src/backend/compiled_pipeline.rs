//! Backend executable-module dispatch contract.

use crate::backend::{
    sealed, BackendError, DispatchConfig, OutputBuffers, Resource, TimedDispatchResult,
};

/// A materialized backend executable ready for repeated dispatch.
///
/// Canonical artifact materializers construct concrete implementations after
/// authenticating a target payload, and materialized instances dispatch
/// through the three entry points below. Dispatch must remain bit-identical
/// across host and resident bindings for the same artifact ABI.
pub trait CompiledPipeline: sealed::Sealed + Send + Sync {
    /// Stable identifier for this pipeline (typically `<backend>:<program-fingerprint>`).
    ///
    /// Used by certificates and debugging to confirm a particular cached
    /// pipeline was reused vs recompiled.
    fn id(&self) -> &str;

    /// Dispatch the precompiled pipeline with borrowed host input rows and
    /// backend-owned timing.
    ///
    /// `device_ns` is `Some` when the backend measured device execution and
    /// `None` when it has no device timer for this dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the backend cannot complete dispatch.
    /// The error message always includes a `Fix: ` remediation section.
    fn dispatch_borrowed_timed(
        &self,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError>;

    /// Dispatch the precompiled pipeline with mixed host/resident handles and
    /// backend-owned timing.
    ///
    /// Resources are bound in public non-shared binding order. Resident
    /// handles stay on the device across launches; outputs are read back into
    /// the returned [`TimedDispatchResult`].
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the backend cannot complete dispatch.
    fn dispatch_persistent_handles_timed(
        &self,
        inputs: &[Resource],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError>;

    /// Dispatch several resident-handle submissions and report each item's
    /// device duration.
    ///
    /// A backend that can submit the whole batch before awaiting any of it
    /// overrides this. Item N's timing then starts when item N-1's work ends
    /// while item N is already submitted, so what it reports is the item's
    /// device time and not the host latency of submitting it. The default
    /// dispatches each item through its own submit-and-wait, which charges
    /// every item the launch latency of a stream that has nothing else on it.
    ///
    /// `outputs` is resized to one entry per batch item and reuses existing
    /// output slots. `device_ns_by_item` is replaced with one entry per batch
    /// item, in submission order, holding `None` for an item whose backend
    /// exposes no device timer.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when any item cannot complete dispatch.
    fn dispatch_persistent_handles_batched_timed(
        &self,
        batches: &[&[Resource]],
        config: &DispatchConfig,
        outputs: &mut Vec<OutputBuffers>,
        device_ns_by_item: &mut Vec<Option<u64>>,
    ) -> Result<(), BackendError> {
        crate::backend::resize_batch_output_slots(
            outputs,
            batches.len(),
            "compiled resident batch outputs",
        )?;
        device_ns_by_item.clear();
        device_ns_by_item
            .try_reserve_exact(batches.len())
            .map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: failed to reserve {} resident batch timing slot(s): {error}. Submit a smaller resident batch.",
                    batches.len()
                ),
            })?;
        for (batch, slot) in batches.iter().zip(outputs.iter_mut()) {
            let timed = self.dispatch_persistent_handles_timed(batch, config)?;
            device_ns_by_item.push(timed.device_ns);
            crate::replace_output_buffers_preserving_slots(timed.outputs, slot);
        }
        Ok(())
    }
}
