use std::sync::Arc;

use vyre::ir::Program;
use vyre_driver::DispatchConfig;
use vyre_driver::{ArtifactMaterializer, BackendError, OutputBuffers, Resource};

use crate::api::case::{BenchContext, BenchError};
use crate::api::metric::elapsed_ns;
use crate::api::resident::{
    allocate_and_upload_resident_payloads, allocate_and_upload_resident_set,
    non_empty_upload_count, optional_resident_upload, program_order_resident_payloads,
    resident_set_resource_count, ResidentDispatch, ResidentResourcePayload,
};

/// Batched resident dispatch outputs plus batch-level wall and per-item device
/// timing.
pub struct ResidentBatchDispatch {
    pub outputs: Vec<OutputBuffers>,
    pub wall_ns_total: u64,
    /// Every batch row's device duration, in submission order.
    ///
    /// `None` when any row reported no positive device timestamp: a partial set
    /// cannot summarize the batch.
    pub device_ns_by_item: Option<Vec<u64>>,
    pub batch_len: usize,
}

impl ResidentBatchDispatch {
    /// Conservative per-item wall latency for steady-state batch throughput.
    pub fn per_item_wall_ns(&self) -> u64 {
        if self.batch_len == 0 {
            return self.wall_ns_total;
        }
        self.wall_ns_total.saturating_add(self.batch_len as u64 - 1) / self.batch_len as u64
    }

    /// Per-item device duration when every batch row reported a positive device
    /// timestamp.
    ///
    /// The rows are independent launches, so the summary is their median at the
    /// same nearest-rank p50 the suite reports for every other distribution. A
    /// mean was reported here before, and one launch delayed by contention
    /// outside the process moved the whole batch: with 16 rows a single 2 ms
    /// outlier raises the reported figure by more than 100 us, and the outlier
    /// population measured on a display-serving device hits roughly half of all
    /// batches. A median states what a row costs; a mean states what the worst
    /// row costs divided by 16.
    pub fn per_item_device_ns(&self) -> Option<u64> {
        let by_item = self.device_ns_by_item.as_ref()?;
        if by_item.is_empty() {
            return None;
        }
        let mut sorted = by_item.clone();
        sorted.sort_unstable();
        Some(batch_median_ns(&sorted))
    }
}

/// The p50 of an ascending sample set, at the suite's nearest-rank convention:
/// `ceil((n - 1) * 0.5)`, so an even count reports the upper middle row rather
/// than averaging the two.
fn batch_median_ns(ascending: &[u64]) -> u64 {
    let last = ascending.len().saturating_sub(1);
    let index = ((last as f64) * 0.5).ceil() as usize;
    ascending[index.min(last)]
}

/// Rotating pool of resident input-buffer sets for persistent benchmarks.
///
/// Persistent megakernel-style cases dispatch compiled handles repeatedly. A
/// single resident set can create false dependencies between measured samples;
/// a small rotating pool lets the benchmark keep host uploads outside the hot
/// path until the pool wraps.
pub struct ResidentInputPool {
    materializer: Arc<dyn ArtifactMaterializer>,
    sets: Vec<Vec<Resource>>,
    input_count: usize,
    next_set: usize,
    cleanup_label: &'static str,
}

/// Submit one materialized artifact through a resident pool when present.
pub fn dispatch_artifact_timed(
    ctx: &BenchContext,
    program: &Program,
    resident: Option<&mut ResidentInputPool>,
    inputs: &[Vec<u8>],
    config: &DispatchConfig,
) -> Result<ResidentDispatch, BenchError> {
    if let Some(resident) = resident {
        let resources = resident.next_set(inputs)?;
        let timed = ctx
            .dispatch_resident_timed(program, resources, config)
            .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
        return Ok(ResidentDispatch {
            timed,
            resident_used: true,
        });
    }
    let timed = ctx
        .dispatch_timed(program, inputs, config)
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    Ok(ResidentDispatch {
        timed,
        resident_used: false,
    })
}

impl ResidentInputPool {
    /// Upload `set_count` copies of `inputs` and return `None` when residency is unsupported.
    pub fn upload_optional(
        ctx: &BenchContext,
        inputs: &[Vec<u8>],
        set_count: usize,
        cleanup_label: &'static str,
    ) -> Result<Option<Self>, BenchError> {
        optional_resident_upload(Self::upload(ctx, inputs, set_count, cleanup_label))
    }

    /// Upload `set_count` copies of input resources plus zero-filled output resources.
    pub fn upload_with_zeroed_outputs_optional(
        ctx: &BenchContext,
        inputs: &[Vec<u8>],
        output_sizes: &[usize],
        set_count: usize,
        cleanup_label: &'static str,
    ) -> Result<Option<Self>, BenchError> {
        optional_resident_upload(Self::upload_with_zeroed_outputs(
            ctx,
            inputs,
            output_sizes,
            set_count,
            cleanup_label,
        ))
    }

    /// Upload `set_count` copies of host inputs plus zero-filled outputs in program binding order.
    pub fn upload_program_ordered_with_zeroed_outputs_optional(
        ctx: &BenchContext,
        program: &Program,
        inputs: &[Vec<u8>],
        set_count: usize,
        cleanup_label: &'static str,
    ) -> Result<Option<Self>, BenchError> {
        let payloads = program_order_resident_payloads(program, inputs, cleanup_label)?;
        optional_resident_upload(Self::upload_payloads(
            ctx,
            &payloads,
            set_count,
            cleanup_label,
        ))
    }

    /// Return the next resident input set, re-uploading when the pool wraps.
    pub fn next_set<'a>(&'a mut self, inputs: &[Vec<u8>]) -> Result<&'a [Resource], BenchError> {
        if self.sets.is_empty() {
            return Err(BenchError::ExecutionFailed(format!(
                "{} resident pool is empty",
                self.cleanup_label
            )));
        }
        let index = self.next_set % self.sets.len();
        if self.input_count != inputs.len() {
            return Err(BenchError::ExecutionFailed(format!(
                "{} resident pool input count changed: pool has {}, caller passed {}",
                self.cleanup_label,
                self.input_count,
                inputs.len()
            )));
        }
        if self.next_set >= self.sets.len() {
            upload_resident_inputs(
                self.materializer.as_ref(),
                &self.sets[index][..self.input_count],
                inputs,
            )
            .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
        }
        self.next_set = self.next_set.saturating_add(1);
        Ok(&self.sets[index])
    }

    /// Dispatch the first `batch_len` resident sets through one materialized
    /// artifact, as one batch.
    ///
    /// The whole batch is submitted before any item is awaited, so an item is
    /// already enqueued when the item before it finishes and each item's device
    /// timer covers its own launch instead of a host round trip. `wall_ns_total`
    /// covers the whole batch, which is what a per-item host window cannot
    /// describe once the items overlap.
    pub fn dispatch_artifact_batch_timed(
        &self,
        ctx: &BenchContext,
        program: &Program,
        batch_len: usize,
        config: &DispatchConfig,
    ) -> Result<ResidentBatchDispatch, BackendError> {
        if batch_len == 0 {
            return Err(BackendError::new(
                "resident artifact batch dispatch requires at least one resident set. Fix: configure a positive resident batch size.",
            ));
        }
        if batch_len > self.sets.len() {
            return Err(BackendError::new(format!(
                "resident artifact batch dispatch requested {batch_len} set(s) but pool has {}. Fix: upload a resident pool at least as large as the requested batch.",
                self.sets.len()
            )));
        }
        let mut sets = Vec::with_capacity(batch_len);
        sets.extend(self.sets[..batch_len].iter().map(Vec::as_slice));
        let started = std::time::Instant::now();
        let dispatched = ctx.dispatch_resident_batch_timed(program, &sets, config)?;
        let wall_ns_total = elapsed_ns(started);
        let mut outputs = Vec::with_capacity(dispatched.len());
        let mut device_ns_by_item = Some(Vec::with_capacity(dispatched.len()));
        for (item_outputs, device_ns) in dispatched {
            device_ns_by_item = match (device_ns_by_item, device_ns.filter(|&ns| ns > 0)) {
                (Some(mut rows), Some(ns)) => {
                    rows.push(ns);
                    Some(rows)
                }
                _ => None,
            };
            outputs.push(item_outputs);
        }
        Ok(ResidentBatchDispatch {
            outputs,
            wall_ns_total,
            device_ns_by_item,
            batch_len,
        })
    }

    /// Re-upload one resource index in every resident set.
    ///
    /// An empty payload uploads nothing. A resident allocation is sized once, and
    /// a backend checks a full upload against that size, so a zero-length upload
    /// is a length mismatch rather than a no-op. A case whose program overwrites
    /// every output word it declares has nothing to reset and passes one.
    pub fn upload_resource_to_all_sets(
        &self,
        index: usize,
        payload: &[u8],
        context: &str,
    ) -> Result<(), BenchError> {
        if payload.is_empty() {
            return Ok(());
        }
        let mut uploads = Vec::with_capacity(self.sets.len());
        for (set_index, set) in self.sets.iter().enumerate() {
            let resource = set.get(index).ok_or_else(|| {
                BenchError::ExecutionFailed(format!(
                    "{context} resident pool set {set_index} is missing resource at index {index}"
                ))
            })?;
            uploads.push((resource, payload));
        }
        for (resource, bytes) in uploads {
            self.materializer
                .upload_resident(resource, bytes)
                .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
        }
        Ok(())
    }

    fn upload(
        ctx: &BenchContext,
        inputs: &[Vec<u8>],
        set_count: usize,
        cleanup_label: &'static str,
    ) -> Result<Self, BackendError> {
        Self::upload_with_zeroed_outputs(ctx, inputs, &[], set_count, cleanup_label)
    }

    fn upload_with_zeroed_outputs(
        ctx: &BenchContext,
        inputs: &[Vec<u8>],
        output_sizes: &[usize],
        set_count: usize,
        cleanup_label: &'static str,
    ) -> Result<Self, BackendError> {
        if set_count == 0 {
            return Ok(Self {
                materializer: Arc::clone(&ctx.materializer),
                sets: Vec::new(),
                input_count: inputs.len(),
                next_set: 0,
                cleanup_label,
            });
        }

        let materializer = Arc::clone(&ctx.materializer);
        let mut sets = Vec::with_capacity(set_count);
        let mut zero_scratch = Vec::new();
        let result = (|| {
            for _ in 0..set_count {
                sets.push(Vec::with_capacity(resident_set_resource_count(
                    inputs,
                    output_sizes,
                )));
                let resource_index = sets.len() - 1;
                allocate_and_upload_resident_set(
                    materializer.as_ref(),
                    &mut sets[resource_index],
                    inputs,
                    output_sizes,
                    &mut zero_scratch,
                )?;
            }
            Ok(())
        })();

        if let Err(error) = result {
            for set in sets {
                for resource in set {
                    if let Err(cleanup_error) = materializer.free_resident(resource) {
                        eprintln!(
                            "{cleanup_label} resident pool rollback cleanup failed: {cleanup_error}"
                        );
                    }
                }
            }
            return Err(error);
        }

        Ok(Self {
            materializer,
            sets,
            input_count: inputs.len(),
            next_set: 0,
            cleanup_label,
        })
    }

    fn upload_payloads(
        ctx: &BenchContext,
        payloads: &[ResidentResourcePayload<'_>],
        set_count: usize,
        cleanup_label: &'static str,
    ) -> Result<Self, BackendError> {
        if set_count == 0 {
            return Ok(Self {
                materializer: Arc::clone(&ctx.materializer),
                sets: Vec::new(),
                input_count: payloads
                    .iter()
                    .filter(|payload| matches!(payload, ResidentResourcePayload::Input(_)))
                    .count(),
                next_set: 0,
                cleanup_label,
            });
        }

        let materializer = Arc::clone(&ctx.materializer);
        let mut sets = Vec::with_capacity(set_count);
        let mut zero_scratch = Vec::new();
        let result = (|| {
            for _ in 0..set_count {
                sets.push(Vec::with_capacity(payloads.len()));
                let resource_index = sets.len() - 1;
                allocate_and_upload_resident_payloads(
                    materializer.as_ref(),
                    &mut sets[resource_index],
                    payloads,
                    &mut zero_scratch,
                )?;
            }
            Ok(())
        })();

        if let Err(error) = result {
            for set in sets {
                for resource in set {
                    if let Err(cleanup_error) = materializer.free_resident(resource) {
                        eprintln!(
                            "{cleanup_label} resident pool rollback cleanup failed: {cleanup_error}"
                        );
                    }
                }
            }
            return Err(error);
        }

        Ok(Self {
            materializer,
            sets,
            input_count: payloads
                .iter()
                .filter(|payload| matches!(payload, ResidentResourcePayload::Input(_)))
                .count(),
            next_set: 0,
            cleanup_label,
        })
    }
}

impl Drop for ResidentInputPool {
    fn drop(&mut self) {
        for set in self.sets.drain(..) {
            for resource in set {
                if let Err(error) = self.materializer.free_resident(resource) {
                    eprintln!(
                        "{} resident pool cleanup failed: {error}",
                        self.cleanup_label
                    );
                }
            }
        }
    }
}

fn upload_resident_inputs(
    materializer: &dyn ArtifactMaterializer,
    resources: &[Resource],
    inputs: &[Vec<u8>],
) -> Result<(), BackendError> {
    let mut uploads = Vec::with_capacity(non_empty_upload_count(inputs, &[]));
    for (resource, input) in resources.iter().zip(inputs.iter()) {
        if !input.is_empty() {
            uploads.push((resource, input.as_slice()));
        }
    }
    for (resource, bytes) in uploads {
        materializer.upload_resident(resource, bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: a resident throughput batch summarizes independent launches, so one
    /// row delayed by contention outside the process must not move the figure
    /// the release contract reads. The mean this replaced moved with every
    /// outlier: sixteen rows of 10 us plus one 2 ms row averaged above 127 us.
    /// Does not catch a device timer that is wrong on every row at once; that
    /// is a driver contract, asserted where the timing events are recorded.
    #[test]
    fn resident_batch_device_summary_is_the_nearest_rank_median() {
        // Sixteen rows: fifteen at 10_000 and one 2 ms outlier. The mean is
        // 134_375; the p50 row is 10_000.
        let mut rows = vec![10_000_u64; 15];
        rows.push(2_000_000);
        let poisoned = ResidentBatchDispatch {
            outputs: Vec::new(),
            wall_ns_total: 41,
            device_ns_by_item: Some(rows),
            batch_len: 16,
        };
        assert_eq!(poisoned.per_item_device_ns(), Some(10_000));

        // Even count: the upper middle row, matching `ceil((n - 1) * 0.5)`.
        let even = ResidentBatchDispatch {
            device_ns_by_item: Some(vec![4, 1, 3, 2]),
            batch_len: 4,
            ..poisoned
        };
        assert_eq!(even.per_item_device_ns(), Some(3));
        assert_eq!(even.per_item_wall_ns(), 11);

        // Odd count: the exact middle row.
        let odd = ResidentBatchDispatch {
            device_ns_by_item: Some(vec![9, 1, 5]),
            batch_len: 3,
            ..even
        };
        assert_eq!(odd.per_item_device_ns(), Some(5));

        // One row: itself.
        let single = ResidentBatchDispatch {
            device_ns_by_item: Some(vec![7]),
            batch_len: 1,
            ..odd
        };
        assert_eq!(single.per_item_device_ns(), Some(7));

        // A row without a positive device timestamp voids the whole batch
        // rather than summarizing the rows that happened to report.
        let incomplete = ResidentBatchDispatch {
            device_ns_by_item: None,
            ..single
        };
        assert_eq!(incomplete.per_item_device_ns(), None);

        let empty = ResidentBatchDispatch {
            device_ns_by_item: Some(Vec::new()),
            batch_len: 0,
            ..incomplete
        };
        assert_eq!(empty.per_item_device_ns(), None);
    }

    /// A materializer that refuses every resident upload, so an attempted upload
    /// is observable as an error and a skipped one as `Ok`.
    struct RefusingMaterializer {
        device: vyre_driver::materialize::MaterializerDevice,
    }

    impl vyre_driver::ArtifactMaterializer for RefusingMaterializer {
        fn device(&self) -> &dyn vyre_driver::Device {
            &self.device
        }

        fn materialize(
            &self,
            _artifact: &vyre::compiler::Artifact,
            _payload: &vyre::compiler::TargetPayload,
        ) -> Result<Box<dyn vyre_driver::ArtifactInstance>, BackendError> {
            Err(BackendError::UnsupportedFeature {
                name: "materialize".to_string(),
                backend: "refusing".to_string(),
            })
        }
    }

    fn pool_with_one_empty_set() -> ResidentInputPool {
        let profile = vyre::compiler::TargetProfile::new("refusing", 1, [64, 1, 1], 64, 1_024, 0)
            .expect("Fix: the fixture target profile must be valid.");
        let device = vyre_driver::materialize::MaterializerDevice::acquire(
            vyre_driver::materialize::DeviceSpec {
                backend: "refusing",
                device: "refusing-device".to_string(),
                format_extension: "refusing",
                format_version: 1,
                profile,
            },
        )
        .expect("Fix: the fixture device must acquire.");
        ResidentInputPool {
            materializer: Arc::new(RefusingMaterializer { device }),
            sets: vec![Vec::new()],
            input_count: 0,
            next_set: 0,
            cleanup_label: "fixture",
        }
    }

    /// WHY: a resident allocation is sized once and a backend checks a full
    /// upload against that size, so a zero-length reset is a length mismatch
    /// rather than a no-op. A release case whose program overwrites every output
    /// word it declares carries an empty reset payload, and that case must not
    /// fail at the reset. The pool short-circuits before it resolves a resource,
    /// which is what the second assertion pins: a non-empty payload still walks
    /// the sets.
    #[test]
    fn an_empty_reset_payload_uploads_nothing() {
        let pool = pool_with_one_empty_set();

        pool.upload_resource_to_all_sets(0, &[], "fixture")
            .expect("Fix: an empty reset payload must upload nothing.");

        let error = pool
            .upload_resource_to_all_sets(0, &[0u8; 4], "fixture")
            .expect_err("Fix: a non-empty reset payload must still resolve its resource.");
        assert!(
            error.to_string().contains("missing resource at index 0"),
            "unexpected error: {error}"
        );
    }
}
