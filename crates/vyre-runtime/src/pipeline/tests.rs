//! Pipeline cache limits, identity hashing and environment parsing.

use super::*;
use crate::backend::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;

#[test]
fn parse_positive_env_rejects_unset_zero_and_invalid() {
    let name = "VYRE_TEST_PARSE_POSITIVE_ENV_UNIQUE";
    std::env::remove_var(name);
    assert_eq!(super::parse_positive_env::<u32>(name, 7), 7);
    std::env::set_var(name, "0");
    assert_eq!(super::parse_positive_env::<u32>(name, 7), 7);
    std::env::set_var(name, "not-a-number");
    assert_eq!(super::parse_positive_env::<usize>(name, 9), 9);
    std::env::set_var(name, "42");
    assert_eq!(super::parse_positive_env::<u32>(name, 7), 42);
    std::env::remove_var(name);
}

mod cache_identity {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};

    fn store_program(output_name: &'static str, value: u32) -> Program {
        Program::wrapped(
            vec![BufferDecl::output(output_name, 0, DataType::U32).with_count(1)],
            [64, 1, 1],
            vec![Node::store(output_name, Expr::u32(0), Expr::u32(value))],
        )
    }

    #[test]
    fn normalized_program_digest_tracks_program_structure() {
        let a = store_program("out", 7);
        let b = store_program("out", 8);
        assert_ne!(
            try_normalized_program_cache_digest(&a).unwrap(),
            try_normalized_program_cache_digest(&b).unwrap(),
            "Fix: backend shader caches must miss when program semantics change."
        );
        assert_eq!(
            try_normalized_program_cache_digest(&a).unwrap(),
            try_normalized_program_cache_digest(&a).unwrap(),
            "Fix: normalized program cache digests must be deterministic."
        );
    }

    #[test]
    fn dispatch_policy_cache_hash_tracks_codegen_policy() {
        let base = DispatchConfig {
            ulp_budget: Some(1),
            workgroup_override: Some([64, 1, 1]),
            ..Default::default()
        };
        let mut changed = base.clone();
        changed.workgroup_override = Some([128, 1, 1]);

        let mut a = blake3::Hasher::new();
        update_dispatch_policy_cache_hash(&mut a, &base);
        let mut b = blake3::Hasher::new();
        update_dispatch_policy_cache_hash(&mut b, &changed);

        assert_ne!(a.finalize(), b.finalize());
        assert_eq!(
            dispatch_policy_cache_string(&base),
            "ulp=Some(1):wg=Some([64, 1, 1]):float=contracted"
        );
    }
}

mod compiled_pipeline_defaults {
    use super::*;
    use crate::backend::CompiledPipeline;
    use crate::Resource;

    /// A batched timed dispatch reports one device duration per item, in
    /// submission order, and an empty batch reports nothing.
    ///
    /// WHY: a resident batch summary is computed from these entries, so an
    /// entry that lands out of order, is dropped, or survives from a
    /// previous batch attributes one item's device time to another item and
    /// silently changes every figure derived from it. This default is what
    /// a backend without a batched submission path uses, so it holds the
    /// same contract as a backend override.
    ///
    /// Does not catch whether an override enqueues the whole batch before
    /// awaiting any of it, which is only observable on a device.
    #[test]
    fn compiled_pipeline_batched_timed_default_reports_one_device_time_per_item() {
        struct TimedBatchPipeline;

        impl crate::backend::sealed::Sealed for TimedBatchPipeline {}

        impl CompiledPipeline for TimedBatchPipeline {
            fn id(&self) -> &str {
                "batched-timed-default"
            }

            fn dispatch_borrowed_timed(
                &self,
                _: &[&[u8]],
                _: &DispatchConfig,
            ) -> Result<crate::TimedDispatchResult, BackendError> {
                panic!("batched timed default must route through resident-handle dispatch")
            }

            fn dispatch_persistent_handles_timed(
                &self,
                inputs: &[Resource],
                _: &DispatchConfig,
            ) -> Result<crate::TimedDispatchResult, BackendError> {
                let Some(Resource::Borrowed(bytes)) = inputs.first() else {
                    return Err(BackendError::new(
                        "batched timed test item requires one borrowed resource. Fix: bind one resource per batch item.",
                    ));
                };
                let device_ns = u64::from(bytes[0]) * 1000;
                Ok(crate::TimedDispatchResult::device_timed(
                    vec![bytes.clone()],
                    7,
                    Some(device_ns),
                ))
            }
        }

        let pipeline = TimedBatchPipeline;
        let items = [
            vec![Resource::Borrowed(vec![3_u8])],
            vec![Resource::Borrowed(vec![1_u8])],
            vec![Resource::Borrowed(vec![2_u8])],
        ];
        let batches: Vec<&[Resource]> = items.iter().map(Vec::as_slice).collect();
        let mut outputs = vec![vec![Vec::with_capacity(8)]];
        let first_slot_ptr = outputs[0][0].as_ptr();
        let mut device_ns_by_item = vec![Some(999_u64)];

        pipeline
            .dispatch_persistent_handles_batched_timed(
                &batches,
                &DispatchConfig::default(),
                &mut outputs,
                &mut device_ns_by_item,
            )
            .unwrap();

        assert_eq!(
            device_ns_by_item,
            vec![Some(3000), Some(1000), Some(2000)],
            "each item reports its own device time, in submission order"
        );
        assert_eq!(
            outputs,
            vec![vec![vec![3_u8]], vec![vec![1_u8]], vec![vec![2_u8]]]
        );
        assert_eq!(
            outputs[0][0].as_ptr(),
            first_slot_ptr,
            "an existing output slot is reused rather than reallocated"
        );

        pipeline
            .dispatch_persistent_handles_batched_timed(
                &[],
                &DispatchConfig::default(),
                &mut outputs,
                &mut device_ns_by_item,
            )
            .unwrap();

        assert!(device_ns_by_item.is_empty(), "an empty batch times nothing");
        assert!(outputs.is_empty(), "an empty batch produces no output set");
    }

    /// A failing item stops the batch with its error and leaves no timing
    /// row for an item that did not run.
    #[test]
    fn compiled_pipeline_batched_timed_default_propagates_item_error() {
        struct FailingSecondItem;

        impl crate::backend::sealed::Sealed for FailingSecondItem {}

        impl CompiledPipeline for FailingSecondItem {
            fn id(&self) -> &str {
                "batched-timed-failure"
            }

            fn dispatch_borrowed_timed(
                &self,
                _: &[&[u8]],
                _: &DispatchConfig,
            ) -> Result<crate::TimedDispatchResult, BackendError> {
                panic!("batched timed default must route through resident-handle dispatch")
            }

            fn dispatch_persistent_handles_timed(
                &self,
                inputs: &[Resource],
                _: &DispatchConfig,
            ) -> Result<crate::TimedDispatchResult, BackendError> {
                match inputs.first() {
                    Some(Resource::Borrowed(bytes)) if bytes[0] == 0 => Err(BackendError::new(
                        "item zero rejected. Fix: submit nonzero items.",
                    )),
                    Some(Resource::Borrowed(bytes)) => Ok(
                        crate::TimedDispatchResult::device_timed(vec![bytes.clone()], 1, Some(5)),
                    ),
                    _ => panic!("test items are borrowed"),
                }
            }
        }

        let items = [
            vec![Resource::Borrowed(vec![4_u8])],
            vec![Resource::Borrowed(vec![0_u8])],
            vec![Resource::Borrowed(vec![6_u8])],
        ];
        let batches: Vec<&[Resource]> = items.iter().map(Vec::as_slice).collect();
        let mut outputs = Vec::new();
        let mut device_ns_by_item = Vec::new();

        let error = FailingSecondItem
            .dispatch_persistent_handles_batched_timed(
                &batches,
                &DispatchConfig::default(),
                &mut outputs,
                &mut device_ns_by_item,
            )
            .expect_err("the rejected item must fail the batch");

        assert!(error.to_string().contains("item zero rejected"), "{error}");
        assert_eq!(
            device_ns_by_item,
            vec![Some(5)],
            "only the item that ran is timed"
        );
    }
}
