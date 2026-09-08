//! WHY: roughly thirty io_uring and tenant-accounting faults were reported as
//! `PipelineError::QueueFull`, whose message reads `io_uring {queue} queue at capacity`.
//! A buffer offset past the registered region, a `usize::try_from` overflow,
//! a tenant retry counter wrapping u64, and a `drained_count` exceeding `published_count`
//! all rendered as a full submission queue, so the message contradicted the fault
//! and a caller could not select on it.
//!
//! Closes: runtime enumeration of the `PipelineError` variant space from source,
//! proof that every variant names its own fault distinctly, proof that only genuine
//! queue exhaustion reports `QueueFull`, and exact assertions for the selectable
//! identity and message text of the four faults named in backlog row 139.
//!
//! Does not catch: call sites that construct an incorrect variant for an unexercised
//! path. That is covered at each call site's own subsystem tests.

use std::collections::BTreeSet;

use vyre_runtime::resident_work_queue::protocol::ProtocolError;
use vyre_runtime::{
    CounterArithmetic, CounterScope, PipelineError, RequestFault, RingEncodingFault,
};

fn declared_pipeline_error_variants() -> BTreeSet<String> {
    let path = vyre_test_support::monorepo::vyre_crate_directory("vyre-runtime")
        .join("src")
        .join("lib.rs");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cannot read the PipelineError declaration at {path:?}: {err}"));
    let body = vyre_test_support::braced_body(&source, "pub enum PipelineError {")
        .unwrap_or_else(|| panic!("no `pub enum PipelineError` declaration in {path:?}"));
    vyre_test_support::top_level_variant_names(body)
}

struct VariantSpec {
    variant_name: &'static str,
    sample: PipelineError,
    discriminator: &'static str,
    io_uring: bool,
    claims_queue_capacity: bool,
}

fn all_variant_specs() -> Vec<VariantSpec> {
    vec![
        VariantSpec {
            variant_name: "IoUringSyscall",
            sample: PipelineError::IoUringSyscall {
                syscall: "io_uring_enter",
                errno: 11,
                fix: "retry the enter call",
            },
            discriminator: "failed: errno=",
            io_uring: true,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "QueueFull",
            sample: PipelineError::QueueFull {
                queue: "submission",
                depth: 64,
                fix: "reap completions, then submit again",
            },
            discriminator: "queue at capacity",
            io_uring: true,
            claims_queue_capacity: true,
        },
        VariantSpec {
            variant_name: "RegionBounds",
            sample: PipelineError::RegionBounds {
                region: "GpuMappedBuffer mapped allocation",
                offset: 8,
                len: 16,
                region_len: 16,
                unit: "bytes",
                fix: "reduce the slot size or enlarge the staging buffer",
            },
            discriminator: "is outside the region's",
            io_uring: true,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "IntegerWidth",
            sample: PipelineError::IntegerWidth {
                quantity: "io_uring ingest slot count",
                value: 4_294_967_296,
                bits: 32,
                fix: "reduce the ingest slot count so it fits the host index width",
            },
            discriminator: "does not fit",
            io_uring: true,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "CounterOverflow",
            sample: PipelineError::CounterOverflow {
                scope: CounterScope::TenantRegistry,
                counter: "registration retry count",
                arithmetic: CounterArithmetic::Sum,
                lhs: u64::MAX,
                rhs: 1,
                bits: 64,
                fix: "retry registration later; the id allocator has not settled",
            },
            discriminator: "leaves the",
            io_uring: false,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "CounterOrder",
            sample: PipelineError::CounterOrder {
                scope: CounterScope::Tenant(7),
                produced_counter: "published_count",
                produced: 5,
                consumed_counter: "drained_count",
                consumed: 8,
                fix: "rebuild this tenant's slot accounting; note_drained ran for slots the tenant never published",
            },
            discriminator: "counters are out of order",
            io_uring: false,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "InvalidRequest",
            sample: PipelineError::InvalidRequest {
                fault: RequestFault::LengthMismatch,
                quantity: "io_uring pump read length",
                observed: 8192,
                bound: 4096,
                fix: "submit exactly chunk_bytes per read, or construct a pump for this chunk size",
            },
            discriminator: "request rejected:",
            io_uring: true,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "SlotInFlight",
            sample: PipelineError::SlotInFlight {
                slot: 2,
                slot_count: 4,
                inflight_tag: 9,
                fix: "drain completions for this slot before reusing it",
            },
            discriminator: "already has a read in flight",
            io_uring: true,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "RingEncoding",
            sample: PipelineError::RingEncoding {
                fault: RingEncodingFault::OutOfBounds,
                fix: "publish into a slot the ring contains",
            },
            discriminator: "resident ring encode rejected",
            io_uring: false,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "Protocol",
            sample: PipelineError::Protocol(ProtocolError::ByteLengthOverflow {
                buffer: "ring",
                fix: "shard the ring",
            }),
            discriminator: "byte length overflow",
            io_uring: false,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "NotLinux",
            sample: PipelineError::NotLinux,
            discriminator: "is Linux-only",
            io_uring: true,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "NvmePassthroughDisabled",
            sample: PipelineError::NvmePassthroughDisabled,
            discriminator: "NVMe passthrough requires",
            io_uring: true,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "Backend",
            sample: PipelineError::Backend("device rejected the module".to_string()),
            discriminator: "backend error:",
            io_uring: false,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "DrainIncomplete",
            sample: PipelineError::DrainIncomplete {
                descriptor: "megakernel",
                claimed: 1,
                expected: 4,
                unit: "work-items",
            },
            discriminator: "drain incomplete",
            io_uring: false,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "IllegalSlotTransition",
            sample: PipelineError::IllegalSlotTransition {
                transition: "claim",
                permitted: &[1],
                current_status: 2,
                fix: "wait for the host to publish the slot",
            },
            discriminator: "illegal ring slot transition",
            io_uring: false,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "UnregisteredResource",
            sample: PipelineError::UnregisteredResource {
                request: "megakernel IO READ",
                resource: "GPU destination",
                handle: 9,
                slot: 3,
                fix: "register the destination before publishing READ requests",
            },
            discriminator: "which is not registered",
            io_uring: false,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "WorkerThreadPanicked",
            sample: PipelineError::WorkerThreadPanicked {
                worker: "IO loop",
                fix: "repair the fault the panic names, then respawn the loop",
            },
            discriminator: "thread panicked before it could be joined",
            io_uring: false,
            claims_queue_capacity: false,
        },
        VariantSpec {
            variant_name: "ReservedOpcode",
            sample: PipelineError::ReservedOpcode {
                tenant_id: 4,
                local_opcode: 2,
                global_opcode: 0x8000_0002,
                fix: "move the tenant opcode window below the reserved range",
            },
            discriminator: "reserved for system opcodes",
            io_uring: false,
            claims_queue_capacity: false,
        },
    ]
}

#[test]
fn every_declared_pipeline_error_variant_is_catalogued() {
    let declared = declared_pipeline_error_variants();
    let specs = all_variant_specs();
    let catalogued: BTreeSet<String> = specs
        .iter()
        .map(|spec| spec.variant_name.to_string())
        .collect();

    assert_eq!(
        declared, catalogued,
        "the source PipelineError enum and the test specification set disagree; a variant was added without test coverage"
    );
}

#[test]
fn every_variant_names_its_own_fault_and_no_other() {
    let specs = all_variant_specs();
    for spec in &specs {
        let rendered = spec.sample.to_string();
        assert!(
            rendered.contains(spec.discriminator),
            "variant {} must render its discriminator {:?}, got: {rendered}",
            spec.variant_name,
            spec.discriminator
        );
        for other in &specs {
            if other.variant_name == spec.variant_name {
                continue;
            }
            assert!(
                !rendered.contains(other.discriminator),
                "variant {} renders {:?}, which names the {} fault it did not observe: {rendered}",
                spec.variant_name,
                other.discriminator,
                other.variant_name
            );
        }
    }
}

#[test]
fn only_queue_full_claims_queue_is_at_capacity() {
    for spec in &all_variant_specs() {
        let rendered = spec.sample.to_string();
        if spec.claims_queue_capacity {
            assert!(
                rendered.contains("queue at capacity"),
                "QueueFull must state that a queue is at capacity: {rendered}"
            );
        } else {
            assert!(
                !rendered.contains("queue at capacity"),
                "variant {} must not claim a queue is at capacity: {rendered}",
                spec.variant_name
            );
        }
    }
}

#[test]
fn no_non_io_uring_variant_names_io_uring() {
    for spec in &all_variant_specs() {
        let rendered = spec.sample.to_string();
        if !spec.io_uring {
            assert!(
                !rendered.contains("io_uring"),
                "variant {} is outside io_uring but names io_uring in message: {rendered}",
                spec.variant_name
            );
        }
    }
}

#[test]
fn four_named_faults_have_exact_message_and_selectable_variant_identity() {
    // 1. Buffer offset past the registered region.
    let buffer_bounds_err = PipelineError::RegionBounds {
        region: "GpuMappedBuffer mapped allocation",
        offset: 8,
        len: 16,
        region_len: 16,
        unit: "bytes",
        fix: "reduce the slot size or enlarge the staging buffer",
    };
    let PipelineError::RegionBounds {
        region,
        offset,
        len,
        region_len,
        unit,
        fix,
    } = buffer_bounds_err
    else {
        panic!("expected RegionBounds variant, got {buffer_bounds_err:?}");
    };
    assert_eq!(region, "GpuMappedBuffer mapped allocation");
    assert_eq!(offset, 8);
    assert_eq!(len, 16);
    assert_eq!(region_len, 16);
    assert_eq!(unit, "bytes");
    assert_eq!(fix, "reduce the slot size or enlarge the staging buffer");
    assert_eq!(
        buffer_bounds_err.to_string(),
        "GpuMappedBuffer mapped allocation bytes range [8, 24) is outside the region's 16 bytes. Fix: reduce the slot size or enlarge the staging buffer"
    );

    // 2. usize::try_from overflow.
    let integer_width_err = PipelineError::IntegerWidth {
        quantity: "io_uring ingest slot count",
        value: 4_294_967_296,
        bits: 32,
        fix: "reduce the ingest slot count so it fits the host index width",
    };
    let PipelineError::IntegerWidth {
        quantity,
        value,
        bits,
        fix,
    } = integer_width_err
    else {
        panic!("expected IntegerWidth variant, got {integer_width_err:?}");
    };
    assert_eq!(quantity, "io_uring ingest slot count");
    assert_eq!(value, 4_294_967_296);
    assert_eq!(bits, 32);
    assert_eq!(
        fix,
        "reduce the ingest slot count so it fits the host index width"
    );
    assert_eq!(
        integer_width_err.to_string(),
        "io_uring ingest slot count 4294967296 does not fit 32 bits. Fix: reduce the ingest slot count so it fits the host index width"
    );

    // 3. Tenant retry counter wrapping u64.
    let tenant_overflow_err = PipelineError::CounterOverflow {
        scope: CounterScope::TenantRegistry,
        counter: "registration retry count",
        arithmetic: CounterArithmetic::Sum,
        lhs: u64::MAX,
        rhs: 1,
        bits: 64,
        fix: "retry registration later; the id allocator has not settled",
    };
    let PipelineError::CounterOverflow {
        scope,
        counter,
        arithmetic,
        lhs,
        rhs,
        bits,
        fix,
    } = tenant_overflow_err
    else {
        panic!("expected CounterOverflow variant, got {tenant_overflow_err:?}");
    };
    assert_eq!(scope, CounterScope::TenantRegistry);
    assert_eq!(counter, "registration retry count");
    assert_eq!(arithmetic, CounterArithmetic::Sum);
    assert_eq!(lhs, u64::MAX);
    assert_eq!(rhs, 1);
    assert_eq!(bits, 64);
    assert_eq!(
        fix,
        "retry registration later; the id allocator has not settled"
    );
    assert_eq!(
        tenant_overflow_err.to_string(),
        "tenant registry registration retry count overflowed: the sum of 18446744073709551615 and 1 leaves the 64-bit range. Fix: retry registration later; the id allocator has not settled"
    );

    // 4. drained_count exceeding published_count.
    let tenant_order_err = PipelineError::CounterOrder {
        scope: CounterScope::Tenant(7),
        produced_counter: "published_count",
        produced: 5,
        consumed_counter: "drained_count",
        consumed: 8,
        fix: "rebuild this tenant's slot accounting; note_drained ran for slots the tenant never published",
    };
    let PipelineError::CounterOrder {
        scope,
        produced_counter,
        produced,
        consumed_counter,
        consumed,
        fix,
    } = tenant_order_err
    else {
        panic!("expected CounterOrder variant, got {tenant_order_err:?}");
    };
    assert_eq!(scope, CounterScope::Tenant(7));
    assert_eq!(produced_counter, "published_count");
    assert_eq!(produced, 5);
    assert_eq!(consumed_counter, "drained_count");
    assert_eq!(consumed, 8);
    assert_eq!(
        fix,
        "rebuild this tenant's slot accounting; note_drained ran for slots the tenant never published"
    );
    assert_eq!(
        tenant_order_err.to_string(),
        "tenant 7 counters are out of order: drained_count 8 exceeds published_count 5 by 3. Fix: rebuild this tenant's slot accounting; note_drained ran for slots the tenant never published"
    );
}
