//! WHY: roughly thirty io_uring and tenant-accounting faults were reported as
//! `PipelineError::QueueFull`, whose message reads `io_uring {queue} queue at
//! capacity`. A buffer offset past the registered region, a `usize::try_from`
//! overflow, a tenant retry counter wrapping u64, and a `drained_count`
//! exceeding `published_count` all rendered as a full submission queue, so the
//! message contradicted the fault and a caller could not select on it.
//!
//! Closes: the selectable identity, the field values, and the exact rendered
//! text of those four faults. A caller selects on the variant and reads the
//! numbers out of it, so both are pinned here rather than a substring.
//!
//! The variant space itself is closed in
//! `vyre-runtime/src/pipeline_error_closure.rs`, which matches every variant
//! with no catch-all arm. An exhaustive match over a `#[non_exhaustive]` enum
//! is legal only inside the declaring crate, so this file used to enumerate the
//! enum by reading `lib.rs` as text and carried its own copy of every variant's
//! fixture. Two copies of one fixture list drift in silence, and the copy that
//! reads source drifts first.
//!
//! Does not catch: call sites that construct an incorrect variant for an
//! unexercised path. That is covered at each call site's own subsystem tests.

use vyre_runtime::{CounterArithmetic, CounterScope, PipelineError};

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
