//! WHY: roughly thirty io_uring and tenant-accounting faults were reported as
//! `PipelineError::QueueFull`, whose message states that an io_uring queue is at
//! capacity. A buffer offset past the registered region, a `usize::try_from`
//! overflow, a tenant retry counter wrapping u64, and a `drained_count` past its
//! `published_count` all rendered as a full submission queue, so the message
//! contradicted the fault and a caller could not select on it.
//!
//! Closes: the `PipelineError` variant space and the claims each variant's
//! message is allowed to make. `facts` matches every variant with no catch-all
//! arm, so a variant added to the enum fails to compile here; `Tag` and
//! `Tag::ALL` are one declaration, so a tag cannot exist that the list omits;
//! and the coverage test below fails until the new tag also has a fixture, so a
//! tag with no variant behind it fails too. A variant therefore cannot enter
//! the enum without a recorded decision about whether it may name io_uring,
//! whether it may state that a queue is at capacity, and whether it carries a
//! `Fix:` clause.
//!
//! The correspondence is proven by the match arms rather than by reading
//! `lib.rs` as text. A source scan of the enum declaration reported the same
//! variant set the compiler already enforces, and it went stale the moment the
//! declaration was reformatted.
//!
//! Does not catch: a call site that constructs the wrong variant. Which fault a
//! given call reports is asserted at that call's own test. It also does not
//! prove that a `fix` string is good advice, only that one is present.
//!
//! An exhaustive match over a `#[non_exhaustive]` enum is legal only inside the
//! crate that defines it, so this closure is a test-only module in `src` rather
//! than an integration test that could not compile it. `lib.rs` declares
//! `extern crate self as vyre_runtime` so the paths below read the way a
//! consumer reads them.

use vyre_runtime::resident_work_queue::protocol::ProtocolError;
use vyre_runtime::{
    CounterArithmetic, CounterScope, PipelineError, RequestFault, RingEncodingFault,
};

/// Declares [`Tag`] and its complete list in one place.
///
/// The list used to be a separate `const ALL: [Self; 18]`, and the comment on
/// it said that a fixed-length array makes a missing entry a compile error.
/// `[Self; 18]` stays valid however many variants the enum grows, so a
/// nineteenth tag would compile, never enter `ALL`, and every test below
/// iterates `ALL`. Writing both from one variant list is what makes the
/// omission unrepresentable rather than merely discouraged.
macro_rules! declare_tags {
    ($($variant:ident),+ $(,)?) => {
        /// One tag per [`PipelineError`] variant.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Tag {
            $($variant),+
        }

        impl Tag {
            /// Every tag, in declaration order.
            const ALL: &'static [Self] = &[$(Self::$variant),+];
        }
    };
}

declare_tags!(
    IoUringSyscall,
    QueueFull,
    RegionBounds,
    IntegerWidth,
    CounterOverflow,
    CounterOrder,
    InvalidRequest,
    SlotInFlight,
    RingEncoding,
    Protocol,
    NotLinux,
    NvmePassthroughDisabled,
    Backend,
    DrainIncomplete,
    IllegalSlotTransition,
    UnregisteredResource,
    WorkerThreadPanicked,
    ReservedOpcode,
);

/// What one variant is allowed to claim, and the substring that names its own
/// fault.
struct VariantFacts {
    /// Which variant this describes.
    tag: Tag,
    /// Substring that names this variant's fault and appears in no other
    /// variant's message.
    discriminator: &'static str,
    /// Whether this variant reports a fault in the io_uring subsystem, and may
    /// therefore name io_uring.
    io_uring: bool,
    /// Whether this variant's message may state that a queue is at capacity.
    claims_queue_capacity: bool,
    /// Whether this variant's message must carry a `Fix:` clause.
    states_a_fix: bool,
}

/// Whether a counter's owner is the io_uring subsystem.
///
/// Exhaustive with no catch-all, so a scope added to [`CounterScope`] fails to
/// compile until someone records whether its messages may name io_uring.
fn scope_is_io_uring(scope: CounterScope) -> bool {
    match scope {
        CounterScope::IoUring => true,
        CounterScope::TenantRegistry | CounterScope::Tenant(_) => false,
    }
}

/// The whole variant space, with no catch-all arm.
///
/// This is the compile-time closure: a variant added to [`PipelineError`] makes
/// this function fail to compile, naming this file.
fn facts(error: &PipelineError) -> VariantFacts {
    match error {
        PipelineError::IoUringSyscall { .. } => VariantFacts {
            tag: Tag::IoUringSyscall,
            discriminator: "failed: errno=",
            io_uring: true,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::QueueFull { .. } => VariantFacts {
            tag: Tag::QueueFull,
            discriminator: "queue at capacity",
            io_uring: true,
            claims_queue_capacity: true,
            states_a_fix: true,
        },
        PipelineError::RegionBounds { .. } => VariantFacts {
            tag: Tag::RegionBounds,
            discriminator: "is outside the region's",
            io_uring: true,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::IntegerWidth { .. } => VariantFacts {
            tag: Tag::IntegerWidth,
            discriminator: "does not fit",
            io_uring: true,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::CounterOverflow { scope, .. } => VariantFacts {
            tag: Tag::CounterOverflow,
            discriminator: "leaves the",
            io_uring: scope_is_io_uring(*scope),
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::CounterOrder { scope, .. } => VariantFacts {
            tag: Tag::CounterOrder,
            discriminator: "counters are out of order",
            io_uring: scope_is_io_uring(*scope),
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::InvalidRequest { .. } => VariantFacts {
            tag: Tag::InvalidRequest,
            discriminator: "request rejected:",
            io_uring: true,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::SlotInFlight { .. } => VariantFacts {
            tag: Tag::SlotInFlight,
            discriminator: "already has a read in flight",
            io_uring: true,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::RingEncoding { .. } => VariantFacts {
            tag: Tag::RingEncoding,
            discriminator: "resident ring encode rejected",
            io_uring: false,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::Protocol(_) => VariantFacts {
            tag: Tag::Protocol,
            discriminator: "byte length overflow",
            io_uring: false,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::NotLinux => VariantFacts {
            tag: Tag::NotLinux,
            discriminator: "is Linux-only",
            io_uring: true,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::NvmePassthroughDisabled => VariantFacts {
            tag: Tag::NvmePassthroughDisabled,
            discriminator: "NVMe passthrough requires",
            io_uring: true,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::Backend(_) => VariantFacts {
            tag: Tag::Backend,
            discriminator: "backend error:",
            io_uring: false,
            claims_queue_capacity: false,
            states_a_fix: false,
        },
        PipelineError::DrainIncomplete { .. } => VariantFacts {
            tag: Tag::DrainIncomplete,
            discriminator: "drain incomplete",
            io_uring: false,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::IllegalSlotTransition { .. } => VariantFacts {
            tag: Tag::IllegalSlotTransition,
            discriminator: "illegal ring slot transition",
            io_uring: false,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::UnregisteredResource { .. } => VariantFacts {
            tag: Tag::UnregisteredResource,
            discriminator: "which is not registered",
            io_uring: false,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::WorkerThreadPanicked { .. } => VariantFacts {
            tag: Tag::WorkerThreadPanicked,
            discriminator: "thread panicked before it could be joined",
            io_uring: false,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
        PipelineError::ReservedOpcode { .. } => VariantFacts {
            tag: Tag::ReservedOpcode,
            discriminator: "reserved for system opcodes",
            io_uring: false,
            claims_queue_capacity: false,
            states_a_fix: true,
        },
    }
}

/// One instance of every variant, in `Tag::ALL` order.
fn every_variant() -> Vec<PipelineError> {
    vec![
        PipelineError::IoUringSyscall {
            syscall: "io_uring_enter",
            errno: 11,
            fix: "retry the enter call",
        },
        PipelineError::QueueFull {
            queue: "submission",
            depth: 64,
            fix: "reap completions, then submit again",
        },
        PipelineError::RegionBounds {
            region: "GpuMappedBuffer mapped allocation",
            offset: 8,
            len: 16,
            region_len: 16,
            unit: "bytes",
            fix: "enlarge the buffer or reduce the read size",
        },
        PipelineError::IntegerWidth {
            quantity: "io_uring completion user_data",
            value: 1 << 40,
            bits: 32,
            fix: "keep user_data inside the u32 slot-id range",
        },
        PipelineError::CounterOverflow {
            scope: CounterScope::IoUring,
            counter: "inflight SQE count",
            arithmetic: CounterArithmetic::Sum,
            lhs: u64::from(u32::MAX),
            rhs: 1,
            bits: 32,
            fix: "poll completions before submitting more work",
        },
        PipelineError::CounterOrder {
            scope: CounterScope::Tenant(3),
            produced_counter: "published_count",
            produced: 5,
            consumed_counter: "drained_count",
            consumed: 7,
            fix: "rebuild this tenant's slot accounting",
        },
        PipelineError::InvalidRequest {
            fault: RequestFault::LengthMismatch,
            quantity: "pump read length",
            observed: 8192,
            bound: 4096,
            fix: "submit exactly chunk_bytes per read",
        },
        PipelineError::SlotInFlight {
            slot: 2,
            slot_count: 4,
            inflight_tag: 9,
            fix: "drain completions for this slot before reusing it",
        },
        PipelineError::RingEncoding {
            fault: RingEncodingFault::OutOfBounds,
            fix: "publish into a slot the ring contains",
        },
        PipelineError::Protocol(ProtocolError::ByteLengthOverflow {
            buffer: "ring",
            fix: "shard the ring",
        }),
        PipelineError::NotLinux,
        PipelineError::NvmePassthroughDisabled,
        PipelineError::Backend("device rejected the module".to_string()),
        PipelineError::DrainIncomplete {
            descriptor: "megakernel",
            claimed: 1,
            expected: 4,
            unit: "work-items",
        },
        PipelineError::IllegalSlotTransition {
            transition: "claim",
            permitted: &[1],
            current_status: 2,
            fix: "wait for the host to publish the slot",
        },
        PipelineError::UnregisteredResource {
            request: "megakernel IO READ",
            resource: "GPU destination",
            handle: 9,
            slot: 3,
            fix: "register the destination before publishing READ requests",
        },
        PipelineError::WorkerThreadPanicked {
            worker: "IO loop",
            fix: "repair the fault the panic names, then respawn the loop",
        },
        PipelineError::ReservedOpcode {
            tenant_id: 4,
            local_opcode: 2,
            global_opcode: 0x8000_0002,
            fix: "move the tenant opcode window below the reserved range",
        },
    ]
}

#[test]
fn every_tag_has_exactly_one_fixture_in_tag_order() {
    let variants = every_variant();
    assert_eq!(
        variants.len(),
        Tag::ALL.len(),
        "Fix: every PipelineError variant needs one instance in every_variant"
    );
    for (index, error) in variants.iter().enumerate() {
        assert_eq!(
            facts(error).tag,
            Tag::ALL[index],
            "Fix: every_variant must list one instance per tag, in Tag::ALL order: {error:?}"
        );
    }
    for &tag in Tag::ALL {
        assert!(
            variants.iter().any(|error| facts(error).tag == tag),
            "Fix: {tag:?} has no instance in every_variant, so nothing exercises its message"
        );
    }
}

#[test]
fn every_variant_names_its_own_fault_and_no_other() {
    let variants = every_variant();
    for error in &variants {
        let rendered = error.to_string();
        let own = facts(error).discriminator;
        assert!(
            rendered.contains(own),
            "Fix: {error:?} must name its own fault with {own:?}: {rendered}"
        );
        for other in &variants {
            let other_facts = facts(other);
            if other_facts.tag == facts(error).tag {
                continue;
            }
            assert!(
                !rendered.contains(other_facts.discriminator),
                "Fix: {error:?} states {:?}, which names the {:?} fault it did not observe: \
                 {rendered}",
                other_facts.discriminator,
                other_facts.tag
            );
        }
    }
}

#[test]
fn only_the_queue_variant_claims_a_queue_is_at_capacity() {
    for error in &every_variant() {
        let rendered = error.to_string();
        if facts(error).claims_queue_capacity {
            continue;
        }
        assert!(
            !rendered.contains("queue at capacity"),
            "Fix: {error:?} states that a queue is at capacity, which it did not observe: \
             {rendered}"
        );
    }
}

#[test]
fn no_variant_outside_io_uring_names_io_uring() {
    for error in &every_variant() {
        let rendered = error.to_string();
        assert!(
            facts(error).io_uring || !rendered.contains("io_uring"),
            "Fix: {error:?} names io_uring for a fault outside that subsystem: {rendered}"
        );
    }
}

#[test]
fn every_variant_states_a_fix_or_is_self_explanatory() {
    for error in &every_variant() {
        let rendered = error.to_string();
        assert_eq!(
            rendered.contains("Fix:"),
            facts(error).states_a_fix,
            "Fix: {error:?} disagrees with its recorded decision about carrying a Fix: clause: \
             {rendered}"
        );
    }
}

#[test]
fn every_variant_renders_a_non_empty_message() {
    for error in &every_variant() {
        assert!(
            !error.to_string().is_empty(),
            "{error:?} renders as an empty message"
        );
    }
}

/// Exhaustive with no catch-all: adding a scope fails to compile here, and the
/// length assertion below then fails until the list covers it.
fn scope_index(scope: CounterScope) -> usize {
    match scope {
        CounterScope::IoUring => 0,
        CounterScope::TenantRegistry => 1,
        CounterScope::Tenant(_) => 2,
    }
}

#[test]
fn every_counter_scope_names_its_owner_distinctly() {
    let scopes = [
        CounterScope::IoUring,
        CounterScope::TenantRegistry,
        CounterScope::Tenant(7),
    ];
    assert_eq!(
        scopes.len(),
        3,
        "Fix: a CounterScope was added without an instance here"
    );
    for (index, scope) in scopes.iter().enumerate() {
        assert_eq!(
            scope_index(*scope),
            index,
            "Fix: keep this list in scope_index order"
        );
    }
    let rendered: Vec<String> = scopes.iter().map(ToString::to_string).collect();
    for (index, name) in rendered.iter().enumerate() {
        assert!(
            !name.is_empty(),
            "Fix: {:?} renders no owner",
            scopes[index]
        );
        for (other_index, other) in rendered.iter().enumerate() {
            assert!(
                index == other_index || name != other,
                "Fix: {:?} and {:?} render the same owner {name}",
                scopes[index],
                scopes[other_index]
            );
        }
    }
    assert_eq!(CounterScope::Tenant(7).to_string(), "tenant 7");
}

/// Exhaustive with no catch-all, for the same reason.
fn arithmetic_index(arithmetic: CounterArithmetic) -> usize {
    match arithmetic {
        CounterArithmetic::Sum => 0,
        CounterArithmetic::Product => 1,
    }
}

#[test]
fn every_arithmetic_is_listed_in_all_and_reads_distinctly() {
    for (index, arithmetic) in CounterArithmetic::ALL.iter().enumerate() {
        assert_eq!(
            arithmetic_index(*arithmetic),
            index,
            "Fix: CounterArithmetic::ALL must stay in arithmetic_index order"
        );
    }
    assert_eq!(
        CounterArithmetic::ALL.len(),
        2,
        "Fix: a CounterArithmetic was added without listing it in ALL"
    );
    assert_ne!(
        CounterArithmetic::Sum.to_string(),
        CounterArithmetic::Product.to_string(),
        "Fix: a sum overflow and a product overflow must not read the same"
    );
}

/// Exhaustive with no catch-all, for the same reason.
fn request_fault_index(fault: RequestFault) -> usize {
    match fault {
        RequestFault::BelowMinimum => 0,
        RequestFault::LengthMismatch => 1,
        RequestFault::Indivisible => 2,
    }
}

#[test]
fn every_request_fault_is_listed_in_all_and_reads_distinctly() {
    for (index, fault) in RequestFault::ALL.iter().enumerate() {
        assert_eq!(
            request_fault_index(*fault),
            index,
            "Fix: RequestFault::ALL must stay in request_fault_index order"
        );
    }
    assert_eq!(
        RequestFault::ALL.len(),
        3,
        "Fix: a RequestFault was added without listing it in ALL"
    );
    let rendered: Vec<String> = RequestFault::ALL.iter().map(ToString::to_string).collect();
    for (index, name) in rendered.iter().enumerate() {
        for (other_index, other) in rendered.iter().enumerate() {
            assert!(
                index == other_index || name != other,
                "Fix: {:?} and {:?} describe the request fault the same way: {name}",
                RequestFault::ALL[index],
                RequestFault::ALL[other_index]
            );
        }
    }
}

/// Each `RequestFault` has to read as a relation between the observed value and
/// the bound, because the variant's message puts the two on either side of it.
#[test]
fn a_request_fault_message_relates_the_observed_value_to_its_bound() {
    for fault in RequestFault::ALL {
        let error = PipelineError::InvalidRequest {
            fault,
            quantity: "iovec storage slots",
            observed: 0,
            bound: 1,
            fix: "pass at least one iovec slot",
        };
        let rendered = error.to_string();
        assert!(
            rendered.contains("iovec storage slots is 0, which"),
            "Fix: the request message must state the quantity and the value it observed: {rendered}"
        );
        assert!(
            rendered.ends_with(" 1. Fix: pass at least one iovec slot"),
            "Fix: the request message must end the relation with the bound it required: {rendered}"
        );
    }
}
