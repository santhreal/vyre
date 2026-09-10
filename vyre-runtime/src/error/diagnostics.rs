//! Structured diagnostics implementation for `PipelineError`.

use super::PipelineError;
use vyre_foundation::diagnostics::{
    CauseKind, CompilerLevel, Diagnostic, DiagnosticStage, RetryClass,
};

impl PipelineError {
    /// Project this error into the versioned structured diagnostic contract.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            Self::IoUringSyscall { syscall, errno, fix } => Diagnostic::error(
                "PIPE001_IO_URING_SYSCALL",
                format!("io_uring {syscall} failed: errno={errno}"),
            )
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix(*fix)
            .with_cause(
                CauseKind::ExternalToolchain,
                "syscall_failed",
                format!("{syscall} errno={errno}"),
            )
            .with_retry(RetryClass::SameDevice)
            .with_context_value("syscall", *syscall)
            .with_context_value("errno", errno.to_string()),
            Self::QueueFull { queue, depth, fix } => Diagnostic::error(
                "PIPE002_QUEUE_FULL",
                format!("io_uring {queue} queue at capacity ({depth} entries)"),
            )
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix(*fix)
            .with_cause(
                CauseKind::ResourceExhausted,
                "queue_capacity",
                format!("{queue} depth={depth}"),
            )
            .with_retry(RetryClass::SameDevice)
            .with_context_value("queue", *queue)
            .with_context_value("depth", depth.to_string()),
            Self::RegionBounds { region, offset, len, region_len, unit, fix } => Diagnostic::error(
                "PIPE003_REGION_BOUNDS",
                format!(
                    "{region} {unit} range [{offset}, {}) is outside the region's {region_len} {unit}",
                    offset.saturating_add(*len)
                ),
            )
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix(*fix)
            .with_cause(
                CauseKind::InvalidInput,
                "out_of_bounds",
                format!("{region} range extends past {region_len} {unit}"),
            )
            .with_retry(RetryClass::RecompileSource)
            .with_context_value("region", *region)
            .with_context_value("offset", offset.to_string())
            .with_context_value("len", len.to_string())
            .with_context_value("region_len", region_len.to_string()),
            Self::IntegerWidth { quantity, value, bits, fix } => Diagnostic::error(
                "PIPE004_INTEGER_WIDTH",
                format!("{quantity} {value} does not fit {bits} bits"),
            )
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix(*fix)
            .with_cause(
                CauseKind::NumericOverflow,
                "integer_overflow",
                format!("{quantity} {value} > {bits} bits"),
            )
            .with_retry(RetryClass::RecompileSource)
            .with_context_value("quantity", *quantity)
            .with_context_value("value", value.to_string())
            .with_context_value("bits", bits.to_string()),
            Self::CounterOverflow { scope, counter, arithmetic, lhs, rhs, bits, fix } => {
                Diagnostic::error(
                    "PIPE005_COUNTER_OVERFLOW",
                    format!("{scope:?} {counter} overflowed in {arithmetic:?}"),
                )
                .with_stage(DiagnosticStage::Submit)
                .with_compiler_level(CompilerLevel::DriverRuntime)
                .with_fix(*fix)
                .with_cause(
                    CauseKind::NumericOverflow,
                    "counter_overflow",
                    format!("{counter} {arithmetic:?} {lhs} and {rhs} exceeds {bits} bits"),
                )
                .with_retry(RetryClass::SameDevice)
                .with_context_value("counter", *counter)
                .with_context_value("lhs", lhs.to_string())
                .with_context_value("rhs", rhs.to_string())
            }
            Self::CounterOrder { scope, produced_counter, produced, consumed_counter, consumed, fix } => {
                Diagnostic::error(
                    "PIPE006_COUNTER_ORDER",
                    format!(
                        "{scope:?} counters out of order: {consumed_counter} {consumed} > {produced_counter} {produced}"
                    ),
                )
                .with_stage(DiagnosticStage::Submit)
                .with_compiler_level(CompilerLevel::DriverRuntime)
                .with_fix(*fix)
                .with_cause(
                    CauseKind::InternalInvariant,
                    "counter_order",
                    format!("{consumed_counter} {consumed} exceeds {produced_counter} {produced}"),
                )
                .with_retry(RetryClass::SameDevice)
                .with_context_value("produced", produced.to_string())
                .with_context_value("consumed", consumed.to_string())
            }
            Self::InvalidRequest { fault, quantity, observed, bound, fix } => Diagnostic::error(
                "PIPE007_INVALID_REQUEST",
                format!("io_uring request rejected: {quantity} is {observed}, which {fault} {bound}"),
            )
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix(*fix)
            .with_cause(
                CauseKind::InvalidInput,
                "invalid_request",
                format!("{quantity} {observed} violates {bound}"),
            )
            .with_retry(RetryClass::RecompileSource)
            .with_context_value("quantity", *quantity)
            .with_context_value("observed", observed.to_string())
            .with_context_value("bound", bound.to_string()),
            Self::SlotInFlight { slot, slot_count, inflight_tag, fix } => Diagnostic::error(
                "PIPE008_SLOT_IN_FLIGHT",
                format!(
                    "io_uring ingest slot {slot} of {slot_count} already in flight (tag {inflight_tag})"
                ),
            )
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix(*fix)
            .with_cause(
                CauseKind::ResourceExhausted,
                "slot_in_flight",
                format!("slot {slot} has tag {inflight_tag}"),
            )
            .with_retry(RetryClass::SameDevice)
            .with_context_value("slot", slot.to_string())
            .with_context_value("inflight_tag", inflight_tag.to_string()),
            Self::RingEncoding { fault, fix } => Diagnostic::error(
                "PIPE009_RING_ENCODING",
                format!("resident ring encode rejected: {fault:?}"),
            )
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix(*fix)
            .with_cause(CauseKind::Encoding, "ring_encoding", format!("{fault:?}"))
            .with_retry(RetryClass::RecompileSource),
            Self::Protocol(err) => Diagnostic::error(
                "PIPE010_PROTOCOL_ERROR",
                format!("host protocol error: {err}"),
            )
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix("check host-device ring protocol headers and payload framing")
            .with_cause(CauseKind::Encoding, "protocol_error", err.to_string())
            .with_retry(RetryClass::RecompileSource),
            Self::NotLinux => Diagnostic::error("PIPE011_NOT_LINUX", "io_uring is Linux-only")
                .with_stage(DiagnosticStage::Submit)
                .with_compiler_level(CompilerLevel::DriverRuntime)
                .with_fix(
                    "run on Linux 5.1+ and attach an AsyncUringStream to UringCompletionPump",
                )
                .with_cause(
                    CauseKind::UnsupportedCapability,
                    "unsupported_os",
                    "io_uring requires Linux",
                )
                .with_retry(RetryClass::Never),
            Self::NvmePassthroughDisabled => Diagnostic::error(
                "PIPE012_NVME_PASSTHROUGH_DISABLED",
                "NVMe passthrough requires the `uring-cmd-nvme` feature + Linux kernel 6.0+",
            )
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix("add `features = [\"uring-cmd-nvme\"]` to your Cargo.toml")
            .with_cause(
                CauseKind::Configuration,
                "feature_disabled",
                "uring-cmd-nvme not enabled",
            )
            .with_retry(RetryClass::Never),
            Self::Backend(message) => Diagnostic::error(
                "PIPE013_BACKEND_ERROR",
                format!("runtime backend failure: {message}"),
            )
            .with_stage(DiagnosticStage::Submit)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix("inspect backend dispatch logs and device generation state")
            .with_cause(
                CauseKind::ExternalToolchain,
                "backend_failure",
                message.clone(),
            )
            .with_retry(RetryClass::SameDevice),
            Self::DrainIncomplete { descriptor, claimed, expected, unit } => Diagnostic::error(
                "PIPE014_DRAIN_INCOMPLETE",
                format!(
                    "{descriptor} drain incomplete: only {claimed} of {expected} {unit} were claimed"
                ),
            )
            .with_stage(DiagnosticStage::Complete)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix(
                "raise the dispatch timeout (BatchDispatchConfig.timeout) or shard the batch into smaller queues",
            )
            .with_cause(
                CauseKind::Timeout,
                "drain_incomplete",
                format!("claimed {claimed} of {expected} {unit}"),
            )
            .with_retry(RetryClass::SameDevice)
            .with_context_value("claimed", claimed.to_string())
            .with_context_value("expected", expected.to_string()),
            Self::IllegalSlotTransition { transition, permitted: _, current_status, fix } => {
                Diagnostic::error(
                    "PIPE015_ILLEGAL_SLOT_TRANSITION",
                    format!(
                        "illegal ring slot transition `{transition}` from status {current_status}"
                    ),
                )
                .with_stage(DiagnosticStage::Submit)
                .with_compiler_level(CompilerLevel::DriverRuntime)
                .with_fix(*fix)
                .with_cause(
                    CauseKind::InternalInvariant,
                    "illegal_transition",
                    format!("transition `{transition}` invalid from {current_status}"),
                )
                .with_retry(RetryClass::SameDevice)
                .with_context_value("transition", *transition)
                .with_context_value("current_status", current_status.to_string())
            }
            Self::UnregisteredResource { request, resource, handle, slot, fix } => {
                Diagnostic::error(
                    "PIPE016_UNREGISTERED_RESOURCE",
                    format!("{request} in slot {slot} names unregistered {resource} handle {handle}"),
                )
                .with_stage(DiagnosticStage::Submit)
                .with_compiler_level(CompilerLevel::DriverRuntime)
                .with_fix(*fix)
                .with_cause(
                    CauseKind::Configuration,
                    "unregistered_resource",
                    format!("{resource} handle {handle} not registered"),
                )
                .with_retry(RetryClass::SameDevice)
                .with_context_value("resource", *resource)
                .with_context_value("handle", handle.to_string())
                .with_context_value("slot", slot.to_string())
            }
            Self::WorkerThreadPanicked { worker, fix } => Diagnostic::error(
                "PIPE017_WORKER_THREAD_PANICKED",
                format!("the {worker} thread panicked before it could be joined"),
            )
            .with_stage(DiagnosticStage::Complete)
            .with_compiler_level(CompilerLevel::DriverRuntime)
            .with_fix(*fix)
            .with_cause(
                CauseKind::InternalInvariant,
                "worker_panic",
                format!("worker {worker} panicked"),
            )
            .with_retry(RetryClass::SameDevice)
            .with_context_value("worker", *worker),
            Self::ReservedOpcode { tenant_id, local_opcode, global_opcode, fix } => {
                Diagnostic::error(
                    "PIPE018_RESERVED_OPCODE",
                    format!(
                        "tenant {tenant_id} local opcode {local_opcode} maps to global opcode {global_opcode} in reserved system range"
                    ),
                )
                .with_stage(DiagnosticStage::Plan)
                .with_compiler_level(CompilerLevel::DriverRuntime)
                .with_fix(*fix)
                .with_cause(
                    CauseKind::InvalidInput,
                    "reserved_opcode",
                    format!("opcode {global_opcode} in system range"),
                )
                .with_retry(RetryClass::Never)
                .with_context_value("tenant_id", tenant_id.to_string())
                .with_context_value("local_opcode", local_opcode.to_string())
                .with_context_value("global_opcode", global_opcode.to_string())
            }
        }
    }
}
