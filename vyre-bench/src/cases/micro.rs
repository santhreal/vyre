//! One owner for the foundation micro-benchmark cases.
//!
//! A micro case is a single IR program over a host fixture rebuilt per sample,
//! checked against a CPU reference. Seven cases each carried their own copy of
//! the identity, the metadata, the dispatch, the reference timing and the run
//! assembly; only the program, the fixture, the reference and the reported work
//! unit ever differed. Those four are the row, and everything else lives here.

use crate::api::case::{
    prepared_program, static_program_bytes_touched, BenchCase, BenchContext, BenchError, BenchId,
    BenchLayer, BenchMetadata, BenchRun, Correctness, DeterminismClass, PerformanceContract,
    PreparedCase, WorkloadClass,
};
use crate::api::metric::{elapsed_ns, BenchMetrics, MetricPoint};
use crate::cases::harness::ContractDescription;
use vyre_foundation::ir::Program;

/// What a micro case reports alongside wall time.
///
/// The two arms are not interchangeable: a compute case reports the work it
/// performed and leaves byte accounting to the program's static buffer sizes,
/// while a case whose output is a single counter must state its traffic
/// explicitly or the roofline reads four bytes of write for a full-buffer scan.
#[derive(Clone, Copy)]
pub(crate) enum MicroWork {
    /// A `flop_count` metric point on both the measured and the reference sample.
    Flops(u64),
    /// Explicit read/write accounting on the measured sample, and no work count.
    Bytes { read: u64, written: u64 },
}

/// One micro benchmark: a program, a fixture, a reference, and a work unit.
pub(crate) struct MicroCase {
    pub(crate) id: &'static str,
    pub(crate) name: &'static str,
    pub(crate) summary: &'static str,
    pub(crate) tags: &'static [&'static str],
    /// The CPU-baseline speedup floor, when the case is held to one.
    pub(crate) contract: Option<ContractDescription>,
    /// The IR program, built once during preparation.
    pub(crate) program: fn() -> Program,
    /// Host input buffers, rebuilt per measured sample.
    pub(crate) fixture: fn() -> Vec<Vec<u8>>,
    /// The CPU reference the dispatched outputs are compared against.
    pub(crate) reference: fn(&[Vec<u8>]) -> Vec<Vec<u8>>,
    pub(crate) work: MicroWork,
}

impl MicroCase {
    pub(crate) const fn new(
        id: &'static str,
        name: &'static str,
        summary: &'static str,
        tags: &'static [&'static str],
        program: fn() -> Program,
        fixture: fn() -> Vec<Vec<u8>>,
        reference: fn(&[Vec<u8>]) -> Vec<Vec<u8>>,
        work: MicroWork,
    ) -> Self {
        Self {
            id,
            name,
            summary,
            tags,
            contract: None,
            program,
            fixture,
            reference,
            work,
        }
    }
}

impl MicroCase {
    /// Metrics for the measured GPU sample.
    fn measured_metrics(
        &self,
        wall_ns: u64,
        dispatch_ns: Option<u64>,
        input_bytes: u64,
        output_bytes: u64,
    ) -> BenchMetrics {
        let mut metrics = BenchMetrics {
            wall_ns: Some(wall_ns),
            dispatch_ns,
            input_bytes: Some(input_bytes),
            output_bytes: Some(output_bytes),
            ..Default::default()
        };
        match self.work {
            MicroWork::Flops(count) => metrics.custom = vec![flop_count(count)],
            MicroWork::Bytes { read, written } => {
                metrics.bytes_read = Some(read);
                metrics.bytes_written = Some(written);
            }
        }
        metrics
    }

    /// Metrics for the CPU reference sample the measured one is reported against.
    ///
    /// The reference does not dispatch, so it carries no device time, and its
    /// byte traffic is the host work the comparison already accounts for.
    fn reference_metrics(&self, wall_ns: u64, input_bytes: u64, output_bytes: u64) -> BenchMetrics {
        BenchMetrics {
            wall_ns: Some(wall_ns),
            input_bytes: Some(input_bytes),
            output_bytes: Some(output_bytes),
            custom: match self.work {
                MicroWork::Flops(count) => vec![flop_count(count)],
                MicroWork::Bytes { .. } => vec![],
            },
            ..Default::default()
        }
    }

    /// The program fingerprint, as the benchmark report prints it.
    #[cfg(test)]
    fn program_fingerprint_hex(&self) -> String {
        (self.program)()
            .fingerprint()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    /// blake3 over every fixture buffer, length-prefixed so a byte moved
    /// between two buffers cannot hash the same.
    #[cfg(test)]
    fn fixture_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        for buffer in (self.fixture)() {
            hasher.update(&(buffer.len() as u64).to_le_bytes());
            hasher.update(&buffer);
        }
        hasher.finalize().to_hex().to_string()
    }
}

fn flop_count(value: u64) -> MetricPoint {
    MetricPoint {
        name: "flop_count".to_string(),
        value,
    }
}

/// This owner's name, as reported by every case it builds.
pub(crate) const MICRO_OWNER: &str = "cases::micro::MicroCase";

impl BenchCase for MicroCase {
    fn id(&self) -> BenchId {
        BenchId(self.id.to_string())
    }

    fn declaration_owner(&self) -> &'static str {
        MICRO_OWNER
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: self.name.to_string(),
            description: self.summary.to_string(),
            tags: self.tags.iter().map(|tag| (*tag).to_string()).collect(),
            layer: BenchLayer::Foundation,
            workload: WorkloadClass::Micro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-bench".to_string(),
        }
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        self.contract.map(|contract| {
            PerformanceContract::cpu_sota_min_speedup(
                contract.primitive,
                contract.baseline_crate,
                contract.baseline_name,
                contract.min_speedup_x,
            )
        })
    }

    fn prepare(&self, _ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new((self.program)()))
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let program = prepared_program(prepared)?;
        let inputs = (self.fixture)();
        let input_bytes = inputs.iter().map(Vec::len).sum::<usize>() as u64;

        let timed = ctx
            .dispatch_timed(program, &inputs, &ctx.dispatch_config)
            .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
        let outputs = timed.outputs;
        let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;

        let started = std::time::Instant::now();
        let reference_outputs = (self.reference)(&inputs);
        let reference_wall_ns = elapsed_ns(started);
        let reference_output_bytes = reference_outputs.iter().map(Vec::len).sum::<usize>() as u64;

        Ok(BenchRun {
            metrics: self.measured_metrics(
                timed.wall_ns,
                timed.device_ns,
                input_bytes,
                output_bytes,
            ),
            baseline_metrics: Some(self.reference_metrics(
                reference_wall_ns,
                input_bytes,
                reference_output_bytes,
            )),
            outputs,
            baseline_outputs: Some(reference_outputs),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        match self.work {
            MicroWork::Flops(_) => prepared_program(prepared)
                .map(static_program_bytes_touched)
                .unwrap_or((0, 0)),
            MicroWork::Bytes { read, written } => (read, written),
        }
    }
}

/// Every micro case, and the only place they are registered.
///
/// A new micro case is added here alongside its `inventory::submit!`, so the
/// table and the registration cannot drift apart, and the workload-digest pin
/// below turns red until the new row records its identity.
pub(crate) const MICRO_CASES: &[&MicroCase] = &[
    &crate::cases::attention::ATTENTION,
    &crate::cases::dfa_match::DFA_MATCH,
    &crate::cases::gather::GATHER,
    &crate::cases::histogram::HISTOGRAM,
    &crate::cases::matmul::MATMUL,
    &crate::cases::stencil::STENCIL3,
    &crate::cases::transpose::TRANSPOSE,
];

inventory::submit! {
    &crate::cases::attention::ATTENTION as &'static dyn BenchCase
}
inventory::submit! {
    &crate::cases::dfa_match::DFA_MATCH as &'static dyn BenchCase
}
inventory::submit! {
    &crate::cases::gather::GATHER as &'static dyn BenchCase
}
inventory::submit! {
    &crate::cases::histogram::HISTOGRAM as &'static dyn BenchCase
}
inventory::submit! {
    &crate::cases::matmul::MATMUL as &'static dyn BenchCase
}
inventory::submit! {
    &crate::cases::stencil::STENCIL3 as &'static dyn BenchCase
}
inventory::submit! {
    &crate::cases::transpose::TRANSPOSE as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::{MicroCase, MicroWork, MICRO_CASES};
    use crate::api::case::BenchCase;

    /// Workload identity of every micro case.
    ///
    /// `program` is the blake3 fingerprint of the canonical wire program.
    /// A deliberate IR-schema or program migration updates this pin only after
    /// the case's semantic contract is proved against the same fixture.
    ///
    /// `fixture` is blake3 over every fixture buffer, length-prefixed. No
    /// pre-collapse run exposed the fixture bytes, so this column is a forward
    /// pin: it cannot testify about the collapse, and it does hold the fixture
    /// still from here on.
    const PINNED_WORKLOADS: &[PinnedWorkload] = &[
        PinnedWorkload {
            id: "foundation.attention.64",
            program: "4bb410db53b85420765d63c15249762ea9cf79ba153820423e43ff5c88e71579",
            fixture: "ede5e815a089bbdd231d17a57bbe1cdf59c097be49479838f9a0a64f8f81f183",
        },
        PinnedWorkload {
            id: "foundation.dfa_match.256k",
            program: "859a41f1108f8b89e7bddd0bf9d5367555c2fe50a7f5610858e19be509ffd576",
            fixture: "0a747a3ac1a8d7831a36f7120a33ece13c9d34dfecda5493bc98ef825c05a435",
        },
        PinnedWorkload {
            id: "foundation.gather.u32.1m",
            program: "fb4d840bf5761203638d82ed1b163cb919fcbf48a34f1dbe078fbfdcd6ca4581",
            fixture: "556f2bccabd62d6d434e97f155a78b1a5dbca9a4e1a8ab9993dc7a19d2aa1217",
        },
        PinnedWorkload {
            id: "foundation.histogram.u32_256.1m",
            program: "9ea0006edd469c9f189a8e1f45f236ded5c998cf7fc62d118ce0f78f8ddf03f6",
            fixture: "b09c20e4f186708fcb827e0949aac9c60a49340870ee3eb25f8404d56cec641b",
        },
        PinnedWorkload {
            id: "foundation.matmul.256",
            program: "b47558ecc1de3f74f9493e4401df07f2923198c7838e69d33fd5db2879c95913",
            fixture: "9d8d7b3b1340fb8ebe3f170166045fa644674ffea70713121e35c58af9453831",
        },
        PinnedWorkload {
            id: "foundation.stencil3.u32.1m",
            program: "6253da7932aee0ef8cbf5c103f635ee098eab27fae32821cf0de8b2da0e47b21",
            fixture: "183866c61def7900b8ad927a5fb4d5b9847ac4274951dacb9334afb64fddb29e",
        },
        PinnedWorkload {
            id: "foundation.transpose.512",
            program: "4891fb16026428d757df8521c865a48a68c3053d4327121674f6a4ec6f7641a2",
            fixture: "a9d0dfc6e815cbbae4077e4d2d200c67452c8db5d0e3f950bfc786dc1c459e0c",
        },
    ];

    struct PinnedWorkload {
        id: &'static str,
        program: &'static str,
        fixture: &'static str,
    }

    /// Every micro case keeps its deliberately recorded program and fixture
    /// identity until a proved migration updates the corresponding pin.
    ///
    /// Derived from `MICRO_CASES` rather than a count, so a case added without a
    /// recorded workload identity fails here instead of shipping unpinned.
    #[test]
    fn every_micro_case_keeps_its_pinned_workload() {
        for case in MICRO_CASES {
            let pinned = PINNED_WORKLOADS
                .iter()
                .find(|pinned| pinned.id == case.id)
                .unwrap_or_else(|| {
                    panic!(
                        "Fix: micro case `{}` has no pinned workload. Record its program fingerprint and fixture digest in PINNED_WORKLOADS with the decision that justifies them.",
                        case.id
                    )
                });

            assert_eq!(
                case.program_fingerprint_hex(),
                pinned.program,
                "Fix: micro case `{}` no longer builds the program its recorded evidence was measured against.",
                case.id
            );
            assert_eq!(
                case.fixture_digest_hex(),
                pinned.fixture,
                "Fix: micro case `{}` changed its fixture bytes.",
                case.id
            );
        }
    }

    /// Micro-case ids are unique, so a copied row cannot register twice under a
    /// stable table length.
    #[test]
    fn micro_case_ids_are_unique() {
        let mut ids: Vec<&str> = MICRO_CASES.iter().map(|case| case.id).collect();
        let registered = ids.len();
        ids.sort_unstable();
        ids.dedup();

        assert_eq!(ids.len(), registered, "Fix: two micro cases share an id.");
    }

    /// Both work arms report what their case measures, and nothing the other
    /// arm reports, across the whole lane range the io accounting is derived
    /// from.
    ///
    /// The match is exhaustive without a wildcard: a third `MicroWork` arm
    /// stops this test compiling until it states what that arm reports.
    #[test]
    fn each_work_arm_reports_only_its_own_accounting() {
        let arms: [fn(u64, u64) -> MicroWork; 2] = [
            |work_units, _| MicroWork::Flops(work_units),
            |_, lanes| MicroWork::Bytes {
                read: lanes.saturating_mul(4),
                written: 4,
            },
        ];
        let mut checked = 0_u32;
        for arm in arms {
            for lanes in 0_u64..=2_048 {
                let input_bytes = lanes.saturating_mul(4);
                let output_bytes = lanes.saturating_mul(8);
                let work = arm(lanes.saturating_mul(3), lanes);
                let case = MicroCase {
                    id: "probe",
                    name: "probe",
                    summary: "probe",
                    tags: &[],
                    contract: None,
                    program: || unreachable!("metric assembly never builds the program"),
                    fixture: Vec::new,
                    reference: |_| Vec::new(),
                    work,
                };
                let measured =
                    case.measured_metrics(11 + lanes, Some(7 + lanes), input_bytes, output_bytes);
                let reference = case.reference_metrics(13 + lanes, input_bytes, output_bytes);

                assert_eq!(measured.wall_ns, Some(11 + lanes));
                assert_eq!(measured.dispatch_ns, Some(7 + lanes));
                assert_eq!(measured.input_bytes, Some(input_bytes));
                assert_eq!(measured.output_bytes, Some(output_bytes));
                assert_eq!(reference.wall_ns, Some(13 + lanes));
                assert_eq!(
                    reference.dispatch_ns, None,
                    "the reference sample never dispatches"
                );
                assert_eq!(reference.input_bytes, Some(input_bytes));
                assert_eq!(reference.output_bytes, Some(output_bytes));

                match work {
                    MicroWork::Flops(count) => {
                        assert_eq!(measured.bytes_read, None);
                        assert_eq!(measured.bytes_written, None);
                        assert_eq!(measured.custom.len(), 1);
                        assert_eq!(measured.custom[0].name, "flop_count");
                        assert_eq!(measured.custom[0].value, count);
                        assert_eq!(reference.custom.len(), 1);
                        assert_eq!(reference.custom[0].name, "flop_count");
                        assert_eq!(reference.custom[0].value, count);
                    }
                    MicroWork::Bytes { read, written } => {
                        assert_eq!(measured.bytes_read, Some(read));
                        assert_eq!(measured.bytes_written, Some(written));
                        assert!(measured.custom.is_empty());
                        assert!(reference.custom.is_empty());
                    }
                }
                checked += 1;
            }
        }

        assert_eq!(checked, 2 * 2_049, "both arms cover the whole lane range");
    }

    /// A `Bytes` case states its own traffic; a `Flops` case inherits the
    /// program's static buffer sizes.
    #[test]
    fn bytes_work_overrides_static_program_accounting() {
        let case = MicroCase {
            id: "probe",
            name: "probe",
            summary: "probe",
            tags: &[],
            contract: None,
            program: || unreachable!("byte accounting never builds the program"),
            fixture: Vec::new,
            reference: |_| Vec::new(),
            work: MicroWork::Bytes {
                read: 262_144,
                written: 4,
            },
        };
        let prepared: crate::api::case::PreparedCase = Box::new(0_u8);

        assert_eq!(case.bytes_touched(&prepared), (262_144, 4));
    }
}
