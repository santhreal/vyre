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

    /// Domain separator for [`MicroCase::program_structure_hex`].
    #[cfg(test)]
    const PROGRAM_STRUCTURE_DOMAIN: &[u8] = b"vyre-bench/micro-case-structural-ir/v1";

    /// Structural identity of the program this case builds.
    ///
    /// BLAKE3 over the canonicalized buffer roster and entry node tree, which
    /// is a function of the IR model alone. `Program::fingerprint` is not used
    /// here: it is BLAKE3 over `canonical_wire_bytes`, whose first framed field
    /// is `WIRE_FORMAT_VERSION`, so a serialization revision moves it for every
    /// program in the workspace while no program's meaning moves. A pin that
    /// cannot fail on the thing its name claims is answered by copying the new
    /// numbers back in, which is how a real workload change rides in unnoticed
    /// behind a version bump.
    #[cfg(test)]
    fn program_structure_hex(&self) -> String {
        let canonical = (self.program)().canonicalized();
        let rendered = format!(
            "buffers:\n{:#?}\n\nentry:\n{:#?}\n",
            canonical.buffers(),
            canonical.entry()
        );
        vyre_foundation::hashing::domain_digest(Self::PROGRAM_STRUCTURE_DOMAIN, rendered.as_bytes())
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
        self.contract.map(ContractDescription::performance_contract)
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
#[cfg(test)]
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
    /// `structure` is blake3 over the canonicalized buffer roster and entry
    /// node tree of the program the case builds. It carries node kinds, operand
    /// expressions, literal values and nesting, and it is a function of the IR
    /// model alone, so a changed operand, a dropped node or a changed ABI moves
    /// it and a serialization revision does not. The column it replaces was
    /// `Program::fingerprint`, blake3 over `canonical_wire_bytes`, whose first
    /// framed field is `WIRE_FORMAT_VERSION`: the 8 to 9 bump in `818171cc5c`
    /// moved all seven at once with no semantic change, which is a pin that
    /// cannot fail on the thing its name claims and whose only answer is to
    /// copy the new numbers in.
    ///
    /// A migration updates this pin only after the case's semantic contract is
    /// proved against the same fixture, and
    /// `every_micro_case_still_computes_its_cpu_reference` is that proof.
    ///
    /// `fixture` is blake3 over every fixture buffer, length-prefixed. No
    /// pre-collapse run exposed the fixture bytes, so this column is a forward
    /// pin: it cannot testify about the collapse, and it does hold the fixture
    /// still from here on.
    const PINNED_WORKLOADS: &[PinnedWorkload] = &[
        PinnedWorkload {
            id: "foundation.attention.64",
            structure: "111338caca3b2bccc0fd43df36adafaf73cd7926672eb7f274d620ca5246e9df",
            fixture: "ede5e815a089bbdd231d17a57bbe1cdf59c097be49479838f9a0a64f8f81f183",
        },
        PinnedWorkload {
            id: "foundation.dfa_match.256k",
            structure: "fef5e85cb73f4258b28c6565ad2a2542457a1246c67a875de5ea9cec442da18f",
            fixture: "0a747a3ac1a8d7831a36f7120a33ece13c9d34dfecda5493bc98ef825c05a435",
        },
        PinnedWorkload {
            id: "foundation.gather.u32.1m",
            structure: "6b8d71aaff0f46e2c73e2c6a0c1ce3618666ef333e4cbce08a9547102782e2d9",
            fixture: "556f2bccabd62d6d434e97f155a78b1a5dbca9a4e1a8ab9993dc7a19d2aa1217",
        },
        PinnedWorkload {
            id: "foundation.histogram.u32_256.1m",
            structure: "0b88138dabe0c4bbf91467d76f67645ee9e311cf47beeab881d014484d01b1f7",
            fixture: "b09c20e4f186708fcb827e0949aac9c60a49340870ee3eb25f8404d56cec641b",
        },
        PinnedWorkload {
            id: "foundation.matmul.256",
            structure: "45ba27bb3a2921b299d8131acda8d30f6bc8d3fae5d12b2c3f314a0bc7875fd5",
            fixture: "9d8d7b3b1340fb8ebe3f170166045fa644674ffea70713121e35c58af9453831",
        },
        PinnedWorkload {
            id: "foundation.stencil3.u32.1m",
            structure: "b74a7e15a496b4604387543a0f0e4e7b57c68f1153eb1a83dafcb685b0e5a320",
            fixture: "183866c61def7900b8ad927a5fb4d5b9847ac4274951dacb9334afb64fddb29e",
        },
        PinnedWorkload {
            id: "foundation.transpose.512",
            structure: "eddfd4390b1df33ff2278c3092b3c6102d917f60c026c3ba4b5f8e03eb555149",
            fixture: "a9d0dfc6e815cbbae4077e4d2d200c67452c8db5d0e3f950bfc786dc1c459e0c",
        },
    ];

    struct PinnedWorkload {
        id: &'static str,
        structure: &'static str,
        fixture: &'static str,
    }

    /// Every micro case keeps its deliberately recorded program and fixture
    /// identity until a proved migration updates the corresponding pin.
    ///
    /// Derived from `MICRO_CASES` rather than a count, so a case added without a
    /// recorded workload identity fails here instead of shipping unpinned.
    #[test]
    fn every_micro_case_keeps_its_pinned_workload() {
        let mut drifted = Vec::new();
        for case in MICRO_CASES {
            let pinned = PINNED_WORKLOADS
                .iter()
                .find(|pinned| pinned.id == case.id)
                .unwrap_or_else(|| {
                    panic!(
                        "Fix: micro case `{}` has no pinned workload. Record its structural digest and fixture digest in PINNED_WORKLOADS with the decision that justifies them.",
                        case.id
                    )
                });
            let structure = case.program_structure_hex();
            if structure != pinned.structure {
                drifted.push(format!(
                    "{}: structure pinned {} but builds {structure}",
                    case.id, pinned.structure
                ));
            }
            let fixture = case.fixture_digest_hex();
            if fixture != pinned.fixture {
                drifted.push(format!(
                    "{}: fixture pinned {} but builds {fixture}",
                    case.id, pinned.fixture
                ));
            }
        }
        assert!(
            drifted.is_empty(),
            "Fix: {} micro case(s) no longer build the workload their recorded evidence was measured against: {}",
            drifted.len(),
            drifted.join("; ")
        );
    }

    /// A pinned program fingerprint moves only when the case still computes its
    /// own CPU reference, so the pin cannot be refreshed to whatever the tree
    /// happens to encode.
    ///
    /// Reference evaluation is the parity oracle, so this proves the semantic
    /// contract of every registered case without a device, derived from
    /// `MICRO_CASES` rather than a list, so a new case is proved too. It does
    /// not prove device parity: a backend that mislowers the same program is
    /// caught by the dispatch comparison in `verify`, not here.
    #[test]
    fn every_micro_case_still_computes_its_cpu_reference() {
        for case in MICRO_CASES {
            let inputs = (case.fixture)();
            let values = inputs
                .iter()
                .cloned()
                .map(vyre_reference::value::Value::from)
                .collect::<Vec<_>>();
            let outputs = vyre_reference::reference_eval(&(case.program)(), &values)
                .unwrap_or_else(|error| {
                    panic!("Fix: `{}` must reference-evaluate: {error}", case.id)
                })
                .into_iter()
                .map(|value| value.to_bytes())
                .collect::<Vec<_>>();
            assert_eq!(
                outputs,
                (case.reference)(&inputs),
                "Fix: `{}` must compute its CPU reference byte for byte",
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
