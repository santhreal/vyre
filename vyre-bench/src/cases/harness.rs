//! One case-builder harness for benchmark cases.
//!
//! A benchmark case is a static [`WorkloadDescription`] plus the handful of
//! operations a description cannot carry. Everything else -- the `BenchId`, the
//! `BenchMetadata`, the suite list, the requirements record, the performance
//! contract, the prepared-payload downcast, the program accessor and the
//! byte-accounting default -- is generated once here instead of being retyped
//! per case.

use crate::api::case::{
    prepared_as, prepared_as_mut, BaselineClass, BenchCase, BenchContext, BenchError, BenchId,
    BenchLayer, BenchMetadata, BenchRequirements, BenchRun, Correctness, DeterminismClass,
    PerformanceContract, PreparedCase, WorkloadClass,
};
use crate::api::suite::SuiteKind;
use vyre_foundation::ir::Program;

/// The speedup floor a case is held to, and what the floor is measured against.
#[derive(Clone, Copy)]
pub(crate) struct ContractDescription {
    pub(crate) primitive: &'static str,
    pub(crate) baseline_crate: &'static str,
    pub(crate) baseline_name: &'static str,
    pub(crate) baseline_class: BaselineClass,
    pub(crate) min_speedup_x: f64,
}

/// Everything about a benchmark case that is data rather than code.
pub(crate) struct WorkloadDescription {
    pub(crate) id: &'static str,
    pub(crate) name: &'static str,
    pub(crate) summary: &'static str,
    pub(crate) tags: &'static [&'static str],
    pub(crate) layer: BenchLayer,
    pub(crate) workload: WorkloadClass,
    pub(crate) determinism: DeterminismClass,
    pub(crate) owner_crate: &'static str,
    pub(crate) suites: &'static [SuiteKind],
    /// Named suites this case is active in *instead of* the built-in list.
    ///
    /// `SuiteKind::Custom` holds an `Arc<str>`, so a custom suite cannot appear
    /// in a `const` suite list. A case naming any custom suite here is active in
    /// exactly those and in no built-in suite.
    pub(crate) custom_suites: &'static [&'static str],
    pub(crate) needs_gpu: bool,
    pub(crate) needs_network: bool,
    pub(crate) min_vram_bytes: Option<u64>,
    pub(crate) min_input_bytes: Option<u64>,
    pub(crate) feature_set: &'static [&'static str],
    pub(crate) contract: Option<ContractDescription>,
}

/// The suites an honest-layer case runs in.
///
/// Two verbatim copies of this list coexisted, and a third copy was spelled out
/// per case. One per-case copy omitted `Smoke`, so `search.binary.u32.1m` was
/// excluded from every smoke run while its siblings were included.
pub(crate) const HONEST_SUITES: &[SuiteKind] = &[
    SuiteKind::Honest,
    SuiteKind::Deep,
    SuiteKind::Release,
    SuiteKind::Smoke,
];

impl WorkloadDescription {
    /// The honest-layer shape: a deterministic honest workload owned by this
    /// crate, needing a GPU with room for its buffers and no network.
    ///
    /// Only the identity, the prose, the tags, the memory floor and the contract
    /// vary between honest cases; the other nine fields were retyped per case.
    pub(crate) const fn honest(
        id: &'static str,
        name: &'static str,
        summary: &'static str,
        tags: &'static [&'static str],
        min_vram_bytes: u64,
        contract: Option<ContractDescription>,
    ) -> Self {
        Self {
            id,
            name,
            summary,
            tags,
            layer: BenchLayer::Honest,
            workload: WorkloadClass::Honest,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-bench",
            suites: HONEST_SUITES,
            custom_suites: &[],
            needs_gpu: true,
            needs_network: false,
            min_vram_bytes: Some(min_vram_bytes),
            min_input_bytes: None,
            feature_set: &[],
            contract,
        }
    }

    /// The fields every case in this crate shares, so a case's own description
    /// restates only what distinguishes it.
    ///
    /// `suites` is empty, which `BenchCase::active_in_suite` reads as membership
    /// in every suite. Update the identity, the prose, the tags and the
    /// classification through struct update syntax.
    pub(crate) const BASE: Self = Self {
        id: "",
        name: "",
        summary: "",
        tags: &[],
        layer: BenchLayer::Foundation,
        workload: WorkloadClass::Micro,
        determinism: DeterminismClass::Deterministic,
        owner_crate: "vyre-bench",
        suites: &[],
        custom_suites: &[],
        needs_gpu: true,
        needs_network: false,
        min_vram_bytes: None,
        min_input_bytes: None,
        feature_set: &[],
        contract: None,
    };
}

/// The workload-specific operations over a prepared payload of type `P`.
pub(crate) struct CaseOps<P: 'static> {
    /// Build the prepared payload. Runs outside the measured window.
    pub(crate) build: fn(&mut BenchContext) -> Result<P, BenchError>,
    /// Execute one measured sample and assemble its `BenchRun`.
    pub(crate) measure: fn(&mut BenchContext, &mut P) -> Result<BenchRun, BenchError>,
    /// Decide correctness from a finished run.
    pub(crate) verify: fn(&BenchRun) -> Result<Correctness, BenchError>,
    /// The IR program the runner may recompile, when the case exposes one.
    pub(crate) program: fn(&P) -> Option<&Program>,
    /// Workload identity when the case is a multi-program sequence and the
    /// single `program` fingerprint would under-describe it.
    pub(crate) fingerprint: Option<fn(&P) -> [u8; 32]>,
    /// Bytes this case reads and writes per sample.
    pub(crate) bytes_touched: fn(&P) -> (u64, u64),
}

/// This owner's name, as reported by every case it builds.
pub(crate) const HARNESS_OWNER: &str = "cases::harness::HarnessCase";

/// A registered benchmark case built from a description and its operations.
pub(crate) struct HarnessCase<P: 'static> {
    pub(crate) workload: &'static WorkloadDescription,
    pub(crate) ops: &'static CaseOps<P>,
}

impl<P: 'static> HarnessCase<P> {
    fn payload<'a>(&self, prepared: &'a PreparedCase) -> Result<&'a P, BenchError> {
        prepared_as::<P>(prepared, self.workload.id)
    }

    fn payload_mut<'a>(&self, prepared: &'a mut PreparedCase) -> Result<&'a mut P, BenchError> {
        let id = self.workload.id;
        prepared_as_mut::<P>(prepared, id)
    }
}

impl<P: 'static> BenchCase for HarnessCase<P> {
    fn id(&self) -> BenchId {
        BenchId(self.workload.id.to_string())
    }

    fn declaration_owner(&self) -> &'static str {
        HARNESS_OWNER
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: self.workload.name.to_string(),
            description: self.workload.summary.to_string(),
            tags: self
                .workload
                .tags
                .iter()
                .map(|tag| (*tag).to_string())
                .collect(),
            layer: self.workload.layer.clone(),
            workload: self.workload.workload.clone(),
            determinism: self.workload.determinism.clone(),
            owner_crate: self.workload.owner_crate.to_string(),
        }
    }

    fn suites(&self) -> &'static [SuiteKind] {
        self.workload.suites
    }

    fn active_in_suite(&self, suite: &SuiteKind) -> bool {
        let custom = self.workload.custom_suites;
        if !custom.is_empty() {
            return match suite {
                SuiteKind::Custom(name) => custom.iter().any(|entry| *entry == &**name),
                _ => false,
            };
        }
        let suites = self.workload.suites;
        suites.is_empty() || suites.contains(suite)
    }

    fn requirements(&self) -> BenchRequirements {
        BenchRequirements {
            needs_gpu: self.workload.needs_gpu,
            needs_network: self.workload.needs_network,
            min_vram_bytes: self.workload.min_vram_bytes,
            min_input_bytes: self.workload.min_input_bytes,
            feature_set: self
                .workload
                .feature_set
                .iter()
                .map(|feature| (*feature).to_string())
                .collect(),
        }
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        self.workload.contract.map(|contract| {
            PerformanceContract::min_speedup(
                contract.primitive,
                contract.baseline_crate,
                contract.baseline_name,
                contract.baseline_class,
                contract.min_speedup_x,
            )
        })
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new((self.ops.build)(ctx)?))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        (self.ops.program)(self.payload(prepared).ok()?)
    }

    fn workload_fingerprint_bytes(&self, prepared: &PreparedCase) -> Option<[u8; 32]> {
        let payload = self.payload(prepared).ok()?;
        match self.ops.fingerprint {
            Some(fingerprint) => Some(fingerprint(payload)),
            None => (self.ops.program)(payload).map(Program::fingerprint),
        }
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let payload = self.payload_mut(prepared)?;
        (self.ops.measure)(ctx, payload)
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        (self.ops.verify)(run)
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        self.payload(prepared)
            .map(self.ops.bytes_touched)
            .unwrap_or((0, 0))
    }
}

/// Default correctness check: exact equality against the captured baseline.
pub(crate) fn verify_exact(run: &BenchRun) -> Result<Correctness, BenchError> {
    run.verify_exact_outputs()
}

/// Default program accessor for cases whose payload exposes no single program.
pub(crate) fn no_program<P>(_prepared: &P) -> Option<&Program> {
    None
}

/// Program accessor for a case whose whole prepared payload is its program.
pub(crate) fn program_payload(prepared: &Program) -> Option<&Program> {
    Some(prepared)
}

#[cfg(test)]
mod tests {
    use super::{ContractDescription, WorkloadDescription, HONEST_SUITES};
    use crate::api::case::{BaselineClass, BenchCase, BenchLayer, DeterminismClass, WorkloadClass};
    use crate::api::suite::SuiteKind;
    use crate::cases::harness::{CaseOps, HarnessCase};

    static HONEST: WorkloadDescription = WorkloadDescription::honest(
        "x.y",
        "X Y",
        "does x to y",
        &["honest", "branchy"],
        4_096,
        Some(ContractDescription {
            primitive: "x",
            baseline_crate: "y",
            baseline_name: "y 1.0",
            baseline_class: BaselineClass::CpuSota,
            min_speedup_x: 7.0,
        }),
    );

    static OPS: CaseOps<()> = CaseOps {
        build: |_| Ok(()),
        measure: |_, _| unreachable!("no sample is dispatched in a declaration test"),
        verify: super::verify_exact,
        program: super::no_program,
        fingerprint: None,
        bytes_touched: |_| (0, 0),
    };

    static CASE: HarnessCase<()> = HarnessCase {
        workload: &HONEST,
        ops: &OPS,
    };

    /// An honest case runs in the smoke suite. One hand-rolled copy of this list
    /// omitted `Smoke`, so `search.binary.u32.1m` was excluded from every smoke
    /// run while its six siblings were included.
    #[test]
    fn honest_suites_include_every_suite_an_honest_case_runs_in() {
        assert!(
            HONEST_SUITES.contains(&SuiteKind::Smoke),
            "Fix: an honest case that is not in the smoke suite is never smoke-tested"
        );
        assert!(HONEST_SUITES.contains(&SuiteKind::Honest));
        assert!(HONEST_SUITES.contains(&SuiteKind::Deep));
        assert!(HONEST_SUITES.contains(&SuiteKind::Release));
    }

    /// The classification an honest case carries is fixed; only its identity, its
    /// prose, its tags, its memory floor and its contract vary.
    #[test]
    fn honest_classification_is_fixed() {
        let metadata = CASE.metadata();

        assert_eq!(metadata.id.0, "x.y");
        assert_eq!(metadata.name, "X Y");
        assert_eq!(metadata.description, "does x to y");
        assert_eq!(metadata.tags, vec!["honest", "branchy"]);
        assert!(matches!(metadata.layer, BenchLayer::Honest));
        assert!(matches!(metadata.workload, WorkloadClass::Honest));
        assert!(matches!(
            metadata.determinism,
            DeterminismClass::Deterministic
        ));
        assert_eq!(metadata.owner_crate, "vyre-bench");
        assert_eq!(CASE.suites(), HONEST_SUITES);
    }

    /// An honest case declares device memory, never a host input floor, and never
    /// a feature gate: it is the plain GPU shape.
    #[test]
    fn honest_requirements_declare_vram_only() {
        let requirements = CASE.requirements();

        assert!(requirements.needs_gpu);
        assert!(!requirements.needs_network);
        assert_eq!(requirements.min_vram_bytes, Some(4_096));
        assert_eq!(requirements.min_input_bytes, None);
        assert!(requirements.feature_set.is_empty());
    }

    /// A declared contract reaches the published record with its floor intact.
    #[test]
    fn declared_contract_reaches_the_published_record() {
        let contract = CASE
            .performance_contract()
            .expect("Fix: a declared contract must be published");

        assert_eq!(contract.primitive, "x");
        assert_eq!(contract.baselines.len(), 1);
        assert_eq!(contract.baselines[0].crate_name, "y");
        assert_eq!(contract.baselines[0].name, "y 1.0");
        assert_eq!(contract.baselines[0].min_speedup_x, 7.0);
    }

    /// Every case the harness builds reports the harness as its owner. The
    /// declaration gate reads this to tell a declared case from an open-coded one.
    #[test]
    fn a_harness_case_reports_the_harness_as_its_owner() {
        assert_eq!(CASE.declaration_owner(), super::HARNESS_OWNER);
        assert_ne!(super::HARNESS_OWNER, "");
    }
}
