//! One case-builder harness for benchmark cases.
//!
//! A benchmark case is a static [`WorkloadDescription`] plus the handful of
//! operations a description cannot carry. Everything else -- the `BenchId`, the
//! `BenchMetadata`, the suite list, the requirements record, the performance
//! contract, the prepared-payload downcast, the program accessor and the
//! byte-accounting default -- is generated once here instead of being retyped
//! per case.

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PerformanceContract, PreparedCase, WorkloadClass,
};
use crate::api::suite::SuiteKind;
use vyre_foundation::ir::Program;

/// The CPU-baseline speedup floor a case is held to.
#[derive(Clone, Copy)]
pub(crate) struct ContractDescription {
    pub(crate) primitive: &'static str,
    pub(crate) baseline_crate: &'static str,
    pub(crate) baseline_name: &'static str,
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
    pub(crate) needs_gpu: bool,
    pub(crate) needs_network: bool,
    pub(crate) min_vram_bytes: Option<u64>,
    pub(crate) min_input_bytes: Option<u64>,
    pub(crate) feature_set: &'static [&'static str],
    pub(crate) contract: Option<ContractDescription>,
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

/// A registered benchmark case built from a description and its operations.
pub(crate) struct HarnessCase<P: 'static> {
    pub(crate) workload: &'static WorkloadDescription,
    pub(crate) ops: &'static CaseOps<P>,
}

impl<P: 'static> HarnessCase<P> {
    fn payload<'a>(&self, prepared: &'a PreparedCase) -> Result<&'a P, BenchError> {
        prepared.downcast_ref::<P>().ok_or_else(|| {
            BenchError::ExecutionFailed(format!(
                "prepared payload for `{}` had the wrong type. Fix: keep the case build and measure steps on one payload type.",
                self.workload.id
            ))
        })
    }

    fn payload_mut<'a>(&self, prepared: &'a mut PreparedCase) -> Result<&'a mut P, BenchError> {
        let id = self.workload.id;
        prepared.downcast_mut::<P>().ok_or_else(|| {
            BenchError::ExecutionFailed(format!(
                "prepared payload for `{id}` had the wrong type. Fix: keep the case build and measure steps on one payload type."
            ))
        })
    }
}

impl<P: 'static> BenchCase for HarnessCase<P> {
    fn id(&self) -> BenchId {
        BenchId(self.workload.id.to_string())
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
            PerformanceContract::cpu_sota_min_speedup(
                contract.primitive,
                contract.baseline_crate,
                contract.baseline_name,
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
