// One dispatch path under comparison: the reference interpreter, or a
// registered backend compiled through a production session.

use std::env;

use vyre::ir::Program;
use vyre_conform::witness_plan::{plan_witness_inputs_into, WitnessInputPlan};
use vyre_driver::BackendRegistration;
use vyre_reference::value::Value;

use super::parity_matrix_divergence::Summary;

pub(crate) enum BackendKind {
    ReferenceBackend,
    // Only the device lane constructs this. Without `gpu` the variant still
    // exists for the match arm below, and nothing builds it, so the suppression
    // names that one variant and expires with the feature.
    #[cfg_attr(
        not(feature = "gpu"),
        expect(dead_code, reason = "constructed only by the gpu-gated device lane")
    )]
    Registered(&'static BackendRegistration),
}

pub(crate) struct BackendRunner {
    pub(crate) id: &'static str,
    pub(crate) kind: BackendKind,
}

impl BackendRunner {
    pub(crate) fn execute_with_plan<'a>(
        &self,
        program: &Program,
        inputs: &'a [Vec<u8>],
        values: &mut Vec<Value>,
        plan: &'a WitnessInputPlan,
        backend_inputs: &mut Vec<&'a [u8]>,
    ) -> Result<Vec<Vec<u8>>, String> {
        plan_witness_inputs_into(inputs, plan, backend_inputs)?;
        match &self.kind {
            BackendKind::ReferenceBackend => {
                values.clear();
                for bytes in backend_inputs.iter() {
                    values.push(Value::from(*bytes));
                }
                vyre_reference::reference_eval(program, values)
                    .map(|outputs| outputs.into_iter().map(|value| value.to_bytes()).collect())
                    .map_err(|error| format!("reference dispatch failed: {error}"))
            }
            BackendKind::Registered(registration) => {
                let production = vyre_conform::production::ProductionSession::from_registration(
                    program,
                    registration,
                )
                .map_err(|error| error.to_string())?;
                production
                    .submit(backend_inputs)
                    .map(|execution| execution.outputs)
                    .map_err(|error| error.to_string())
            }
        }
    }
}

pub(crate) fn backend_runners(summary: &mut Summary) -> Vec<BackendRunner> {
    let selected = env::var("VYRE_BACKEND")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let mut registrations: Vec<&'static BackendRegistration> =
        vyre_registry_link::backend::live_backend_registry()
            .expect("valid backend registry")
            .iter()
            .collect();
    registrations.retain(|registration| {
        // Runner one already is the reference, so keeping the reference oracle
        // here would compare it against itself for every op.
        !registration.reference_oracle
            && selected
                .as_deref()
                .is_none_or(|backend| registration.id == backend)
    });
    registrations.sort_by(|left, right| left.id.cmp(right.id));
    summary.backends_linked = registrations.len() + 1;

    let mut runners = vec![BackendRunner {
        id: "reference",
        kind: BackendKind::ReferenceBackend,
    }];

    for registration in registrations {
        if let Some(runner) = build_backend_runner(registration) {
            runners.push(runner);
        }
    }

    summary.backends_runnable = runners.len();
    runners
}

pub(crate) fn build_backend_runner(
    registration: &'static BackendRegistration,
) -> Option<BackendRunner> {
    (registration.target_compiler.is_some() && registration.materializer.is_some()).then_some(
        BackendRunner {
            id: registration.id,
            kind: BackendKind::Registered(registration),
        },
    )
}
