// One dispatch path under comparison: the reference interpreter, or a
// registered backend compiled through a production session.

use std::env;

use vyre::ir::Program;
use vyre_conform::witness_plan::{plan_witness_inputs_into, WitnessInputPlan};
use vyre_driver::BackendRegistration;
use vyre_reference::value::Value;

use super::parity_matrix_divergence::Summary;

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) enum BackendKind {
    ReferenceBackend,
    Registered(&'static BackendRegistration),
}

pub(crate) struct BackendRunner {
    pub(crate) id: &'static str,
    pub(crate) kind: BackendKind,
}

impl BackendRunner {
    pub(crate) fn execute(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        values: &mut Vec<Value>,
    ) -> Result<Vec<Vec<u8>>, String> {
        match &self.kind {
            BackendKind::ReferenceBackend => {
                values.clear();
                for bytes in inputs {
                    values.push(Value::from(bytes.as_slice()));
                }
                vyre_reference::reference_eval(program, values)
                    .map(|outputs| outputs.into_iter().map(|value| value.to_bytes()).collect())
                    .map_err(|error| format!("reference dispatch failed: {error}"))
            }
            BackendKind::Registered(_) => {
                let mut backend_inputs = Vec::new();
                self.execute_with_plan(program, inputs, values, None, &mut backend_inputs)
            }
        }
    }

    pub(crate) fn execute_with_plan<'a>(
        &self,
        program: &Program,
        inputs: &'a [Vec<u8>],
        values: &mut Vec<Value>,
        plan: Option<&'a WitnessInputPlan>,
        backend_inputs: &mut Vec<&'a [u8]>,
    ) -> Result<Vec<Vec<u8>>, String> {
        match &self.kind {
            BackendKind::ReferenceBackend => {
                values.clear();
                if let Some(plan) = plan {
                    plan_witness_inputs_into(inputs, plan, backend_inputs)?;
                    for bytes in backend_inputs.iter() {
                        values.push(Value::from(*bytes));
                    }
                } else {
                    for bytes in inputs {
                        values.push(Value::from(bytes.as_slice()));
                    }
                }
                vyre_reference::reference_eval(program, values)
                    .map(|outputs| outputs.into_iter().map(|value| value.to_bytes()).collect())
                    .map_err(|error| format!("reference dispatch failed: {error}"))
            }
            BackendKind::Registered(registration) => {
                let run_submission = |planned_inputs: &[&[u8]]| -> Result<Vec<Vec<u8>>, String> {
                    let production =
                        vyre_conform::production::ProductionSession::from_registration(
                            program,
                            registration,
                        )
                        .map_err(|error| error.to_string())?;
                    production
                        .submit(planned_inputs)
                        .map(|execution| execution.outputs)
                        .map_err(|error| error.to_string())
                };

                if let Some(plan) = plan {
                    plan_witness_inputs_into(inputs, plan, backend_inputs)?;
                    run_submission(backend_inputs)
                } else {
                    let plan_storage = WitnessInputPlan::for_program(program)?;
                    let mut local_inputs = Vec::new();
                    plan_witness_inputs_into(inputs, &plan_storage, &mut local_inputs)?;
                    run_submission(&local_inputs)
                }
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
