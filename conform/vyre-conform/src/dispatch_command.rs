//! The `dispatch` subcommand: sequential single-backend conformance over selected ops.

use crate::backend_selection::backend_registration;
use crate::operation_selection::{prepare_entry, select_entries, unified_entries};
use crate::reference_parity::compare_backend_against_reference;
use vyre_conform_spec::ConformanceResult;

pub(crate) fn dispatch_pairs(
    backend_id: &str,
    ops: &str,
) -> Result<Vec<ConformanceResult>, String> {
    let entries = unified_entries();
    let selected_entries = select_entries(&entries, ops, None)?;
    let mut pairs = Vec::with_capacity(selected_entries.len());
    let backend_id = backend_id.to_string();

    for entry in selected_entries {
        let prepared = match prepare_entry(entry) {
            Ok(prepared) => prepared,
            Err(error) => {
                pairs.push(ConformanceResult {
                    op_id: entry.id.into(),
                    backend_id: backend_id.clone(),
                    passed: false,
                    message: error,
                    replay_capsule: None,
                });
                continue;
            }
        };
        let backend = match backend_registration(&backend_id) {
            Ok(backend) => backend,
            Err(error) => {
                pairs.push(ConformanceResult {
                    op_id: entry.id.into(),
                    backend_id: backend_id.clone(),
                    passed: false,
                    message: format!(
                        "backend acquisition failed before dispatch: {error}. Fix: isolate or reset the backend after the preceding failing op, then repair the op that poisoned device state."
                    ),
                    replay_capsule: None,
                });
                continue;
            }
        };
        pairs.push(compare_backend_against_reference(backend, &prepared));
    }

    Ok(pairs)
}
