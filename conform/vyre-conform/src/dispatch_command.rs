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
    // The registry is a static built at link time, so a lookup inside the loop
    // answers the same question once per op. An id no linked driver registers
    // used to become one identical conformance failure per op: a spirv run
    // wrote 349 rows saying `unknown backend`, and a reader counting rows saw a
    // judged backend with 349 defects rather than a backend nothing judged.
    let backend = backend_registration(backend_id)?;
    let mut pairs = Vec::with_capacity(selected_entries.len());

    for entry in selected_entries {
        let prepared = match prepare_entry(entry) {
            Ok(prepared) => prepared,
            Err(error) => {
                pairs.push(ConformanceResult {
                    op_id: entry.id.into(),
                    backend_id: backend.id.into(),
                    passed: false,
                    message: error,
                    replay_capsule: None,
                });
                continue;
            }
        };
        pairs.push(compare_backend_against_reference(backend, &prepared));
    }

    Ok(pairs)
}
