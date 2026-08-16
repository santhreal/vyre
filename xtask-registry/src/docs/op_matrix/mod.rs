//! Hold `docs/optimization/OP_MATRIX.toml` to the live operation registry.
//!
//! The gate renders the matrix from two sources, the manual scan construct rows
//! and the live registry, validates the merged rows, and reports the artifact
//! it would write. One module per stage.

mod manual_rows;
mod record;
mod registered_rows;
mod render;
mod validation;

use std::path::Path;

use xtask::artifact_gate::Inspection;

use manual_rows::manual_records;
use registered_rows::registered_records;
use render::render_matrix;
use validation::validate_records;

/// The artifact this gate owns, relative to the workspace root.
///
/// It used to be a caller flag defaulting to a relative path, so the matrix a
/// run compared against depended on the working directory the run started in.
const MATRIX_PATH: &str = "docs/optimization/OP_MATRIX.toml";

xtask::artifact_gate! {
    /// Holds `docs/optimization/OP_MATRIX.toml` to the live operation registry.
    OpMatrixGate,
    name: "op-matrix",
    help: "Render the canonical op matrix from the live operation registry and the manual scan \
       construct rows, and report each line docs/optimization/OP_MATRIX.toml disagrees on. \
       Proves every registered op id carries a canonical tier namespace, is registered by one \
       semantic source, appears in one family, and that every family declares owners and \
       tests. Proves nothing about whether the named owner and test paths exist.",
    inspect: |ctx| inspect(&ctx.root),
}

/// The matrix the registry generates, and every row problem found producing it.
fn inspect(root: &Path) -> Inspection {
    let mut inspection = Inspection::new();
    let (matrix, problems) = build_matrix(root);
    for problem in problems {
        inspection.blocked(
            MATRIX_PATH,
            problem,
            "Correct the registration or the manual scan row the sentence names. The row is left \
             out of the matrix until it is, so the artifact understates the op surface.",
        );
    }
    inspection.generates_text(MATRIX_PATH, matrix);
    inspection
}

/// The rendered matrix, and every problem found in the rows that produced it.
///
/// Validation used to stop at the first violation and abort the run before
/// anything was rendered, so a second duplicate family stayed invisible until
/// the first was fixed and the artifact went unrefreshed exactly when it was
/// most wrong. Every violation is now collected and the matrix is rendered from
/// the rows that survived.
fn build_matrix(root: &Path) -> (String, Vec<String>) {
    let mut problems = Vec::new();
    let mut records = manual_records();
    records.extend(registered_records(root, &mut problems));
    problems.extend(validate_records(&records));

    records.sort_by(|left, right| {
        (
            left.tier.matrix_value(),
            left.family.as_str(),
            left.ops.first().map(String::as_str),
        )
            .cmp(&(
                right.tier.matrix_value(),
                right.family.as_str(),
                right.ops.first().map(String::as_str),
            ))
    });

    (render_matrix(&records), problems)
}
