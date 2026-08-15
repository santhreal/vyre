//! Gate: every C-AST fixture builder is named by its family's parity case table.
//!
//! # The defect class this closes
//!
//! A C-AST family has two arms, a CPU classification arm in `vyre-libs/tests`
//! and a backend parity arm in a driver's `tests`. Each arm used to enumerate
//! cases by writing one `#[test]` per fixture it happened to remember, so the
//! two lists were independent and drifted:
//!
//!   * `fixture_gnu_restrict_qualifier` was named by the CPU arm and by no
//!     backend arm, leaving GNU `__restrict` normalization CPU-proven and
//!     backend-unproven;
//!   * `fixture_inner_typedef_shadows_outer` reached no backend arm either, so
//!     the one semantic-gap construct that exists to exercise scope-dependent
//!     typedef visibility was never dispatched;
//!   * six declarator-matrix cases and one advanced-declaration case reached a
//!     backend classifier but never a backend property-graph lowerer.
//!
//! Both arms now iterate the family's `CASES` table, so reaching the table
//! reaches both arms at once and no arm has a list of its own. That moves the
//! whole class down to one question: is anything missing from the table? This
//! gate answers it against the fixture file on disk.
//!
//! # Why it cannot go stale
//!
//! The member set is read at run time out of each fixture source file, by
//! `parity_matrix::fixture_names_in_source`, so a new `pub(crate) fn fixture_*`
//! turns this test RED until the table records a decision about it. A hardcoded
//! list of expected fixtures here would be the same defect one level up.
//!
//! # What it does not catch
//!
//! It cannot tell whether a case's assertions are meaningful, only that both
//! arms run it. It also cannot see a family whose fixture file this gate does
//! not name; `families_named_here_match_the_fixture_directory` covers that by
//! reading the fixtures directory and requiring each C-AST family file it finds
//! to be either gated or explicitly excused.

use std::collections::BTreeSet;
use std::path::PathBuf;

use vyre_test_support::monorepo::vyre_workspace_root;

use crate::c_frontend::parity_matrix::{assert_case_table_covers_fixture_file, ParityCase};
use crate::{
    declaration_advanced_constructs, declarator_matrix_constructs, semantic_gap_constructs,
};

/// Directory holding the shared C-frontend fixture files.
fn fixtures_directory() -> PathBuf {
    vyre_workspace_root().join("tests/support/c_frontend/fixtures")
}

/// Read one fixture source file, failing with the path when it is not there.
fn fixture_source(file: &str) -> String {
    let path = fixtures_directory().join(file);
    std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "Fix: cannot read the fixture source at {} ({error}); the gate derives its member \
             set from that file, so point it at the file the family actually includes",
            path.display()
        )
    })
}

/// Every family whose fixtures carry a parity case table.
fn gated_families() -> Vec<(&'static str, &'static str, &'static [ParityCase])> {
    vec![
        (
            "declarator_matrix_constructs",
            "declarator_matrix_constructs.rs",
            declarator_matrix_constructs::CASES,
        ),
        (
            "declaration_advanced_constructs",
            "declaration_advanced_constructs.rs",
            declaration_advanced_constructs::CASES,
        ),
        (
            "semantic_gap_constructs",
            "semantic_gap_constructs.rs",
            semantic_gap_constructs::CASES,
        ),
    ]
}

#[test]
fn every_fixture_builder_is_a_parity_case() {
    for (family, file, cases) in gated_families() {
        assert_case_table_covers_fixture_file(family, &fixture_source(file), cases);
    }
}

/// Fixture files that carry no parity case table, each with the reason.
///
/// A fixture family belongs here only when its constructs are proven by a
/// contract that is not a four-stage CPU/backend comparison, which is why the
/// reason is recorded next to the name rather than in a commit message.
///
/// A file that is not a family at all, such as a token stream two families
/// index by row, belongs here for the same reason: the directory is the set, so
/// every file in it carries a decision.
const FAMILIES_WITHOUT_A_CASE_TABLE: &[(&str, &str)] = &[
    (
        "asm_extended_operands.rs",
        "extended-asm operand and goto-label lowering asserts property-graph edges, not a \
         four-stage row comparison",
    ),
    (
        "complete_construct_corpus.rs",
        "a corpus sweep over whole translation units, enumerated by the corpus rather than by \
         one construct per case",
    ),
    (
        "compound_literal_designated_init.rs",
        "compound-literal lowering asserts property-graph shape, not row-for-row parity",
    ),
    (
        "declaration_container_nodes.rs",
        "container-node contracts assert parent and child links, not stage parity",
    ),
    (
        "expression_ambiguity.rs",
        "expression-shape families run through the expression-shape buffer, which is a fifth \
         stage the parity matrix does not model",
    ),
    (
        "expression_builtin.rs",
        "expression-shape family; see expression_ambiguity.rs",
    ),
    (
        "expression_postfix.rs",
        "expression-shape family; see expression_ambiguity.rs",
    ),
    (
        "expression_precedence.rs",
        "expression-shape family; see expression_ambiguity.rs",
    ),
    (
        "expression_precedence_e2e.rs",
        "expression-shape family; see expression_ambiguity.rs",
    ),
    (
        "expression_shape_gap_constructs.rs",
        "expression-shape family; see expression_ambiguity.rs",
    ),
    (
        "expression_shape_pg.rs",
        "expression-shape family; see expression_ambiguity.rs",
    ),
    (
        "gemini_named_fixtures.rs",
        "named single-construct regressions, each asserting one row rather than a stage chain",
    ),
    (
        "gnu_attribute_statements.rs",
        "GNU attribute statement lowering asserts property-graph edges",
    ),
    (
        "gnu_builtin_control_flow.rs",
        "GNU builtin control flow asserts property-graph edges",
    ),
    (
        "initializer_designator_streams.rs",
        "a token stream shared by the two initializer-designator families, indexed by row from \
         both, so the parity is asserted in each family and not from here",
    ),
    (
        "linux_macro_builtin_qualifier.rs",
        "preprocessor-stage corpus: its fixtures start from source bytes, before the VAST \
         builder the matrix begins at",
    ),
    (
        "pg_lowering_deep_constructs.rs",
        "deep property-graph lowering asserts semantic node categories and edges",
    ),
    (
        "switch_case_complex_bodies.rs",
        "switch body lowering asserts property-graph edges",
    ),
    (
        "typedef_disambiguation.rs",
        "typedef disambiguation asserts annotation flags per row against a hand-built stream",
    ),
    (
        "vast_builder_token_streams.rs",
        "raw builder streams, deliberately malformed in places, so no classified or lowered \
         stage exists to compare",
    ),
];

#[test]
fn families_named_here_match_the_fixture_directory() {
    let directory = fixtures_directory();
    let mut found = BTreeSet::new();
    for entry in std::fs::read_dir(&directory).unwrap_or_else(|error| {
        panic!(
            "Fix: cannot list {} ({error}); the gate derives its family set from that directory",
            directory.display()
        )
    }) {
        let entry = entry.expect("Fix: unreadable directory entry under the fixtures directory");
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".rs") {
            found.insert(name);
        }
    }
    assert!(
        !found.is_empty(),
        "Fix: found no fixture files under {}; the gate cannot derive a family set",
        directory.display()
    );

    let gated: BTreeSet<String> = gated_families()
        .iter()
        .map(|(_, file, _)| (*file).to_string())
        .collect();
    let excused: BTreeSet<String> = FAMILIES_WITHOUT_A_CASE_TABLE
        .iter()
        .map(|(file, _)| (*file).to_string())
        .collect();

    let undecided: Vec<&String> = found
        .iter()
        .filter(|file| !gated.contains(*file) && !excused.contains(*file))
        .collect();
    assert!(
        undecided.is_empty(),
        "Fix: fixture file(s) {undecided:?} are neither covered by a parity CASES table nor \
         listed in FAMILIES_WITHOUT_A_CASE_TABLE with a reason; give the family a CASES table so \
         both arms run it, or record why it is proven another way"
    );

    let vanished: Vec<&String> = gated
        .iter()
        .chain(&excused)
        .filter(|file| !found.contains(*file))
        .collect();
    assert!(
        vanished.is_empty(),
        "Fix: this gate names fixture file(s) {vanished:?} that no longer exist under the \
         fixtures directory; drop the stale entries so the family set matches the tree"
    );
}
