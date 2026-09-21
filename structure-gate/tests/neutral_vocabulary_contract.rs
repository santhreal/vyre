//! The neutral-vocabulary rule's matcher, held against its own wordlist.
//!
//! The rule is a wordlist, so the way it compares a word to a line IS the rule.
//! A matcher that reads `warp` but not `warps` covers whichever spelling the
//! author happened not to use, and the finding count stays at zero while the
//! vendor word sits in the tree. That is the same failure as having no rule, and
//! it is invisible from the outside because both states report nothing.
//!
//! Every case below derives its inputs from `backend-vocabulary.toml` and
//! `docs/CRATE_OWNERSHIP.toml` at run time. A term added to the wordlist, a
//! layer that becomes neutral, or a new interface allowance is covered without
//! an edit here, so a row cannot join the contract untested.

use structure_gate::backend_vocabulary::{
    contract_failures, segments_of, vocabulary_failures, Neutrality,
};
use structure_gate::workspace_root;

/// The contract as the checkout states it.
fn contract() -> Neutrality {
    Neutrality::read(&workspace_root())
        .expect("Fix: the neutral-vocabulary contract must be readable for any rule to cover it")
}

/// Every term is matched in the singular, so no row is dead on arrival.
#[test]
fn every_term_is_matched_in_the_spelling_its_row_states() {
    let neutrality = contract();
    assert!(
        !neutrality.contract.terms.is_empty(),
        "Fix: the wordlist declares no term, so the rule covers nothing"
    );
    for term in &neutrality.contract.terms {
        let line = format!("// this line names {} once", term.word);
        let found = neutrality.words_in(&line);
        assert!(
            found.iter().any(|hit| hit.word == term.word),
            "Fix: `{}` is a declared term and the matcher does not find it in `{line}`",
            term.word
        );
    }
}

/// Every term is matched in its plural too.
///
/// This is the case the matcher used to miss. `AFFINE_GROUPED_WARPS_PER_WORKGROUP`
/// and `Sub-warps` sat in substrate-neutral crates while the sweep reported zero,
/// because the comparison required the final segment to equal the term exactly.
///
/// The plural is built from the term's own segments rather than from its raw
/// spelling, because the segment splitter breaks an all-caps run before a
/// trailing lowercase letter: `CUDAs` is `cud` and `as`, which is a spelling no
/// author writes, while `cudas`, `Cudas` and `CUDAS` all reach one segment.
#[test]
fn every_term_is_matched_in_its_plural() {
    let neutrality = contract();
    for term in &neutrality.contract.terms {
        let stem = segments_of(&term.word).join("_");
        for plural in [
            format!("{stem}s"),
            format!("{}S", stem.to_ascii_uppercase()),
            format!("Selected{}s", capitalized(&stem)),
        ] {
            let line = format!("// this line names {plural} once");
            let found = neutrality.words_in(&line);
            assert!(
                found.iter().any(|hit| hit.word == term.word),
                "Fix: `{plural}` is `{}` in the plural and the matcher does not find it, so the \
                 wordlist covers one spelling of its own term",
                term.word
            );
        }
    }
}

/// `text` with its first letter uppercased.
fn capitalized(text: &str) -> String {
    let mut letters = text.chars();
    match letters.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + letters.as_str(),
        None => String::new(),
    }
}

/// A plural is one trailing `s` and nothing longer.
///
/// The bound matters in both directions: without it the rule would report any
/// identifier whose final segment merely starts with a term, which is the
/// substring matching the segment comparison exists to avoid.
#[test]
fn a_longer_suffix_is_not_a_plural() {
    let neutrality = contract();
    for term in &neutrality.contract.terms {
        for suffix in ["ing", "ed", "like", "ish"] {
            let derived = format!("{}{suffix}", term.word);
            let line = format!("// this line names {derived} once");
            assert!(
                !neutrality
                    .words_in(&line)
                    .iter()
                    .any(|hit| hit.word == term.word),
                "Fix: `{derived}` is not `{}` and the matcher reports it, so the rule fires on \
                 words it never decided about",
                term.word
            );
        }
    }
}

/// A term's letters inside one longer segment are not the term.
///
/// Held against the lowercase spelling of the term, because that is the one form
/// that joins its neighbours into a single segment. Mixed case would split at the
/// case boundary and test the splitter instead of the comparison.
#[test]
fn a_term_inside_a_longer_segment_is_not_the_term() {
    let neutrality = contract();
    for term in &neutrality.contract.terms {
        let segments = segments_of(&term.word);
        if segments.len() != 1 {
            continue;
        }
        let buried = format!("bar{}ista", segments[0]);
        assert_eq!(
            segments_of(&buried).len(),
            1,
            "Fix: `{buried}` must be one segment for this case to test what it claims"
        );
        let line = format!("// this line names {buried} once");
        assert!(
            !neutrality
                .words_in(&line)
                .iter()
                .any(|hit| hit.word == term.word),
            "Fix: `{buried}` carries the letters of `{}` and nothing else, and the matcher \
             reports it as the term",
            term.word
        );
    }
}

/// Camel case does not hide a term from the rule.
#[test]
fn a_camel_case_identifier_states_the_term_it_carries() {
    let neutrality = contract();
    for term in &neutrality.contract.terms {
        let segments = segments_of(&term.word);
        if segments.len() != 1 {
            continue;
        }
        let mut camel = String::from("Selected");
        let mut letters = segments[0].chars();
        if let Some(first) = letters.next() {
            camel.push(first.to_ascii_uppercase());
            camel.extend(letters);
        }
        camel.push_str("Device");
        let line = format!("pub struct {camel} {{}}");
        assert!(
            neutrality
                .words_in(&line)
                .iter()
                .any(|hit| hit.word == term.word),
            "Fix: `{camel}` states `{}` and the matcher does not find it, so a type name hides \
             from the rule",
            term.word
        );
    }
}

/// Every roster member is reached by the rule.
///
/// The roster is derived from the layer table crossed with the ownership
/// registry, so this is what proves a crate that joins a neutral layer is
/// actually scanned rather than merely eligible.
#[test]
fn every_roster_member_is_reached_by_the_rule() {
    let neutrality = contract();
    let roster = neutrality.roster();
    assert!(
        !roster.is_empty(),
        "Fix: the roster is empty, so the rule scans no crate"
    );
    let term = neutrality
        .contract
        .terms
        .first()
        .expect("Fix: the wordlist must declare a term");
    for (package, directory) in roster {
        let file = format!("{directory}/src/lib.rs");
        let line = format!("// {} is named here", term.word);
        let failures = vocabulary_failures(&neutrality, [(file.clone(), 1, line)]);
        assert_eq!(
            failures.len(),
            1,
            "Fix: `{package}` is on the roster and a line under `{file}` naming `{}` produces \
             no finding",
            term.word
        );
    }
}

/// An interface allowance blanks its own name and only under its own directory.
#[test]
fn an_interface_allowance_reaches_its_own_directory_only() {
    let neutrality = contract();
    assert!(
        !neutrality.contract.interfaces.is_empty(),
        "Fix: no interface allowance is declared, so this case tests nothing"
    );
    for interface in &neutrality.contract.interfaces {
        let inside = format!("{}named.rs", interface.prefix);
        let line = format!("// {} appears here", interface.name);
        assert!(
            !neutrality.mask_interface_names(&line, &inside).contains(
                interface
                    .name
                    .split(['-', '_', '.'])
                    .next()
                    .unwrap_or(&interface.name)
            ),
            "Fix: `{}` is allowed under `{}` and is not blanked there",
            interface.name,
            interface.prefix
        );
        let outside = format!("elsewhere/{}", inside);
        assert!(
            neutrality
                .mask_interface_names(&line, &outside)
                .contains(&interface.name),
            "Fix: `{}` is allowed only under `{}` and is blanked under `{outside}` as well",
            interface.name,
            interface.prefix
        );
    }
}

/// Masking preserves column positions so a reported column still maps to source.
#[test]
fn masking_preserves_the_width_of_the_line() {
    let neutrality = contract();
    for interface in &neutrality.contract.interfaces {
        let file = format!("{}named.rs", interface.prefix);
        let line = format!("// {} appears here", interface.name);
        assert_eq!(
            neutrality.mask_interface_names(&line, &file).len(),
            line.len(),
            "Fix: masking `{}` changes the line width, so a reported column no longer maps to \
             the source",
            interface.name
        );
    }
}

/// The contract data still describes the tree it judges.
#[test]
fn the_contract_data_still_describes_the_checkout() {
    let root = workspace_root();
    let neutrality = contract();
    let failures = contract_failures(&root, &neutrality);
    assert!(
        failures.is_empty(),
        "Fix: the neutral-vocabulary contract no longer matches the checkout:\n{}",
        failures.join("\n")
    );
}

/// No substrate-neutral crate names a concrete backend in production source.
#[test]
fn no_neutral_crate_names_a_concrete_backend() {
    let failures =
        structure_gate::backend_vocabulary::neutral_vocabulary_failures(&workspace_root());
    assert!(
        failures.is_empty(),
        "Fix: {} substrate-neutral production line(s) name a concrete backend:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
