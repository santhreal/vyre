//! A source file may not claim a tier its own placement contradicts.
//!
//! [`super::ops::tier_of`] derives an op's tier from its registered id alone,
//! so the tier follows placement and nothing a comment says can change it. That
//! makes a prose tier claim either redundant or false, and thirty of them in
//! `vyre-libs` were false: they read "Tier 2.5", which the composability check
//! reads as *must not compose*, on ops that are compositions by construction.
//!
//! The check reads the tier from the id and the claims from the file, so
//! neither side is a hardcoded list. Moving an op between crates re-derives its
//! tier and turns every stale claim in its file red.

use super::*;
use std::collections::BTreeMap;

/// The tier vocabulary a comment may claim: 2, 2.5 and 3. A tier this gate
/// cannot name is not a claim it can judge, so `Other` reads as no claim.
fn tier_number(tier: Tier) -> Option<&'static str> {
    match tier {
        Tier::T2 => Some("2"),
        Tier::T2_5 => Some("2.5"),
        Tier::T3 => Some("3"),
        Tier::Other => None,
    }
}

/// Every distinct `Tier <n>` number named in `text`, with the 1-based line.
///
/// `Tier-3` and `Tier 3` are the same claim, and a trailing `.5` belongs to the
/// number rather than to the sentence, so `Tier 2.5.` reads as 2.5 and not as
/// 2. A number is only read when `Tier` starts a word, so `Frontier 3` is not a
/// claim.
fn tier_claims(text: &str) -> Vec<(usize, String)> {
    let mut claims = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let mut rest = line;
        while let Some(offset) = rest.find("Tier") {
            let (before, at) = rest.split_at(offset);
            if before
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_')
            {
                rest = &at[4..];
                continue;
            }
            let tail = at[4..].trim_start_matches([' ', '-']);
            let digits: String = tail
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let number = digits.trim_end_matches('.');
            if !number.is_empty() {
                claims.push((index + 1, number.to_string()));
            }
            rest = &at[4..];
        }
    }
    claims
}

pub(super) fn check_12_tier_claims(report: &mut Report, ops: &[OpInfo]) -> usize {
    report.note(
        "Tier claims (no source file names a tier its registered op ids contradict)".to_string(),
    );
    let Some(root) = workspace_root() else {
        report.find(violation(
            "  ✗ workspace root not reachable from xtask. Fix: run from the vyre workspace checkout."
                .to_string(),
        ));
        return 1;
    };

    // One file can register many ops. They agree on tier unless the file
    // straddles crates, which placement already forbids, so collect the set and
    // accept any member.
    let mut by_file: BTreeMap<&str, BTreeSet<&'static str>> = BTreeMap::new();
    for op in ops {
        if op.source_file.is_empty() || op.source_file == "<unattributed>" {
            continue;
        }
        if let Some(number) = tier_number(op.tier) {
            by_file
                .entry(op.source_file.as_str())
                .or_default()
                .insert(number);
        }
    }

    let mut flagged = 0usize;
    for (file, allowed) in &by_file {
        let path = root.join(file.replace('\\', "/"));
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) => {
                // Not a skip. A registered op names this file as its source, so
                // a file the check cannot open is a file whose tier claims went
                // unread while the gate still reported zero.
                flagged += 1;
                report.find(Finding::in_file(
                    file,
                    format!("a registered operation names `{file}` as its source and it cannot be read: {error}"),
                    "restore the file the registration points at, or repoint the registration at the path it moved to",
                ));
                continue;
            }
        };
        for (line, claimed) in tier_claims(&text) {
            if allowed.contains(claimed.as_str()) {
                continue;
            }
            flagged += 1;
            report.find(Finding::in_file(
                file,
                format!(
                    "{file}:{line} claims Tier {claimed}, but its registered operations are Tier {}",
                    allowed.iter().copied().collect::<Vec<_>>().join(" / ")
                ),
                "delete the tier from the comment, or move the operation to the crate whose prefix carries that tier; `docs/lego-block-rule.md` owns the mapping",
            ));
        }
    }

    if flagged == 0 {
        report.note("  ✓ no source file contradicts its own tier".to_string());
    }
    flagged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_is_read_as_a_word_and_a_number() {
        let claims = tier_claims(
            "//! Tier 2.5 primitive.\n\
             //! A Frontier 3 is not a claim.\n\
             //! Tier-3 spelled with a hyphen.\n\
             //! Sentence ends at Tier 2.\n\
             //! Tier three is prose, not a number.\n",
        );
        assert_eq!(
            claims,
            vec![
                (1, "2.5".to_string()),
                (3, "3".to_string()),
                (4, "2".to_string()),
            ]
        );
    }

    /// WHY: `tier_of` derives the tier from the op id, so a file registering a
    /// `vyre-libs::` op is Tier 3 and any other number in it is false. The gate
    /// has to read the derived tier rather than a table, or moving a crate
    /// leaves the table right and the check wrong.
    #[test]
    fn every_tier_maps_to_the_prefix_that_derives_it() {
        assert_eq!(
            tier_number(tier_of("vyre-primitives::hardware::subgroup_add")),
            Some("2")
        );
        assert_eq!(
            tier_number(tier_of("vyre-primitives::scan::exclusive")),
            Some("2.5")
        );
        assert_eq!(tier_number(tier_of("vyre-libs::math::dot")), Some("3"));
        assert_eq!(tier_number(tier_of("vyre-foundation::whatever")), None);
    }
}
