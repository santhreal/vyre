//! The changelog and the release notes body, generated from the release train
//! and the unreleased fragments.
//!
//! A fragment is one file under `release/changes/unreleased/`, named by its id,
//! carrying a category and a text. Fragments used to be tables in one shared
//! file; every fragment opened with the same `[[fragments]]` header, so a merge
//! aligned that header as common context and kept the keys of the second block
//! while dropping its header, which parsed as a duplicate key in the fragment
//! above and stopped every release document from regenerating. One file per
//! fragment cannot collide that way.
//!
//! The changelog `## [Unreleased]` section and `release/evidence/docs/release-notes-body.md`
//! are the same rendered section under two headings, so a release has one
//! authored source for what it says it contains.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::release::release_train::RELEASE_TRAIN_TOML_PATH;

/// Where the unreleased fragments live.
const FRAGMENTS: &str = "release/changes/unreleased";
/// The changelog whose unreleased section this gate owns.
const CHANGELOG: &str = "CHANGELOG.md";
/// The notes body `gh release create --notes-file` attaches to the final tag.
const NOTES: &str = "release/evidence/docs/release-notes-body.md";
/// The script that performs the launch steps in order.
const LAUNCH: &str = "scripts/final-launch.sh";
/// Keep-a-changelog categories, in the order a section renders them.
const CATEGORIES: [&str; 6] = [
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
];
/// Column the changelog wraps entry text at.
const WRAP_COLUMN: usize = 79;
/// Bound on any one document this gate reads.
const MAX_DOCUMENT_BYTES: u64 = 8_388_608;

/// The launch steps, in the order `final-launch.sh` must perform them.
///
/// Order is the contract: publishing before the candidate tag exists, or
/// recording the release before the gate that clears it, cannot be undone once
/// the crates are on crates.io.
const LAUNCH_STEPS: [&str; 10] = [
    "git tag -a \"$VYRE_RELEASE_TAG_VYRE_RC\"",
    "git push origin \"$VYRE_RELEASE_TAG_VYRE_RC\"",
    "-- vyre-release-gate\n",
    "VYRE_RELEASE_APPROVED=\"$VYRE_RELEASE_PUBLISH_APPROVAL_TOKEN\" bash scripts/publish-release.sh",
    "git tag -a \"$VYRE_RELEASE_TAG_VYRE\"",
    "git push origin \"$VYRE_RELEASE_TAG_VYRE\"",
    "gh release create \"$VYRE_RELEASE_TAG_VYRE\"",
    "> release/evidence/final/public-launch-completion.json",
    "-- launch-state --output",
    "-- vyre-release-gate --launch-complete\n",
];

/// The changelog and release notes state what the fragments and the train say.
pub struct ReleaseDocs;

impl Gate for ReleaseDocs {
    fn name(&self) -> &'static str {
        "release-docs"
    }

    fn help(&self) -> &'static str {
        "Hold the changelog and the release notes body to the release train and the unreleased fragments; --write regenerates both"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let train = read_toml(&ctx.root, RELEASE_TRAIN_TOML_PATH)?;
        let mut report = Report::clean();

        train_findings(&train, &mut report);
        let grouped = fragment_entries(&ctx.root, &mut report)?;
        if report.count() > 0 {
            return Ok(report);
        }

        let section = changelog_section(&train, &grouped)?;
        let changelog = read_text(&ctx.root, CHANGELOG)?;
        let rendered = match replace_unreleased(&changelog, &section) {
            Ok(text) => text,
            Err(message) => {
                report.find(Finding::in_file(
                    CHANGELOG,
                    message,
                    "restore the `## [Unreleased]` heading and the released section under it; \
                     the gate rewrites the span between them",
                ));
                return Ok(report);
            }
        };
        let notes = notes_body(&train, &section)?;

        if ctx.write {
            crate::generated_document::write(&ctx.root.join(CHANGELOG), &rendered)?;
            crate::generated_document::write(&ctx.root.join(NOTES), &notes)?;
            report.note("wrote the changelog and the release notes body".to_string());
            return Ok(report);
        }

        for (path, expected) in [(CHANGELOG, &rendered), (NOTES, &notes)] {
            if read_text(&ctx.root, path)?.as_str() != expected.as_str() {
                report.find(Finding::in_file(
                    path,
                    "the generated release content disagrees with the fragments and the train",
                    "regenerate it with `cargo_full run --bin xtask -- release-docs --write`; \
                     never hand-edit a generated release document",
                ));
            }
        }
        for token in required_tokens(&train) {
            if !changelog.contains(&token) {
                report.find(Finding::in_file(
                    CHANGELOG,
                    format!(
                        "the release train requires the token `{token}`, which the changelog does not carry"
                    ),
                    "state the fact the token names in a fragment, or drop the token from \
                     `required_release_note_tokens`",
                ));
            }
        }
        launch_order_findings(&ctx.root, &mut report)?;
        report.note(format!(
            "{} fragment(s) across {} category(ies)",
            grouped.values().map(Vec::len).sum::<usize>(),
            grouped.len()
        ));
        Ok(report)
    }
}

/// Findings against the release train itself.
///
/// A train that names a version key no group carries, or a package two groups
/// claim, publishes something under a version nobody declared.
fn train_findings(train: &toml::Table, report: &mut Report) {
    let versions = train.get("versions").and_then(toml::Value::as_table);
    if versions.is_none_or(toml::Table::is_empty) {
        report.find(Finding::in_file(
            RELEASE_TRAIN_TOML_PATH,
            "[versions] declares no version",
            "declare the version each release group ships under",
        ));
    }
    let groups = train.get("release_groups").and_then(toml::Value::as_table);
    let Some(groups) = groups.filter(|table| !table.is_empty()) else {
        report.find(Finding::in_file(
            RELEASE_TRAIN_TOML_PATH,
            "[release_groups] declares no group",
            "declare each release group with its repository, version key and packages",
        ));
        return;
    };

    let mut owner: BTreeMap<&str, &str> = BTreeMap::new();
    for (name, group) in groups {
        let group = group.as_table();
        let repository = group
            .and_then(|table| table.get("repository"))
            .and_then(toml::Value::as_str);
        if !repository.is_some_and(|value| value.matches('/').count() == 1) {
            report.find(Finding::in_file(
                RELEASE_TRAIN_TOML_PATH,
                format!("release group `{name}` declares no owner/repository"),
                "name the repository the group publishes from, as `owner/repository`",
            ));
        }
        let version_key = group
            .and_then(|table| table.get("version"))
            .and_then(toml::Value::as_str);
        let known = version_key.is_some_and(|key| {
            versions.is_some_and(|table| table.contains_key(key))
        });
        if !known {
            report.find(Finding::in_file(
                RELEASE_TRAIN_TOML_PATH,
                format!(
                    "release group `{name}` references the version key `{}`, which [versions] does not declare",
                    version_key.unwrap_or("<missing>")
                ),
                "point the group at a declared version key, or declare that version",
            ));
        }
        let packages = group
            .and_then(|table| table.get("packages"))
            .and_then(toml::Value::as_array);
        let Some(packages) = packages.filter(|list| !list.is_empty()) else {
            report.find(Finding::in_file(
                RELEASE_TRAIN_TOML_PATH,
                format!("release group `{name}` declares no package"),
                "list the packages the group publishes",
            ));
            continue;
        };
        for package in packages {
            let Some(package) = package.as_str().filter(|value| !value.is_empty()) else {
                report.find(Finding::in_file(
                    RELEASE_TRAIN_TOML_PATH,
                    format!("release group `{name}` lists a package that is not a name"),
                    "list each package as its crate name",
                ));
                continue;
            };
            if let Some(previous) = owner.insert(package, name) {
                if previous != name {
                    report.find(Finding::in_file(
                        RELEASE_TRAIN_TOML_PATH,
                        format!("package `{package}` belongs to both `{previous}` and `{name}`"),
                        "give the package one release group; two groups publish it twice, \
                         under two versions",
                    ));
                }
            }
        }
    }

    let actions = train
        .get("external_actions")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut ids: Vec<&str> = actions
        .iter()
        .filter_map(|action| action.as_table()?.get("id")?.as_str())
        .filter(|id| !id.is_empty())
        .collect();
    ids.sort_unstable();
    let unique = ids.len();
    ids.dedup();
    if ids.len() != unique || ids.len() != actions.len() {
        report.find(Finding::in_file(
            RELEASE_TRAIN_TOML_PATH,
            "an approval-gated external action has no id, or two share one",
            "give every [[external_actions]] entry a unique id; the launch record \
             cites an action by id",
        ));
    }
    let required = crate::release::launch_contract::required_external_actions().len();
    if actions.len() != required {
        report.find(Finding::in_file(
            RELEASE_TRAIN_TOML_PATH,
            format!(
                "the train declares {} approval-gated external action(s) and the launch \
                 contract needs {required}",
                actions.len()
            ),
            "declare one [[external_actions]] entry per action the launch performs outside \
             this repository; the count is the launch contract's, not a number this gate \
             restates",
        ));
    }
}

/// Entry texts by category, read one fragment per file.
///
/// The directory is listed rather than the git index: a fragment is written
/// beside the change it describes and the documents regenerate before it is
/// staged, so reading only tracked files would silently drop the newest one.
fn fragment_entries(
    root: &Path,
    report: &mut Report,
) -> Result<BTreeMap<String, Vec<String>>, GateError> {
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let directory = root.join(FRAGMENTS);
    let listing = fs::read_dir(&directory).map_err(|error| {
        GateError::new(
            format!("could not list `{FRAGMENTS}`: {error}"),
            "create the fragment directory; a release describes itself from the fragments \
             in it",
        )
    })?;
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in listing {
        let path = entry
            .map_err(|error| {
                GateError::new(
                    format!("could not read an entry of `{FRAGMENTS}`: {error}"),
                    "check the checkout is readable",
                )
            })?
            .path();
        if path.extension().is_some_and(|extension| extension == "toml") {
            files.push(path);
        }
    }
    files.sort_unstable();
    if files.is_empty() {
        report.find(Finding::in_file(
            FRAGMENTS,
            "no unreleased fragment is on disk",
            "write one fragment per change, as `release/changes/unreleased/<id>.toml` \
             carrying `category` and `text`",
        ));
        return Ok(grouped);
    }

    for path in &files {
        let where_ = path
            .strip_prefix(root)
            .unwrap_or(path.as_path())
            .display()
            .to_string();
        let fragment = read_toml(root, &where_)?;
        let unexpected: Vec<&str> = fragment
            .keys()
            .map(String::as_str)
            .filter(|key| *key != "category" && *key != "text")
            .collect();
        if !unexpected.is_empty() {
            report.find(Finding::in_file(
                where_.clone(),
                format!("unexpected key(s) {}", unexpected.join(", ")),
                "a fragment carries `category` and `text`, and its id is the file name",
            ));
            continue;
        }
        let category = fragment
            .get("category")
            .and_then(toml::Value::as_str)
            .unwrap_or_default();
        if !CATEGORIES.contains(&category) {
            report.find(Finding::in_file(
                where_.clone(),
                format!("`{category}` is not a changelog category"),
                format!("use one of {}", CATEGORIES.join(", ")),
            ));
            continue;
        }
        let text = normalize(
            fragment
                .get("text")
                .and_then(toml::Value::as_str)
                .unwrap_or_default(),
        );
        if text.is_empty() {
            report.find(Finding::in_file(
                where_.clone(),
                "text is empty",
                "state what changed, in one paragraph a reader of the changelog can act on",
            ));
            continue;
        }
        if let Some(previous) = seen.insert(text.clone(), where_.clone()) {
            report.find(Finding::in_file(
                where_,
                format!("text repeats the one in {previous}"),
                "one change is one fragment; delete the duplicate",
            ));
            continue;
        }
        grouped.entry(category.to_string()).or_default().push(text);
    }
    Ok(grouped)
}

/// Collapse every run of whitespace to one space.
fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The tokens the train requires release prose to carry.
fn required_tokens(train: &toml::Table) -> Vec<String> {
    train
        .get("required_release_note_tokens")
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|token| token.as_str().map(str::to_string))
        .collect()
}

/// The rendered `## [Unreleased]` section.
fn changelog_section(
    train: &toml::Table,
    grouped: &BTreeMap<String, Vec<String>>,
) -> Result<String, GateError> {
    let mut lines = vec!["## [Unreleased]".to_string(), String::new()];
    lines.extend(train_identities(train)?);
    for category in CATEGORIES {
        let Some(entries) = grouped.get(category) else {
            continue;
        };
        lines.push(String::new());
        lines.push(format!("### {category}"));
        lines.push(String::new());
        for text in entries {
            lines.extend(wrap(text));
        }
    }
    Ok(lines.join("\n") + "\n")
}

/// The artifact identities a release has to name, stated from the train.
///
/// These used to live in a hand-maintained per-version notes file that could
/// disagree with the train. Generating them means the requirement is met by
/// construction instead of checked after the fact.
fn train_identities(train: &toml::Table) -> Result<Vec<String>, GateError> {
    let field = |table: &str, key: &str| -> Result<String, GateError> {
        train
            .get(table)
            .and_then(toml::Value::as_table)
            .and_then(|section| section.get(key))
            .and_then(toml::Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                GateError::new(
                    format!("{RELEASE_TRAIN_TOML_PATH} declares no {table}.{key}"),
                    "declare the version and both tags the release ships under",
                )
            })
    };
    let mut lines = vec![format!(
        "Vyre {} releases from candidate tag `{}` and final tag `{}`.",
        field("versions", "vyre")?,
        field("tags", "vyre_rc")?,
        field("tags", "vyre")?
    )];
    let pinned: Vec<String> = required_tokens(train)
        .into_iter()
        .filter(|token| token.contains('@'))
        .map(|token| format!("`{token}`"))
        .collect();
    if !pinned.is_empty() {
        lines.push(format!(
            "Backend crates carried at that version: {}.",
            pinned.join(", ")
        ));
    }
    Ok(lines)
}

/// One entry wrapped to the changelog column, as a bullet with hanging indent.
fn wrap(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = "- ".to_string();
    let mut empty = true;
    for word in text.split_whitespace() {
        if !empty && current.chars().count() + 1 + word.chars().count() > WRAP_COLUMN {
            lines.push(current);
            current = "  ".to_string();
            empty = true;
        }
        if !empty {
            current.push(' ');
        }
        current.push_str(word);
        empty = false;
    }
    if !empty {
        lines.push(current);
    }
    lines
}

/// The changelog with its unreleased section replaced.
fn replace_unreleased(changelog: &str, section: &str) -> Result<String, String> {
    let start = changelog
        .find("## [Unreleased]")
        .ok_or_else(|| "the changelog carries no `## [Unreleased]` section".to_string())?;
    let after = start + "## [Unreleased]".len();
    let end = changelog[after..]
        .find("\n## [")
        .map(|offset| after + offset)
        .ok_or_else(|| "the changelog carries no released section after `## [Unreleased]`".to_string())?;
    Ok(format!(
        "{}{}\n{}",
        &changelog[..start],
        section.trim_end(),
        &changelog[end..]
    ))
}

/// The same section under the final tag instead of `Unreleased`.
fn notes_body(train: &toml::Table, section: &str) -> Result<String, GateError> {
    let tag = train
        .get("tags")
        .and_then(toml::Value::as_table)
        .and_then(|tags| tags.get("vyre"))
        .and_then(toml::Value::as_str)
        .ok_or_else(|| {
            GateError::new(
                format!("{RELEASE_TRAIN_TOML_PATH} declares no tags.vyre"),
                "declare the final tag the release notes are attached to",
            )
        })?;
    Ok(section.replacen("## [Unreleased]", &format!("# {tag}"), 1))
}

/// Findings against the order the launch script performs its steps in.
fn launch_order_findings(root: &Path, report: &mut Report) -> Result<(), GateError> {
    let launch = read_text(root, LAUNCH)?;
    let mut positions = Vec::with_capacity(LAUNCH_STEPS.len());
    for step in LAUNCH_STEPS {
        match launch.find(step) {
            Some(at) => positions.push(at),
            None => report.find(Finding::in_file(
                LAUNCH,
                format!("the guarded launch step `{}` is missing", step.trim_end()),
                "restore the step; the launch performs candidate tags, the prepublication \
                 gate, publish, final tags, the release record, completion evidence and the \
                 final gate, in that order",
            )),
        }
    }
    if positions.len() == LAUNCH_STEPS.len() && !positions.is_sorted() {
        report.find(Finding::in_file(
            LAUNCH,
            "the guarded launch steps are out of order",
            "publish after the candidate tag and the prepublication gate, and record the \
             release after the publish; the order cannot be undone once crates are public",
        ));
    }
    Ok(())
}

/// Read one file under a bound no release document approaches.
fn read_text(root: &Path, relative: &str) -> Result<String, GateError> {
    let path = root.join(relative);
    let size = fs::metadata(&path)
        .map_err(|error| {
            GateError::new(
                format!("could not read `{relative}`: {error}"),
                "restore the file the release documents are generated from",
            )
        })?
        .len();
    if size > MAX_DOCUMENT_BYTES {
        return Err(GateError::new(
            format!("`{relative}` is {size} bytes, over the {MAX_DOCUMENT_BYTES} byte bound"),
            "a release document this large is not a release document; check what wrote it",
        ));
    }
    fs::read_to_string(&path).map_err(|error| {
        GateError::new(
            format!("could not read `{relative}`: {error}"),
            "restore the file the release documents are generated from",
        )
    })
}

/// Read and parse one TOML file.
fn read_toml(root: &Path, relative: &str) -> Result<toml::Table, GateError> {
    let text = read_text(root, relative)?;
    toml::from_str::<toml::Table>(&text).map_err(|error| {
        GateError::new(
            format!("`{relative}` does not parse as TOML: {error}"),
            "fix the syntax the parser names; a release cannot describe itself from a \
             document that does not parse",
        )
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A fragment longer than the wrap column becomes a bullet with a hanging
    /// indent, and never a line the column cannot hold.
    #[test]
    fn wrapping_hangs_the_continuation_and_holds_the_column() {
        let text = "one two three four five six seven eight nine ten eleven twelve \
                    thirteen fourteen fifteen sixteen seventeen eighteen";
        let lines = wrap(text);
        assert!(lines.len() > 1, "the text is longer than one line");
        assert!(lines[0].starts_with("- "));
        for line in &lines[1..] {
            assert!(line.starts_with("  "), "continuation lines hang: {line}");
        }
        for line in &lines {
            assert!(
                line.chars().count() <= WRAP_COLUMN,
                "line over the column: {line}"
            );
        }
        let joined = lines.join(" ").replace("- ", "").replace("  ", " ");
        assert_eq!(normalize(&joined), normalize(text));
    }

    /// One word longer than the column stays on its own line rather than being
    /// broken: a URL or a path split across lines stops resolving.
    #[test]
    fn a_word_over_the_column_is_not_broken() {
        let long = "a".repeat(WRAP_COLUMN + 10);
        assert_eq!(wrap(&long), vec![format!("- {long}")]);
    }

    /// The replaced span ends at the first released section, so history below
    /// it survives regeneration.
    #[test]
    fn replacing_the_unreleased_section_keeps_the_released_history() {
        let changelog = "# Changelog\n\n## [Unreleased]\n\nold\n\n## [0.1.0]\n\nkept\n";
        let replaced =
            replace_unreleased(changelog, "## [Unreleased]\n\nnew\n").expect("a section to replace");
        assert_eq!(
            replaced,
            "# Changelog\n\n## [Unreleased]\n\nnew\n\n## [0.1.0]\n\nkept\n"
        );
    }

    /// A changelog with no released section is refused rather than truncated.
    #[test]
    fn a_changelog_with_no_released_section_is_refused() {
        let error = replace_unreleased("## [Unreleased]\n\nonly\n", "## [Unreleased]\n")
            .expect_err("no released section");
        assert!(error.contains("released section"), "{error}");
    }
}
