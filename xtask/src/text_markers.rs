use std::collections::BTreeSet;

pub(crate) const NEGATIVE_MARKERS: &[&str] = &[
    "err",
    "error",
    "reject",
    "invalid",
    "unsupported",
    "fail",
    "panic",
    "malformed",
];

pub(crate) const BOUNDARY_MARKERS: &[&str] = &[
    "boundary",
    "overflow",
    "underflow",
    "zero",
    "empty",
    "limit",
    "cap",
    "max",
    "min",
];

pub(crate) fn contains_any(text: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| text.contains(marker))
}

pub(crate) fn markdown_line_is_release_rule_text(lowered: &str) -> bool {
    contains_any(
        lowered,
        &[
            "no-stub",
            "no shipped source",
            "must not",
            "not only",
            "not optional",
            "not a ",
            "no todo",
            "todo/fixme",
        ],
    )
}

pub(crate) fn classify_text<'a>(
    text: &str,
    rules: &'a [(&'static str, &[&str])],
) -> BTreeSet<&'static str> {
    rules
        .iter()
        .filter_map(|(label, markers)| contains_any(text, markers).then_some(*label))
        .collect()
}
