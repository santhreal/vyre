//! WHY: a submission rejection's text is the only thing that tells a reader
//! which contract broke, and `InstanceMessages` lets each backend supply its
//! own. Nothing stopped two of the five rejections in one record from rendering
//! the same sentence, and one record did exactly that: a driver spelled an
//! unpreserved retained value with the neutral text for an unproduced output, so
//! for as long as that wording stood, a value the execution never preserved was
//! reported as a value it never produced. Diagnosis of a real device failure had
//! to start by proving which of the two classes the message meant.
//!
//! The contract pinned here is that within one record the five rejections read
//! as five distinct sentences, and that both value-shaped rejections name the
//! value they are about. The record is destructured rather than field-accessed,
//! so a sixth rejection added to `InstanceMessages` fails to compile here until
//! it is rendered and compared with the rest.
//!
//! Does not catch: a backend record this suite cannot name. A record declared
//! `const` inside a driver crate is private to it, so the same assertion has to
//! live beside that record; concrete driver crates carry their own copies for the
//! resident record for that reason. This suite covers the neutral record, which
//! is every backend that does not override one.

use vyre_driver::materialize::{InstanceMessages, NEUTRAL_MESSAGES};
use vyre_megakernel::ArtifactValueId;

/// Render every rejection in `messages` as (field, text) for one value.
fn rendered(messages: InstanceMessages, value: ArtifactValueId) -> Vec<(&'static str, String)> {
    let InstanceMessages {
        foreign_artifact,
        unmapped_buffer,
        missing_output_value,
        missing_retained_value,
        completion_consumed,
    } = messages;
    vec![
        ("foreign_artifact", foreign_artifact().to_string()),
        ("unmapped_buffer", unmapped_buffer("scratch").to_string()),
        (
            "missing_output_value",
            missing_output_value(value).to_string(),
        ),
        (
            "missing_retained_value",
            missing_retained_value(value).to_string(),
        ),
        ("completion_consumed", completion_consumed().to_string()),
    ]
}

#[test]
fn no_two_neutral_rejections_render_one_sentence() {
    let texts = rendered(NEUTRAL_MESSAGES, ArtifactValueId(7));
    let mut compared = 0usize;
    for (index, (left, left_text)) in texts.iter().enumerate() {
        for (right, right_text) in texts.iter().skip(index + 1) {
            assert_ne!(
                left_text, right_text,
                "Fix: rejections `{left}` and `{right}` render the same sentence, so a reader cannot tell which contract broke. Give each its own wording."
            );
            compared += 1;
        }
    }
    assert_eq!(
        compared,
        texts.len() * (texts.len() - 1) / 2,
        "Fix: every pair of rejections must be compared, not a representative one."
    );
}

#[test]
fn each_value_shaped_rejection_names_its_value() {
    let value = ArtifactValueId(4_294_967_295);
    let texts = rendered(NEUTRAL_MESSAGES, value);
    for field in ["missing_output_value", "missing_retained_value"] {
        let text = texts
            .iter()
            .find(|(name, _)| *name == field)
            .map(|(_, text)| text.as_str())
            .unwrap_or_else(|| panic!("Fix: `{field}` must be rendered by this suite"));
        assert!(
            text.contains(&value.0.to_string()),
            "Fix: `{field}` reads `{text}` and never names value {}, so the reader cannot find the resource that failed.",
            value.0
        );
    }
}

#[test]
fn the_output_and_retained_rejections_name_different_events() {
    let texts = rendered(NEUTRAL_MESSAGES, ArtifactValueId(1));
    let text_for = |field: &str| {
        texts
            .iter()
            .find(|(name, _)| *name == field)
            .map(|(_, text)| text.clone())
            .expect("Fix: this suite must render every rejection it asserts on")
    };
    let produced = text_for("missing_output_value");
    let preserved = text_for("missing_retained_value");
    assert!(
        produced.contains("produce"),
        "Fix: an unproduced output must say so; it reads `{produced}`."
    );
    assert!(
        preserved.contains("preserve"),
        "Fix: an unpreserved retained value must say so; it reads `{preserved}`."
    );
}
