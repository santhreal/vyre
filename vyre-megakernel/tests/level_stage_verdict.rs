//! The target-payload level accepts its own attachment and rejects a wrong one.
//!
//! WHY this suite exists: the level stage this crate registers is what the level
//! registry calls to verify a target payload, and a stage that answers
//! `Verified` for every attachment reads exactly like one that checks the
//! association. A payload is authenticated over its own bytes at construction,
//! so the fact a caller can still get wrong is which neutral artifact the
//! payload implements: an attachment states that pairing and this suite holds
//! the stage to it.
//!
//! What this does NOT catch: a payload whose framing digest is stale. Every
//! constructor authenticates, and `TargetPayload::from_bytes` refuses a
//! tampered encoding, so no such payload can be handed to the stage from here.

use vyre_foundation::optimizer::level_contract::{stage_for_level, LevelVerdict};
use vyre_megakernel::{
    Artifact, PayloadAttachment, TargetPayload, TargetPayloadFormat, TargetProfile,
};
use vyre_spec::IrLevel;

#[path = "../../tests/support/artifact_fixtures.rs"]
mod artifact_fixtures;

use artifact_fixtures::{entry_point, neutral_artifact};

fn format() -> TargetPayloadFormat {
    TargetPayloadFormat::new("test.target-binary", 1)
        .expect("Fix: the fixture format must be valid")
}

fn profile() -> TargetProfile {
    TargetProfile::new("test.target-binary", 1, [64, 1, 1], 64, 1_024, 0)
        .expect("Fix: the fixture profile must be valid")
}

fn payload_for(artifact: &Artifact) -> TargetPayload {
    TargetPayload::new(
        artifact,
        format(),
        profile(),
        vec![entry_point(artifact)],
        vec![4, 2],
    )
    .expect("Fix: the fixture payload must seal")
}

/// An attachment naming the artifact the payload implements verifies; one
/// naming another artifact does not.
#[test]
fn target_payload_stage_rejects_an_attachment_to_another_artifact() {
    let stage =
        stage_for_level(IrLevel::TargetPayload).expect("Fix: the target-payload stage must exist");

    let neutral = neutral_artifact([8, 1, 1]);
    let other = neutral_artifact([16, 1, 1]);
    assert_ne!(
        neutral.digest(),
        other.digest(),
        "Fix: the two fixture artifacts must differ for this case to mean anything"
    );

    let matched = PayloadAttachment {
        neutral: neutral.digest(),
        payload: payload_for(&neutral),
    };
    assert_eq!(
        stage.verify(&matched),
        LevelVerdict::Verified,
        "Fix: a payload attached to the artifact it implements must verify"
    );
    assert_eq!(stage.is_canonical(&matched), LevelVerdict::Verified);

    let crossed = PayloadAttachment {
        neutral: other.digest(),
        payload: payload_for(&neutral),
    };
    let verdict = stage.verify(&crossed);
    assert!(
        matches!(verdict, LevelVerdict::Rejected(_)),
        "Fix: a payload attached to an artifact it does not implement must be rejected, got \
         {verdict:?}"
    );
}

/// The stage refuses a subject of another level rather than verifying it.
#[test]
fn target_payload_stage_refuses_another_levels_subject() {
    let stage =
        stage_for_level(IrLevel::TargetPayload).expect("Fix: the target-payload stage must exist");
    let neutral = neutral_artifact([8, 1, 1]);
    assert_eq!(
        stage.verify(&payload_for(&neutral)),
        LevelVerdict::WrongSubject {
            expected: "PayloadAttachment"
        },
        "Fix: a bare payload is not the level's subject; the attachment states the artifact"
    );
}
