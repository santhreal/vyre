//! The target-payload level's verifier, canonical form, and analyses.
//!
//! A payload authenticates its own bytes at construction, so a payload that
//! exists already carries a digest over its body. What a caller supplies
//! separately, and can supply wrongly, is the neutral artifact the payload is
//! claimed to implement: an attachment pairs the two, and verification is the
//! association check plus the payload's own framing round trip.

use std::any::Any;

use vyre_foundation::optimizer::level_contract::{
    LevelAnalysis, LevelStage, LevelStageRegistration, LevelVerdict,
};
use vyre_spec::IrLevel;

use crate::envelope::{TargetPayload, TARGET_PAYLOAD_SCHEMA_VERSION};
use crate::Digest;

/// The target-payload level's subject: a payload and the neutral artifact it is
/// claimed to implement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PayloadAttachment {
    /// Identity of the neutral artifact this payload must implement.
    pub neutral: Digest,
    /// The authenticated payload.
    pub payload: TargetPayload,
}

/// Target-payload stage: the emitted module for one backend.
struct TargetPayloadStage;

impl LevelStage for TargetPayloadStage {
    fn level(&self) -> IrLevel {
        IrLevel::TargetPayload
    }

    fn subject(&self) -> &'static str {
        "PayloadAttachment"
    }

    fn verify(&self, subject: &dyn Any) -> LevelVerdict {
        let Some(attachment) = subject.downcast_ref::<PayloadAttachment>() else {
            return LevelVerdict::WrongSubject {
                expected: "PayloadAttachment",
            };
        };
        let payload = &attachment.payload;
        let mut rejections = Vec::new();
        if payload.neutral_artifact() != attachment.neutral {
            rejections.push(format!(
                "payload implements {:?}, not the attached {:?}",
                payload.neutral_artifact(),
                attachment.neutral
            ));
        }
        if payload.schema_version() != TARGET_PAYLOAD_SCHEMA_VERSION {
            rejections.push(format!(
                "payload schema {} is not {TARGET_PAYLOAD_SCHEMA_VERSION}",
                payload.schema_version()
            ));
        }
        match payload
            .to_bytes()
            .and_then(|bytes| TargetPayload::from_bytes(&bytes).map(|decoded| decoded.digest()))
        {
            Ok(digest) if digest == payload.digest() => {}
            Ok(_) => rejections.push("payload digest does not cover its own framing".to_string()),
            Err(error) => rejections.push(error.to_string()),
        }
        if rejections.is_empty() {
            LevelVerdict::Verified
        } else {
            LevelVerdict::Rejected(rejections)
        }
    }

    fn is_canonical(&self, subject: &dyn Any) -> LevelVerdict {
        let Some(attachment) = subject.downcast_ref::<PayloadAttachment>() else {
            return LevelVerdict::WrongSubject {
                expected: "PayloadAttachment",
            };
        };
        let entries = attachment.payload.entries();
        if entries.windows(2).any(|pair| pair[0].name > pair[1].name) {
            return LevelVerdict::rejected("payload entries are not ordered by name");
        }
        for entry in entries {
            if entry
                .resource_bindings
                .windows(2)
                .any(|pair| (pair[0].group, pair[0].slot) > (pair[1].group, pair[1].slot))
            {
                return LevelVerdict::rejected(format!(
                    "entry `{}` bindings are not ordered by group and slot",
                    entry.name
                ));
            }
        }
        LevelVerdict::Verified
    }

    fn analyses(&self) -> &'static [LevelAnalysis] {
        &[LevelAnalysis {
            name: "payload_entry_bindings",
            invalidated_by_rewrite: false,
        }]
    }
}

inventory::submit! {
    LevelStageRegistration { stage: &TargetPayloadStage }
}

/// The level this crate registers a stage for.
///
/// A real function rather than a constant: a reader links this crate by calling
/// it, and a constant would inline at the call site and link nothing, leaving
/// the target-payload level with no registered stage in that binary.
#[must_use]
pub fn registered_level_stage() -> IrLevel {
    TargetPayloadStage.level()
}
