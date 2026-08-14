//! Capture-mode contract rows and the mode lookup used by detector routing.

use super::{CaptureMode, CaptureModeContract};

impl CaptureMode {
    /// Every mode, in contract order. One owner for iteration + coherence checks.
    pub const ALL: [CaptureMode; 6] = [
        CaptureMode::NonCapture,
        CaptureMode::Count,
        CaptureMode::Span,
        CaptureMode::NamedCapture,
        CaptureMode::RepeatedCapture,
        CaptureMode::GroupExtraction,
    ];

    /// The contract row for this mode, the single code-side source of truth for
    /// its `mode_id`, `output_shape`, routing bits, and null policy.
    #[must_use]
    pub const fn contract_row(self) -> CaptureModeContract {
        match self {
            CaptureMode::NonCapture => CaptureModeContract {
                mode_id: "noncapture",
                output_shape: "whole_match_only",
                accelerator_eligible: true,
                verifier_required: false,
                null_policy: "not_applicable",
            },
            CaptureMode::Count => CaptureModeContract {
                mode_id: "count",
                output_shape: "match_count_per_pattern",
                accelerator_eligible: true,
                verifier_required: false,
                null_policy: "not_applicable",
            },
            CaptureMode::Span => CaptureModeContract {
                mode_id: "span",
                output_shape: "whole_match_span",
                accelerator_eligible: true,
                verifier_required: false,
                null_policy: "absent-match-has-no-span",
            },
            CaptureMode::NamedCapture => CaptureModeContract {
                mode_id: "named_capture",
                output_shape: "named_group_span_records",
                accelerator_eligible: false,
                verifier_required: true,
                null_policy: "unmatched-group-null",
            },
            CaptureMode::RepeatedCapture => CaptureModeContract {
                mode_id: "repeated_capture",
                output_shape: "ordered_group_span_list",
                accelerator_eligible: false,
                verifier_required: true,
                null_policy: "empty-repeat-yields-empty-list",
            },
            CaptureMode::GroupExtraction => CaptureModeContract {
                mode_id: "group_extraction",
                output_shape: "row_group_value_table",
                accelerator_eligible: false,
                verifier_required: true,
                null_policy: "unmatched-group-null",
            },
        }
    }

    /// Whether the GPU accelerator path can serve this mode directly (no
    /// verifier). The three whole-match modes are eligible; group-extraction is not.
    #[must_use]
    pub const fn accelerator_eligible(self) -> bool {
        self.contract_row().accelerator_eligible
    }

    /// Whether a scalar (CPU-semantics) verifier must own this mode's output.
    /// The exact complement of [`accelerator_eligible`](Self::accelerator_eligible)
    /// under this contract, but named separately because the two are independent
    /// contract fields, a future mode could be neither (unsupported) rather than
    /// exactly one.
    #[must_use]
    pub const fn verifier_required(self) -> bool {
        self.contract_row().verifier_required
    }

    /// Look up a mode by its stable `mode_id` string (the reverse of
    /// `contract_row().mode_id`), for consumers that receive the mode as config
    /// text. Returns `None` for an unknown id.
    #[must_use]
    pub fn from_mode_id(mode_id: &str) -> Option<CaptureMode> {
        CaptureMode::ALL
            .into_iter()
            .find(|mode| mode.contract_row().mode_id == mode_id)
    }
}

#[cfg(test)]
mod tests {
    use super::super::CaptureMode;

    #[test]
    fn capture_mode_routing_splits_accelerator_from_verifier() {
        // Exactly the three whole-match modes run on the accelerator; the three
        // group-extraction modes require the verifier. Under this contract the
        // two bits are exact complements, assert both directions so a future
        // "neither" (unsupported) mode can't slip through as accelerator-eligible.
        for mode in CaptureMode::ALL {
            assert_eq!(
                mode.accelerator_eligible(),
                !mode.verifier_required(),
                "{mode:?}: accelerator_eligible must be the complement of verifier_required"
            );
        }
        let accel: Vec<CaptureMode> = CaptureMode::ALL
            .into_iter()
            .filter(|m| m.accelerator_eligible())
            .collect();
        assert_eq!(
            accel,
            vec![
                CaptureMode::NonCapture,
                CaptureMode::Count,
                CaptureMode::Span
            ],
            "only the whole-match modes are accelerator-eligible"
        );
    }

    #[test]
    fn capture_mode_id_round_trips_and_is_unique() {
        use std::collections::BTreeSet;
        let mut ids = BTreeSet::new();
        for mode in CaptureMode::ALL {
            let id = mode.contract_row().mode_id;
            assert!(ids.insert(id), "duplicate mode_id `{id}`");
            assert_eq!(
                CaptureMode::from_mode_id(id),
                Some(mode),
                "mode_id `{id}` must round-trip back to {mode:?}"
            );
        }
        assert_eq!(ids.len(), 6, "all six modes must have distinct ids");
        assert_eq!(CaptureMode::from_mode_id("no_such_mode"), None);
    }
}
