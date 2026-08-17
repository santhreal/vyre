//! Unit test adapter delegating to [`vyre_reference::composition_witness::evaluate_formula_witness`].

use super::ast::{RuleCondition, RuleFormula};
use vyre_reference::composition_witness::{
    RuleConditionWitness, RuleEvaluationContextWitness, RuleFormulaWitness,
};

/// Evaluation context the reference evaluator queries when resolving each condition variant.
pub trait RuleEvaluationContext {
    /// Number of times pattern `pattern_id` matched in the current record.
    fn pattern_count(&self, _pattern_id: u32) -> u32 {
        0
    }

    /// File size in bytes for the current record.
    fn file_size(&self) -> u64 {
        0
    }

    /// Resolve a named field value.
    fn field_value(&self, _name: &str) -> Option<&str> {
        None
    }
}

impl From<&RuleCondition> for RuleConditionWitness {
    fn from(cond: &RuleCondition) -> Self {
        match cond {
            RuleCondition::PatternExists { pattern_id } => Self::PatternExists {
                pattern_id: *pattern_id,
            },
            RuleCondition::PatternCountGt {
                pattern_id,
                threshold,
            } => Self::PatternCountGt {
                pattern_id: *pattern_id,
                threshold: *threshold,
            },
            RuleCondition::PatternCountGte {
                pattern_id,
                threshold,
            } => Self::PatternCountGte {
                pattern_id: *pattern_id,
                threshold: *threshold,
            },
            RuleCondition::FileSizeLt(t) => Self::FileSizeLt(*t),
            RuleCondition::FileSizeLte(t) => Self::FileSizeLte(*t),
            RuleCondition::FileSizeGt(t) => Self::FileSizeGt(*t),
            RuleCondition::FileSizeGte(t) => Self::FileSizeGte(*t),
            RuleCondition::FileSizeEq(t) => Self::FileSizeEq(*t),
            RuleCondition::FileSizeNe(t) => Self::FileSizeNe(*t),
            RuleCondition::LiteralTrue => Self::LiteralTrue,
            RuleCondition::LiteralFalse => Self::LiteralFalse,
            RuleCondition::RegexMatch { field, pattern } => Self::RegexMatch {
                field: field.clone(),
                pattern: pattern.clone(),
            },
            RuleCondition::SubstringMatch { haystack, needle } => Self::SubstringMatch {
                haystack: haystack.clone(),
                needle: needle.clone(),
            },
            RuleCondition::PrefixMatch { value, prefix } => Self::PrefixMatch {
                value: value.clone(),
                prefix: prefix.clone(),
            },
            RuleCondition::SuffixMatch { value, suffix } => Self::SuffixMatch {
                value: value.clone(),
                suffix: suffix.clone(),
            },
            RuleCondition::RangeMatch { value, min, max } => Self::RangeMatch {
                value: *value,
                min: *min,
                max: *max,
            },
            RuleCondition::SetMembership { value, set } => Self::SetMembership {
                value: value.clone(),
                set: set.clone(),
            },
            RuleCondition::FieldInSet { field, set } => Self::FieldInSet {
                field: field.clone(),
                set: set.clone(),
            },
            RuleCondition::Opaque(ext) => Self::Opaque(ext.clone()),
        }
    }
}

impl From<&RuleFormula> for RuleFormulaWitness {
    fn from(formula: &RuleFormula) -> Self {
        match formula {
            RuleFormula::Condition(cond) => Self::Condition(RuleConditionWitness::from(cond)),
            RuleFormula::And(left, right) => Self::And(
                Box::new(RuleFormulaWitness::from(left.as_ref())),
                Box::new(RuleFormulaWitness::from(right.as_ref())),
            ),
            RuleFormula::Or(left, right) => Self::Or(
                Box::new(RuleFormulaWitness::from(left.as_ref())),
                Box::new(RuleFormulaWitness::from(right.as_ref())),
            ),
            RuleFormula::Not(inner) => {
                Self::Not(Box::new(RuleFormulaWitness::from(inner.as_ref())))
            }
        }
    }
}

struct ContextAdapter<'a, C: RuleEvaluationContext + ?Sized>(&'a C);

impl<'a, C: RuleEvaluationContext + ?Sized> RuleEvaluationContextWitness for ContextAdapter<'a, C> {
    fn pattern_count(&self, pattern_id: u32) -> u32 {
        self.0.pattern_count(pattern_id)
    }

    fn file_size(&self) -> u64 {
        self.0.file_size()
    }

    fn field_value(&self, name: &str) -> Option<&str> {
        self.0.field_value(name)
    }
}

/// Evaluate a [`RuleFormula`] against `ctx` by delegating to the canonical reference witness.
#[must_use]
pub fn evaluate_formula<C: RuleEvaluationContext + ?Sized>(formula: &RuleFormula, ctx: &C) -> bool {
    let witness = RuleFormulaWitness::from(formula);
    let adapter = ContextAdapter(ctx);
    vyre_reference::composition_witness::evaluate_formula_witness(&witness, &adapter)
}

/// Evaluate a single [`RuleCondition`] against `ctx` by delegating to the canonical reference witness.
#[must_use]
pub fn evaluate_condition<C: RuleEvaluationContext + ?Sized>(
    condition: &RuleCondition,
    ctx: &C,
) -> bool {
    let witness = RuleConditionWitness::from(condition);
    let adapter = ContextAdapter(ctx);
    vyre_reference::composition_witness::evaluate_condition_witness(&witness, &adapter)
}
#[cfg(test)]
mod tests {
    use super::*;

    struct StaticCtx<'a> {
        counts: &'a [(u32, u32)],
        size: u64,
        fields: &'a [(&'a str, &'a str)],
    }

    impl<'a> RuleEvaluationContext for StaticCtx<'a> {
        fn pattern_count(&self, pid: u32) -> u32 {
            self.counts
                .iter()
                .find(|(p, _)| *p == pid)
                .map(|(_, c)| *c)
                .unwrap_or(0)
        }
        fn file_size(&self) -> u64 {
            self.size
        }
        fn field_value(&self, name: &str) -> Option<&str> {
            self.fields
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| *v)
        }
    }

    fn empty_ctx() -> StaticCtx<'static> {
        StaticCtx {
            counts: &[],
            size: 0,
            fields: &[],
        }
    }

    #[test]
    fn literal_true_and_false() {
        assert!(evaluate_condition(
            &RuleCondition::LiteralTrue,
            &empty_ctx()
        ));
        assert!(!evaluate_condition(
            &RuleCondition::LiteralFalse,
            &empty_ctx()
        ));
    }

    #[test]
    fn pattern_exists_uses_count() {
        let ctx = StaticCtx {
            counts: &[(7, 3)],
            size: 0,
            fields: &[],
        };
        assert!(evaluate_condition(
            &RuleCondition::PatternExists { pattern_id: 7 },
            &ctx
        ));
        assert!(!evaluate_condition(
            &RuleCondition::PatternExists { pattern_id: 8 },
            &ctx
        ));
    }

    #[test]
    fn pattern_count_gt_gte() {
        let ctx = StaticCtx {
            counts: &[(1, 5)],
            size: 0,
            fields: &[],
        };
        assert!(evaluate_condition(
            &RuleCondition::PatternCountGt {
                pattern_id: 1,
                threshold: 4,
            },
            &ctx
        ));
        assert!(!evaluate_condition(
            &RuleCondition::PatternCountGt {
                pattern_id: 1,
                threshold: 5,
            },
            &ctx
        ));
        assert!(evaluate_condition(
            &RuleCondition::PatternCountGte {
                pattern_id: 1,
                threshold: 5,
            },
            &ctx
        ));
        assert!(!evaluate_condition(
            &RuleCondition::PatternCountGte {
                pattern_id: 1,
                threshold: 6,
            },
            &ctx
        ));
    }

    #[test]
    fn file_size_predicates() {
        let ctx = StaticCtx {
            counts: &[],
            size: 100,
            fields: &[],
        };
        assert!(evaluate_condition(&RuleCondition::FileSizeLt(101), &ctx));
        assert!(!evaluate_condition(&RuleCondition::FileSizeLt(100), &ctx));
        assert!(evaluate_condition(&RuleCondition::FileSizeLte(100), &ctx));
        assert!(evaluate_condition(&RuleCondition::FileSizeGt(99), &ctx));
        assert!(evaluate_condition(&RuleCondition::FileSizeGte(100), &ctx));
        assert!(evaluate_condition(&RuleCondition::FileSizeEq(100), &ctx));
        assert!(evaluate_condition(&RuleCondition::FileSizeNe(99), &ctx));
        assert!(!evaluate_condition(&RuleCondition::FileSizeNe(100), &ctx));
    }

    #[test]
    fn substring_prefix_suffix() {
        let ctx = StaticCtx {
            counts: &[],
            size: 0,
            fields: &[("path", "src/foo/bar.rs")],
        };
        assert!(evaluate_condition(
            &RuleCondition::SubstringMatch {
                haystack: "path".into(),
                needle: "/foo/".into(),
            },
            &ctx
        ));
        assert!(evaluate_condition(
            &RuleCondition::PrefixMatch {
                value: "path".into(),
                prefix: "src/".into(),
            },
            &ctx
        ));
        assert!(evaluate_condition(
            &RuleCondition::SuffixMatch {
                value: "path".into(),
                suffix: ".rs".into(),
            },
            &ctx
        ));
        assert!(!evaluate_condition(
            &RuleCondition::SuffixMatch {
                value: "path".into(),
                suffix: ".py".into(),
            },
            &ctx
        ));
        assert!(!evaluate_condition(
            &RuleCondition::SubstringMatch {
                haystack: "missing".into(),
                needle: "x".into(),
            },
            &ctx
        ));
    }

    #[test]
    fn range_match_inclusive() {
        let cond = RuleCondition::RangeMatch {
            value: 50,
            min: 10,
            max: 100,
        };
        assert!(evaluate_condition(&cond, &empty_ctx()));
        let cond = RuleCondition::RangeMatch {
            value: 5,
            min: 10,
            max: 100,
        };
        assert!(!evaluate_condition(&cond, &empty_ctx()));
    }

    #[test]
    fn field_in_set_resolves_via_context() {
        let ctx = StaticCtx {
            counts: &[],
            size: 0,
            fields: &[("detector_id", "aws-access-key")],
        };
        use smallvec::smallvec;
        let cond = RuleCondition::FieldInSet {
            field: "detector_id".into(),
            set: smallvec!["github-pat".into(), "aws-access-key".into()],
        };
        assert!(evaluate_condition(&cond, &ctx));
        let cond = RuleCondition::FieldInSet {
            field: "detector_id".into(),
            set: smallvec!["stripe".into()],
        };
        assert!(!evaluate_condition(&cond, &ctx));
        let cond = RuleCondition::FieldInSet {
            field: "missing".into(),
            set: smallvec!["x".into()],
        };
        assert!(!evaluate_condition(&cond, &ctx));
    }

    #[test]
    fn set_membership() {
        use smallvec::smallvec;
        let cond = RuleCondition::SetMembership {
            value: "blue".into(),
            set: smallvec!["red".into(), "blue".into(), "green".into()],
        };
        assert!(evaluate_condition(&cond, &empty_ctx()));
        let cond = RuleCondition::SetMembership {
            value: "yellow".into(),
            set: smallvec!["red".into(), "blue".into()],
        };
        assert!(!evaluate_condition(&cond, &empty_ctx()));
    }

    #[test]
    fn regex_match_uses_field_value() {
        let ctx = StaticCtx {
            counts: &[],
            size: 0,
            fields: &[("commit", "abcdef1234567890")],
        };
        let cond = RuleCondition::RegexMatch {
            field: "commit".into(),
            pattern: "^[0-9a-f]+$".into(),
        };
        assert!(evaluate_condition(&cond, &ctx));
        let cond = RuleCondition::RegexMatch {
            field: "commit".into(),
            pattern: "^[A-Z]+$".into(),
        };
        assert!(!evaluate_condition(&cond, &ctx));
        // Unknown field → false.
        let cond = RuleCondition::RegexMatch {
            field: "missing".into(),
            pattern: ".*".into(),
        };
        assert!(!evaluate_condition(&cond, &ctx));
    }

    #[test]
    fn formula_and_or_not_short_circuit() {
        let ctx = empty_ctx();
        let f = RuleFormula::and(
            RuleFormula::condition(RuleCondition::LiteralTrue),
            RuleFormula::condition(RuleCondition::LiteralTrue),
        );
        assert!(evaluate_formula(&f, &ctx));

        let f = RuleFormula::and(
            RuleFormula::condition(RuleCondition::LiteralTrue),
            RuleFormula::condition(RuleCondition::LiteralFalse),
        );
        assert!(!evaluate_formula(&f, &ctx));

        let f = RuleFormula::or(
            RuleFormula::condition(RuleCondition::LiteralFalse),
            RuleFormula::condition(RuleCondition::LiteralTrue),
        );
        assert!(evaluate_formula(&f, &ctx));

        let f = RuleFormula::not_formula(RuleFormula::condition(RuleCondition::LiteralFalse));
        assert!(evaluate_formula(&f, &ctx));
    }

    #[test]
    fn nested_formula() {
        // (PatternExists(7) AND FileSizeLt(2048)) OR NOT PatternCountGt(99, 1000)
        let ctx = StaticCtx {
            counts: &[(7, 3), (99, 50)],
            size: 1024,
            fields: &[],
        };
        let f = RuleFormula::or(
            RuleFormula::and(
                RuleFormula::condition(RuleCondition::PatternExists { pattern_id: 7 }),
                RuleFormula::condition(RuleCondition::FileSizeLt(2048)),
            ),
            RuleFormula::not_formula(RuleFormula::condition(RuleCondition::PatternCountGt {
                pattern_id: 99,
                threshold: 1000,
            })),
        );
        assert!(evaluate_formula(&f, &ctx));
    }
}
