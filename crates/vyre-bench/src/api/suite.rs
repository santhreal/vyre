use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Which benchmark suite a case belongs to.
///
/// `Custom` carries an owned name so a suite named at run time costs one
/// allocation instead of a permanent one. It is not `Copy` for that reason.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuiteKind {
    Smoke,
    Release,
    Deep,
    Gpu,
    Sweep,
    CrossBackend,
    Evolve,
    Adversarial,
    Competition,
    Honest,
    Custom(Arc<str>),
}

impl std::str::FromStr for SuiteKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "smoke" => Ok(SuiteKind::Smoke),
            "release" => Ok(SuiteKind::Release),
            "deep" => Ok(SuiteKind::Deep),
            "gpu" => Ok(SuiteKind::Gpu),
            "sweep" => Ok(SuiteKind::Sweep),
            "cross-backend" | "cross_backend" => Ok(SuiteKind::CrossBackend),
            "evolve" => Ok(SuiteKind::Evolve),
            "adversarial" => Ok(SuiteKind::Adversarial),
            "competition" => Ok(SuiteKind::Competition),
            "honest" => Ok(SuiteKind::Honest),
            other => Ok(SuiteKind::Custom(Arc::from(other))),
        }
    }
}

impl SuiteKind {
    /// Name a custom suite without going through `FromStr`.
    #[must_use]
    pub fn custom(name: &str) -> Self {
        SuiteKind::Custom(Arc::from(name))
    }

    pub fn as_str(&self) -> &str {
        match self {
            SuiteKind::Smoke => "smoke",
            SuiteKind::Release => "release",
            SuiteKind::Deep => "deep",
            SuiteKind::Gpu => "gpu",
            SuiteKind::Sweep => "sweep",
            SuiteKind::CrossBackend => "cross-backend",
            SuiteKind::Evolve => "evolve",
            SuiteKind::Adversarial => "adversarial",
            SuiteKind::Competition => "competition",
            SuiteKind::Honest => "honest",
            SuiteKind::Custom(value) => value,
        }
    }
}
