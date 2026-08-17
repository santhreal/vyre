//! Lookup target for a dialect op's lowering path.
//!
//! Restored for SemVer baseline compatibility with published `vyre-driver` releases.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Lookup target for a dialect op's lowering path.
///
/// The in-tree variants map to the typed slots on a dialect op's lowering table.
/// Out-of-tree backends register by stable backend id via the table's
/// `extensions` map and are looked up by `Target::Extension("backend-id")`.
/// The enum is `#[non_exhaustive]` so adding an in-tree variant does not
/// break downstream matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Target {
    /// CUDA PTX AOT artifact target.
    Ptx,
    /// SPIR-V AOT artifact target.
    SpirV,
    /// Primary text target.
    PrimaryText,
    /// Primary binary target.
    PrimaryBinary,
    /// Secondary text target.
    SecondaryText,
    /// native-module IR. Reserved for a native-module emitter.
    NativeModule,
    /// Portable reference backend. Always available.
    ReferenceBackend,
    /// Out-of-tree backend registered by stable id. Matches the
    /// string a consumer wrote into the lowering table extension map.
    /// Examples are backend-owned stable identifiers.
    Extension(&'static str),
}

impl Target {
    /// Stable AOT target id consumed by the `vyre-driver` AOT emitter registry.
    #[must_use]
    pub fn aot_target_id(self) -> &'static str {
        match self {
            Self::Ptx | Self::SecondaryText => "secondary_text",
            Self::SpirV => "spv",
            Self::PrimaryText => "primary_text",
            Self::PrimaryBinary => "primary_binary",
            Self::NativeModule => "native_module",
            Self::ReferenceBackend => "reference_backend",
            Self::Extension(id) => id,
        }
    }

    /// File-extension hint for AOT bundles.
    #[must_use]
    pub fn extension(self) -> &'static str {
        self.aot_target_id()
    }
}

impl Serialize for Target {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.aot_target_id())
    }
}

impl<'de> Deserialize<'de> for Target {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "ptx" | "secondary_text" => Ok(Self::Ptx),
            "spv" | "spirv" => Ok(Self::SpirV),
            "primary_text" => Ok(Self::PrimaryText),
            "primary_binary" => Ok(Self::PrimaryBinary),
            "native_module" => Ok(Self::NativeModule),
            "reference_backend" => Ok(Self::ReferenceBackend),
            other => Err(serde::de::Error::custom(format!(
                "unsupported target `{other}`: unknown targets are rejected fail-closed to avoid unbounded memory growth. Fix: use a supported in-tree target name (ptx, spv, primary_text, primary_binary, native_module, reference_backend)."
            ))),
        }
    }
}
