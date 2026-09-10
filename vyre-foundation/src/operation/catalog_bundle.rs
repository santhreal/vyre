//! Explicit immutable catalog bundle whose contents, versions, extension provenance,
//! and digest participate in request and artifact identity.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use super::records::{LoweringProvider, SemanticDescriptor};
use super::registration::OperationRegistration;

/// Provenance metadata for an external or independently versioned dialect extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtensionProvenance {
    /// Extension identifier or crate name.
    pub name: &'static str,
    /// Declared extension schema / capability version.
    pub version: u32,
}

/// Explicit immutable catalog bundle containing only execution-relevant semantic descriptors
/// and callable lowering providers.
#[derive(Clone, Debug)]
pub struct OperationCatalogBundle {
    /// Schema version of the catalog bundle itself.
    pub version: u32,
    /// BLAKE3 256-bit digest of the bundle contents and extension provenance.
    pub digest: [u8; 32],
    /// Identity-indexed semantic descriptors.
    pub descriptors: BTreeMap<&'static str, SemanticDescriptor>,
    /// Identity-indexed lowering/implementation providers.
    pub lowering_providers: BTreeMap<&'static str, LoweringProvider>,
    /// Extension provenance records.
    pub extensions: BTreeMap<&'static str, ExtensionProvenance>,
}

impl OperationCatalogBundle {
    /// Construct an empty catalog bundle.
    #[must_use]
    pub fn empty() -> Self {
        let descriptors = BTreeMap::new();
        let lowering_providers = BTreeMap::new();
        let extensions = BTreeMap::new();
        let digest = Self::compute_digest(1, &descriptors, &lowering_providers, &extensions);
        Self {
            version: 1,
            digest,
            descriptors,
            lowering_providers,
            extensions,
        }
    }

    /// Construct a catalog bundle from explicit parts.
    #[must_use]
    pub fn from_parts(
        descriptors: BTreeMap<&'static str, SemanticDescriptor>,
        lowering_providers: BTreeMap<&'static str, LoweringProvider>,
        extensions: BTreeMap<&'static str, ExtensionProvenance>,
    ) -> Self {
        let digest = Self::compute_digest(1, &descriptors, &lowering_providers, &extensions);
        Self {
            version: 1,
            digest,
            descriptors,
            lowering_providers,
            extensions,
        }
    }

    /// Return the process-wide catalog bundle, built once from the global
    /// registry.
    ///
    /// The walk over the link-time inventory grows with the registration
    /// count, so it happens here and once. A caller that needs an owned value
    /// clones this one rather than reading the inventory again.
    #[must_use]
    pub fn global() -> &'static Self {
        static BUNDLE: LazyLock<OperationCatalogBundle> =
            LazyLock::new(OperationCatalogBundle::from_registry);
        &BUNDLE
    }

    /// Compute the immutable catalog bundle from the global registry.
    ///
    /// Reachable only through `global`, which runs it once.
    #[must_use]
    pub(crate) fn from_registry() -> Self {
        let mut descriptors = BTreeMap::new();
        let mut lowering_providers = BTreeMap::new();
        let extensions = BTreeMap::new();

        for desc in inventory::iter::<SemanticDescriptor> {
            descriptors.insert(desc.id, *desc);
        }
        for prov in inventory::iter::<LoweringProvider> {
            lowering_providers.insert(prov.id, *prov);
        }

        // Also bridge from OperationRegistration for backwards compatibility
        for reg in inventory::iter::<OperationRegistration> {
            descriptors
                .entry(reg.id)
                .or_insert_with(|| reg.descriptor());
            lowering_providers
                .entry(reg.id)
                .or_insert_with(|| reg.lowering_provider());
        }

        let digest = Self::compute_digest(1, &descriptors, &lowering_providers, &extensions);
        Self {
            version: 1,
            digest,
            descriptors,
            lowering_providers,
            extensions,
        }
    }

    /// Add a semantic descriptor and recompute the digest.
    #[must_use]
    pub fn with_descriptor(mut self, descriptor: SemanticDescriptor) -> Self {
        self.descriptors.insert(descriptor.id, descriptor);
        self.digest = Self::compute_digest(
            self.version,
            &self.descriptors,
            &self.lowering_providers,
            &self.extensions,
        );
        self
    }

    /// Add a lowering provider and recompute the digest.
    #[must_use]
    pub fn with_lowering(mut self, lowering: LoweringProvider) -> Self {
        self.lowering_providers.insert(lowering.id, lowering);
        self.digest = Self::compute_digest(
            self.version,
            &self.descriptors,
            &self.lowering_providers,
            &self.extensions,
        );
        self
    }

    /// Add both a semantic descriptor and an optional lowering provider.
    #[must_use]
    pub fn with_operation(
        mut self,
        descriptor: SemanticDescriptor,
        lowering: Option<LoweringProvider>,
    ) -> Self {
        let id = descriptor.id;
        self.descriptors.insert(id, descriptor);
        if let Some(lowering) = lowering {
            self.lowering_providers.insert(id, lowering);
        }
        self.digest = Self::compute_digest(
            self.version,
            &self.descriptors,
            &self.lowering_providers,
            &self.extensions,
        );
        self
    }

    /// Incorporate an independently versioned dialect extension into this closed catalog bundle.
    #[must_use]
    pub fn with_extension(
        mut self,
        name: &'static str,
        version: u32,
        descriptors: impl IntoIterator<Item = SemanticDescriptor>,
        lowerings: impl IntoIterator<Item = LoweringProvider>,
    ) -> Self {
        self.extensions
            .insert(name, ExtensionProvenance { name, version });
        for desc in descriptors {
            self.descriptors.insert(desc.id, desc);
        }
        for low in lowerings {
            self.lowering_providers.insert(low.id, low);
        }
        self.digest = Self::compute_digest(
            self.version,
            &self.descriptors,
            &self.lowering_providers,
            &self.extensions,
        );
        self
    }

    /// Compute the deterministic 256-bit BLAKE3 digest of bundle contents.
    pub fn compute_digest(
        version: u32,
        descriptors: &BTreeMap<&'static str, SemanticDescriptor>,
        lowering_providers: &BTreeMap<&'static str, LoweringProvider>,
        extensions: &BTreeMap<&'static str, ExtensionProvenance>,
    ) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-foundation::catalog_bundle::v1\n");
        hasher.update(&version.to_le_bytes());
        for (&name, ext) in extensions {
            hasher.update(b"ext:");
            hasher.update(name.as_bytes());
            hasher.update(&ext.version.to_le_bytes());
        }
        for (&id, desc) in descriptors {
            hasher.update(b"op:");
            hasher.update(id.as_bytes());
            hasher.update(&desc.semantic_version.to_le_bytes());
            hasher.update(&[desc.tier as u8]);
            if let Some(cat) = desc.category {
                hasher.update(cat.as_bytes());
            }
            for law in desc.laws {
                hasher.update(law.as_bytes());
            }
            if let Some(sig) = desc.signature {
                hasher.update(&(sig.inputs.len() as u64).to_le_bytes());
                for in_param in sig.inputs {
                    hasher.update(in_param.name.as_bytes());
                    hasher.update(in_param.ty.as_bytes());
                }
                hasher.update(&(sig.outputs.len() as u64).to_le_bytes());
                for out_param in sig.outputs {
                    hasher.update(out_param.name.as_bytes());
                    hasher.update(out_param.ty.as_bytes());
                }
                hasher.update(&[sig.bytes_extraction as u8]);
            }
            if let Some(eff) = desc.explicit_effects {
                hasher.update(&[
                    eff.reads as u8,
                    eff.writes as u8,
                    eff.atomics as u8,
                    eff.synchronizes as u8,
                ]);
            }
            if let Some(caps) = desc.explicit_capabilities {
                hasher.update(&[
                    caps.subgroup_ops as u8,
                    caps.f16 as u8,
                    caps.bf16 as u8,
                    caps.f64 as u8,
                    caps.async_dispatch as u8,
                    caps.indirect_dispatch as u8,
                    caps.tensor_ops as u8,
                    caps.trap as u8,
                    caps.distributed_collectives as u8,
                ]);
                hasher.update(&caps.static_storage_bytes.to_le_bytes());
            }
            if lowering_providers.contains_key(id) {
                hasher.update(b":lowering:present\n");
            }
        }
        *hasher.finalize().as_bytes()
    }

    /// Derive an artifact identity from this bundle and a request/program fingerprint.
    #[must_use]
    pub fn artifact_identity(&self, request_or_program_digest: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-foundation::artifact_identity::v1\n");
        hasher.update(&self.digest);
        hasher.update(request_or_program_digest);
        *hasher.finalize().as_bytes()
    }

    /// Return the 256-bit BLAKE3 digest of the catalog bundle.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Look up a semantic descriptor by stable operation id.
    #[must_use]
    pub fn descriptor(&self, id: &str) -> Option<&SemanticDescriptor> {
        self.descriptors.get(id)
    }

    /// Look up a lowering provider by stable operation id.
    #[must_use]
    pub fn lowering(&self, id: &str) -> Option<&LoweringProvider> {
        self.lowering_providers.get(id)
    }

    /// Check whether an operation id is present in the bundle.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.descriptors.contains_key(id)
    }

    /// Return the number of operations in the bundle.
    #[must_use]
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Check whether the bundle is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Return the extensions contained in this bundle.
    #[must_use]
    pub fn extensions(&self) -> &BTreeMap<&'static str, ExtensionProvenance> {
        &self.extensions
    }

    /// Return an extension provenance by name.
    #[must_use]
    pub fn extension(&self, name: &str) -> Option<&ExtensionProvenance> {
        self.extensions.get(name)
    }
}
