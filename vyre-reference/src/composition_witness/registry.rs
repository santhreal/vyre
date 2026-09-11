//! The composition-family witness registry.
//!
//! One row per composition family, in one place. A family is an operation
//! category as the live operation registry names it, which is what
//! `docs/generated/op-inventory.toml` and `docs/generated/catalog.toml` are
//! generated from. Each row states either the `composition_witness` module
//! that owns the family's independent sequential witnesses, together with a
//! probe that reaches one of them, or the reason the family carries no witness.
//!
//! Before this table the crate published a few hundred witness functions and
//! nothing said which family each one served. A family could be added to the
//! operation registry, ship without an independent witness, and leave every
//! suite green, because "does this family have a witness" was not a question
//! any code asked. `oracle_witness_registry_is_derived` asks it against the
//! generated inventory at run time.
//!
//! Witnesses are keyed by family only. No witness is selected by transform:
//! the pass-composition witnesses (`compose_passes_witness`,
//! `passes_commute_on_witness`) take the mapping as data rather than being
//! looked up by a transform name, so there is no transform table to derive.

/// Digest of one family's witness output, produced by running that witness on
/// a fixed input.
///
/// A probe exists so a registry row is a reachable code path rather than a
/// name. It states nothing about whether the witness is right, which is what
/// the per-family known-answer contracts are for.
pub type WitnessProbe = fn() -> u64;

/// What independent witness a composition family carries.
#[derive(Clone, Copy)]
pub enum FamilyWitness {
    /// The family's witnesses are owned by one `composition_witness` module.
    Owned {
        /// Module of `composition_witness` that owns them.
        module: &'static str,
        /// Probe reaching one of that module's witnesses.
        probe: WitnessProbe,
    },
    /// The family carries no independent witness.
    Absent {
        /// Why an independent sequential witness would state nothing.
        reason: &'static str,
    },
}

/// One composition family and its witness ownership.
#[derive(Clone, Copy)]
pub struct CompositionWitnessFamily {
    /// Operation category the family covers, as the operation registry names it.
    pub family: &'static str,
    /// Independent witness for the family, or the reason there is none.
    pub witness: FamilyWitness,
}

impl CompositionWitnessFamily {
    /// The module owning this family's witnesses, when it has one.
    #[must_use]
    pub const fn module(&self) -> Option<&'static str> {
        match self.witness {
            FamilyWitness::Owned { module, .. } => Some(module),
            FamilyWitness::Absent { .. } => None,
        }
    }

    /// The probe reaching this family's witnesses, when it has one.
    #[must_use]
    pub const fn probe(&self) -> Option<WitnessProbe> {
        match self.witness {
            FamilyWitness::Owned { probe, .. } => Some(probe),
            FamilyWitness::Absent { .. } => None,
        }
    }
}

/// Every composition family, with the witness module that answers for it.
///
/// Ordered by family id so a diff of this table reads as a diff of the
/// vocabulary.
pub const COMPOSITION_WITNESS_FAMILIES: &[CompositionWitnessFamily] = &[
    CompositionWitnessFamily {
        family: "bitset",
        witness: FamilyWitness::Owned {
            module: "bitset",
            probe: probe_bitset,
        },
    },
    CompositionWitnessFamily {
        family: "builder",
        witness: FamilyWitness::Absent {
            reason: "the builder ops are index arithmetic over a caller-supplied map, so a \
                     sequential witness would restate the index expression it is checking",
        },
    },
    CompositionWitnessFamily {
        family: "decode",
        witness: FamilyWitness::Owned {
            module: "encoding",
            probe: probe_encoding,
        },
    },
    CompositionWitnessFamily {
        family: "fixpoint",
        witness: FamilyWitness::Owned {
            module: "csr",
            probe: probe_csr,
        },
    },
    CompositionWitnessFamily {
        family: "geom",
        witness: FamilyWitness::Owned {
            module: "geometry",
            probe: probe_geometry,
        },
    },
    CompositionWitnessFamily {
        family: "graph",
        witness: FamilyWitness::Owned {
            module: "graph",
            probe: probe_graph,
        },
    },
    CompositionWitnessFamily {
        family: "hardware",
        witness: FamilyWitness::Absent {
            reason: "a hardware primitive is one machine operation, and vyre-primitives already \
                     owns its host body through ReferenceFacet",
        },
    },
    CompositionWitnessFamily {
        family: "hash",
        witness: FamilyWitness::Owned {
            module: "hash",
            probe: probe_hash,
        },
    },
    CompositionWitnessFamily {
        family: "label",
        witness: FamilyWitness::Owned {
            module: "csr",
            probe: probe_csr,
        },
    },
    CompositionWitnessFamily {
        family: "llm",
        witness: FamilyWitness::Owned {
            module: "math",
            probe: probe_math,
        },
    },
    CompositionWitnessFamily {
        family: "logical",
        witness: FamilyWitness::Owned {
            module: "bitset",
            probe: probe_bitset,
        },
    },
    CompositionWitnessFamily {
        family: "math",
        witness: FamilyWitness::Owned {
            module: "math",
            probe: probe_math,
        },
    },
    CompositionWitnessFamily {
        family: "nn",
        witness: FamilyWitness::Owned {
            module: "math",
            probe: probe_math,
        },
    },
    CompositionWitnessFamily {
        family: "opt",
        witness: FamilyWitness::Owned {
            module: "math",
            probe: probe_math,
        },
    },
    CompositionWitnessFamily {
        family: "parsing",
        witness: FamilyWitness::Owned {
            module: "parsing",
            probe: probe_parsing,
        },
    },
    CompositionWitnessFamily {
        family: "pattern",
        witness: FamilyWitness::Owned {
            module: "pattern",
            probe: probe_pattern,
        },
    },
    CompositionWitnessFamily {
        family: "predicate",
        witness: FamilyWitness::Owned {
            module: "reasoning",
            probe: probe_reasoning,
        },
    },
    CompositionWitnessFamily {
        family: "reduce",
        witness: FamilyWitness::Owned {
            module: "reduction",
            probe: probe_reduction,
        },
    },
    CompositionWitnessFamily {
        family: "representation",
        witness: FamilyWitness::Owned {
            module: "math",
            probe: probe_math,
        },
    },
    CompositionWitnessFamily {
        family: "scan",
        witness: FamilyWitness::Owned {
            module: "pattern",
            probe: probe_pattern,
        },
    },
    CompositionWitnessFamily {
        family: "security",
        witness: FamilyWitness::Owned {
            module: "csr",
            probe: probe_csr,
        },
    },
    CompositionWitnessFamily {
        family: "text",
        witness: FamilyWitness::Owned {
            module: "text",
            probe: probe_text,
        },
    },
    CompositionWitnessFamily {
        family: "vfs",
        witness: FamilyWitness::Absent {
            reason: "path resolution is host filesystem naming rather than a numerical \
                     composition, and the vfs contracts assert the resolved path directly",
        },
    },
    CompositionWitnessFamily {
        family: "visual",
        witness: FamilyWitness::Absent {
            reason: "a raster composite is judged against byte-identical frame goldens, which \
                     bind tighter than a witness restating the blend",
        },
    },
];

/// Modules of `composition_witness` that no operation category names.
///
/// A witness module reached by no family is dead unless someone records why.
/// These two back operations the registry files under another category:
/// do-calculus surgeries back graph analyses, and frontier planning backs the
/// reduction and graph ops that consume the plan.
pub const MODULES_WITHOUT_A_CATALOG_FAMILY: &[(&str, &str)] = &[
    (
        "causal",
        "do-calculus surgery witnesses back operations the registry files under graph",
    ),
    (
        "scheduling",
        "frontier planning witnesses back the reduce and graph operations that consume the plan",
    ),
];

/// The family row for `family`, if the registry has one.
#[must_use]
pub fn witness_family(family: &str) -> Option<&'static CompositionWitnessFamily> {
    COMPOSITION_WITNESS_FAMILIES
        .iter()
        .find(|row| row.family == family)
}

/// Every family row, in family-id order.
pub fn witness_families() -> impl ExactSizeIterator<Item = &'static CompositionWitnessFamily> {
    COMPOSITION_WITNESS_FAMILIES.iter()
}

/// Digest of a word sequence, so a probe reports one number for any witness shape.
fn digest_words(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for word in words {
        state ^= word;
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    state
}

fn probe_bitset() -> u64 {
    digest_words(
        super::bitset_popcount_witness(&[0x0000_00ff, 0xffff_0000, 0])
            .into_iter()
            .map(u64::from),
    )
}

fn probe_csr() -> u64 {
    digest_words(
        super::resolve_family_witness(&[0b0001, 0b0110, 0b1000], 0b0010)
            .into_iter()
            .map(u64::from),
    )
}

fn probe_encoding() -> u64 {
    digest_words(
        super::hex_decode_packed_witness(b"0a1F")
            .into_iter()
            .map(u64::from),
    )
}

fn probe_geometry() -> u64 {
    digest_words(
        super::clifford2_product_witness([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
            .into_iter()
            .map(f64::to_bits),
    )
}

fn probe_graph() -> u64 {
    digest_words(
        super::canonicalize_union_find_witness(&[0, 0, 1, 2])
            .into_iter()
            .map(u64::from),
    )
}

fn probe_hash() -> u64 {
    digest_words([u64::from(super::crc32_witness(b"vyre"))])
}

fn probe_math() -> u64 {
    digest_words(
        super::matmul_u32_witness(&[1, 2, 3, 4], &[5, 6, 7, 8], None, 2, 2, 2)
            .into_iter()
            .map(u64::from),
    )
}

fn probe_parsing() -> u64 {
    digest_words(
        super::whitespace_classify_word_witness(&[0x2020_0941, 0x0a42_2043])
            .into_iter()
            .map(u64::from),
    )
}

fn probe_pattern() -> u64 {
    digest_words(
        super::bracket_match_witness(&[1, 1, 2, 2], 4)
            .into_iter()
            .map(u64::from),
    )
}

fn probe_reasoning() -> u64 {
    digest_words(
        super::compose_passes_witness(&[0, 1, 2], &[2, 0, 1], 3, &[1, 2, 0], 3)
            .into_iter()
            .map(u64::from),
    )
}

fn probe_reduction() -> u64 {
    digest_words(
        super::inclusive_prefix_sum_witness(&[1, 2, 3, 4])
            .into_iter()
            .map(u64::from),
    )
}

fn probe_text() -> u64 {
    digest_words(super::byte_histogram_witness(b"vyre").into_iter().map(u64::from))
}
