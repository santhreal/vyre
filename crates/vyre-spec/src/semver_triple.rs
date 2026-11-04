//! The one declaration of a `major.minor.patch` version record.
//!
//! Four crates in this workspace version four different contracts: an
//! extension schema, a protocol, a canonical wire schema, and a registry op.
//! Each one is the same record with the same constructor and the same
//! `Display`, and each one used to spell that record out again, so a fix to
//! the ordering, the formatting, or the field set landed in one copy and left
//! three behind.
//!
//! [`semver_triple!`] is that record. The invoking module supplies the type
//! name, its documentation, and its derives, because the derive set is the one
//! part that differs: a version that crosses the wire needs `Serialize` and a
//! version that stays in process does not. Everything the four copies agreed
//! on is generated here.
//!
//! Domain-specific behaviour stays with the domain: fixed version constants,
//! parsing, and compatibility predicates belong in an additional `impl` block
//! next to the invocation.

/// Declare a `major.minor.patch` version record with its constructor and
/// `Display`.
///
/// `Ord` is not generated: it is derived at the invocation, so the field
/// declaration order below is what makes ordinary comparison lexicographic
/// major to minor to patch.
///
/// ```
/// vyre_spec::semver_triple! {
///     /// Version of the thing being versioned.
///     #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
///     pub struct ThingVersion;
/// }
///
/// let version = ThingVersion::new(1, 4, 2);
/// assert_eq!(version.to_string(), "1.4.2");
/// assert!(ThingVersion::new(1, 5, 0) > version);
/// ```
#[macro_export]
macro_rules! semver_triple {
    (
        $(#[$attribute:meta])*
        $visibility:vis struct $name:ident;
    ) => {
        $(#[$attribute])*
        $visibility struct $name {
            /// Breaking-change component.
            pub major: u32,
            /// Backward-compatible addition component.
            pub minor: u32,
            /// Backward-compatible fix component.
            pub patch: u32,
        }

        impl $name {
            /// Construct a version from explicit numeric components.
            #[must_use]
            pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
                Self {
                    major,
                    minor,
                    patch,
                }
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
            }
        }
    };
}
