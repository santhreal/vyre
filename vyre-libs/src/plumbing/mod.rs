//! Shared plumbing every composition needs and no domain owns.
//!
//! A directory under `src/` is a dialect: one family of compositions with its
//! own registered operations. The four modules here are not that. They carry
//! the facts a composition of any dialect needs before, around and after the
//! IR it builds: what its buffer arguments are, what the built `Program`
//! declares, what its registration says about it, and what the host does to
//! launch it. Each one had several dialects needing it and no dialect able to
//! own it, which is why they sat loose at the crate root and made the root read
//! as a pile rather than a table of contents.
//!
//! The module is `pub(crate)`. The items a consumer pins are re-exported from
//! `lib.rs` at the paths they have always had, so this directory adds no second
//! path to anything.

pub(crate) mod host;
pub(crate) mod operand;
pub(crate) mod program;
pub(crate) mod registration;
