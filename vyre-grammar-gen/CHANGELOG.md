# Changelog

All notable changes to `vyre-grammar-gen` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The C11 token ids are no longer declared here. `vyre-spec::c11_token` owns the
  numbering, this crate depends down onto it, and `c11_lexer` re-exports it, so
  `vyre_grammar_gen::c11_lexer::TOK_*` still resolves and no value changed. The
  ids are the wire contract between the blobs this crate emits and the GPU lexer
  that decodes them, and a second declaration could drift without failing a
  build.

## [0.6.0] - 2026-05-28

### Added
- Changelog established at the current release. Earlier history predates this
  file and is recorded in the version control log.

### Current capabilities
- Host-side C11 grammar table generator for the vyre GPU C parser: emits the DFA
  lexer plus LR(1) action/goto tables as binary blobs that `vyre-libs::parsing`
  loads as read-only buffers.
