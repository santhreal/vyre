# Operation List

**Status: Superseded**

The canonical operation contract is
[`docs/generated/OP_SCHEMA.json`](generated/OP_SCHEMA.json). It contains every
linked operation ID, tier, category, signature, feature route, oracle, backend
status, algebraic-law registration, and composition chain.

Use [`docs/generated/OP_INVENTORY.md`](generated/OP_INVENTORY.md) for a
Markdown table. Use [`docs/catalog/`](catalog/) to browse operations by
subsystem.

Regenerate and verify the authoritative views with:

```bash
cargo run -p xtask --bin xtask -- operation-schema
cargo run -p xtask --bin xtask -- list-ops
cargo run -p xtask --bin xtask -- catalog
cargo run -p xtask --bin xtask -- operation-schema --check
```
