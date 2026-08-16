# Vyre architecture

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the short page the
architecture gate checks against the live tree, and the entry point for the
architecture chapters.
[`docs/architecture/crates.md`](docs/architecture/crates.md) states what each
crate owns and what it must not hold.
[`docs/lego-block-rule.md`](docs/lego-block-rule.md) states the two placement
rules: composed, not rewritten; intrinsic means uncomposable.
[`THESIS.md`](THESIS.md) is the design argument. Per-crate `README` files apply
the placement rules to one crate.
