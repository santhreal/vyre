| finding id | status | commit hash | notes |
| --- | --- | --- | --- |
| F-SP-3 | working-tree |  | signal registry now reports duplicate providers instead of panicking |
| F-SP-4 | working-tree |  | regex validation now applies explicit compile budgets |
| F-SP-5 | working-tree |  | literal-family variant tokens validated against supported set |
| F-SP-6 | working-tree |  | exact literals now hard-cap oversized needles |
| F-SP-7 | working-tree |  | expansion now honors authored variant list |
| F-SP-8 | working-tree |  | expanded patterns preserve raw bytes and variant metadata |
| F-SP-9 | working-tree |  | unsupported signal types now fail loudly in expansion |
| F-SP-10 | working-tree |  | mixed file selectors now fail compilation |
| F-SP-11 | working-tree |  | unsupported file selectors now fail applicability lowering |
| F-SP-12 | working-tree |  | rule applicability now ORs per-clause gates |
| F-SP-13 | working-tree |  | magic-prefix intersection keeps tighter compatible prefixes |
| F-SP-17 | working-tree |  | internal scanner clause names now include artifact namespace |
| F-SP-18 | pending |  | bundle temp-file collision fix not landed yet |
| F-SP-19 | pending |  | bundle header-first bounded load not landed yet |
| F-SP-20 | pending |  | bundle payload-size overflow rejection not landed yet |
| F-SP-21 | pending |  | index header-first bounded load not landed yet |
| F-SP-22 | pending |  | index vector-count caps not landed yet |
| F-SP-24 | pending |  | rule-name lookup index not landed yet |
| F-SP-25 | pending |  | borrowed/materialized match metadata change not landed yet |
| F-SP-26 | working-tree |  | batch splitting now preserves non-budget build errors |
| F-SP-27 | working-tree |  | rule-id u32 conversions now checked |
| F-SP-28 | pending |  | collector still clones immutable compile artifacts into plans |
| F-SP-29 | working-tree |  | oversized string ids no longer alias slot 0 |
| F-SP-30 | working-tree |  | hit counts now cap to cached-position budget |
| F-SP-31 | working-tree |  | regex-backed scan paths no longer abort collector setup |
| F-SP-32 | working-tree |  | DFA compilation now caches by literal-set fingerprint |
| F-SP-33 | working-tree |  | decode budget exhaustion now skips offending branch and continues siblings |
| F-SP-34 | pending |  | mixed-alphabet base64 provenance tightening not fully covered |
| F-SP-35 | working-tree |  | hex decode now scans raw bytes instead of requiring UTF-8 |
| F-SP-36 | working-tree |  | URL decode now scans raw bytes instead of requiring UTF-8 |
| F-SP-37 | working-tree |  | gzip decode now rejects limit-hit truncation as success |
| F-SP-38 | working-tree |  | dispatch now validates exact backend output arity |
| F-SP-39 | pending |  | SARIF unknown-severity downgrade fix not landed yet |
| F-SP-40 | pending |  | SARIF primary-region propagation not landed yet |
| F-SP-41 | pending |  | SARIF exploit-chain join key still too coarse |
