//! Every scan construct row carries a tier, a backend route per backend, a
//! proof gate and a diagnostic.

use std::collections::BTreeSet;

use toml::Value;

use super::toml_rows::{
    assert_existing_paths, read_bench_targets, read_toml, required_array, required_bool,
    required_str, string_set, workspace_root,
};

const REQUIRED_SCAN_CONSTRUCT_TIERS: [&str; 5] = [
    "supported",
    "rejected",
    "approximated",
    "accelerator-only",
    "verifier-required",
];

const REQUIRED_SCAN_CONSTRUCT_BACKENDS: [&str; 9] = [
    "cpu_ref",
    "cuda",
    "wgpu",
    "metal",
    "hyperscan",
    "vectorscan",
    "rust_regex",
    "dpu",
    "fpga",
];

#[test]
fn op_matrix_scan_construct_tiers_have_proof_and_diagnostics() {
    let root = workspace_root();
    let matrix = read_toml(&root.join("docs/optimization/OP_MATRIX.toml"));
    let bench_targets = read_bench_targets(&root);

    let tier_values = string_set(
        matrix
            .get("scan_construct_tier_values")
            .and_then(Value::as_array)
            .expect("Fix: OP_MATRIX.toml must declare scan_construct_tier_values."),
    );
    let route_values = string_set(
        matrix
            .get("scan_construct_route_values")
            .and_then(Value::as_array)
            .expect("Fix: OP_MATRIX.toml must declare scan_construct_route_values."),
    );
    for required in REQUIRED_SCAN_CONSTRUCT_TIERS.iter().copied() {
        assert!(
            tier_values.contains(required),
            "Fix: OP_MATRIX.toml scan_construct_tier_values must include `{required}`."
        );
    }

    let rows = matrix
        .get("scan_construct")
        .and_then(Value::as_array)
        .expect("Fix: OP_MATRIX.toml must contain [[scan_construct]] rows.");

    let mut seen_ids = BTreeSet::new();
    let mut seen_tiers = BTreeSet::new();
    for row in rows {
        let id = required_str(row, "id");
        assert!(
            seen_ids.insert(id.to_string()),
            "Fix: duplicate OP_MATRIX scan construct id `{id}`."
        );

        let tier = required_str(row, "tier");
        assert!(
            tier_values.contains(tier),
            "Fix: OP_MATRIX scan construct `{id}` uses unregistered tier `{tier}`."
        );
        seen_tiers.insert(tier.to_string());

        let dialect_class = required_str(row, "dialect_class");
        assert!(
            !dialect_class.trim().is_empty(),
            "Fix: OP_MATRIX scan construct `{id}` must name a dialect_class."
        );

        let diagnostic_code = required_str(row, "diagnostic_code");
        assert!(
            diagnostic_code.starts_with("VYRE_SCAN_"),
            "Fix: OP_MATRIX scan construct `{id}` diagnostic_code `{diagnostic_code}` must use the VYRE_SCAN_ namespace."
        );

        let user_diagnostic = required_str(row, "user_diagnostic");
        assert!(
            user_diagnostic.len() >= 32,
            "Fix: OP_MATRIX scan construct `{id}` must include an operator-visible user_diagnostic."
        );

        let approximation_policy = required_str(row, "approximation_policy");
        assert!(
            !approximation_policy.trim().is_empty(),
            "Fix: OP_MATRIX scan construct `{id}` must name an approximation_policy."
        );

        let constructs = required_array(row, "constructs");
        assert!(
            !constructs.is_empty(),
            "Fix: OP_MATRIX scan construct `{id}` must list at least one syntax construct."
        );

        assert_existing_paths(&root, id, "proof_gates", required_array(row, "proof_gates"));
        if let Some(targets) = row.get("bench_targets").and_then(Value::as_array) {
            for target in targets {
                if let Some(target) = target.as_str() {
                    assert!(
                        bench_targets.contains(target),
                        "Fix: OP_MATRIX scan construct `{id}` references missing bench target `{target}`."
                    );
                }
            }
        }

        let routes = row
            .get("backend_routes")
            .and_then(Value::as_table)
            .unwrap_or_else(|| {
                panic!("Fix: OP_MATRIX scan construct `{id}` must contain backend_routes.")
            });
        let mut row_routes = Vec::new();
        for backend in REQUIRED_SCAN_CONSTRUCT_BACKENDS.iter().copied() {
            let route = routes
                .get(backend)
                .and_then(Value::as_str)
                .unwrap_or_else(|| {
                    panic!("Fix: OP_MATRIX scan construct `{id}` must route backend `{backend}`.")
                });
            assert!(
                route_values.contains(route),
                "Fix: OP_MATRIX scan construct `{id}` backend `{backend}` uses invalid route `{route}`."
            );
            row_routes.push(route);
        }

        let verifier_required = required_bool(row, "verifier_required");
        let accelerator_only = required_bool(row, "accelerator_only");
        match tier {
            "supported" => assert!(
                row_routes.iter().any(|route| *route == "native"),
                "Fix: supported scan construct `{id}` must have at least one native route."
            ),
            "rejected" => assert!(
                row_routes.iter().all(|route| *route == "unsupported"),
                "Fix: rejected scan construct `{id}` must route every backend to unsupported."
            ),
            "approximated" => {
                assert!(
                    approximation_policy != "exact",
                    "Fix: approximated scan construct `{id}` must not use exact approximation_policy."
                );
                assert!(
                    verifier_required,
                    "Fix: approximated scan construct `{id}` must require verifier proof."
                );
                assert!(
                    row_routes.iter().any(|route| *route == "prefilter"),
                    "Fix: approximated scan construct `{id}` must have at least one prefilter route."
                );
            }
            "accelerator-only" => {
                assert!(
                    accelerator_only,
                    "Fix: accelerator-only scan construct `{id}` must set accelerator_only = true."
                );
                assert!(
                    row_routes
                        .iter()
                        .any(|route| *route == "external-accelerator"),
                    "Fix: accelerator-only scan construct `{id}` must have an external-accelerator route."
                );
            }
            "verifier-required" => {
                assert!(
                    verifier_required,
                    "Fix: verifier-required scan construct `{id}` must set verifier_required = true."
                );
                assert!(
                    row_routes.iter().any(|route| *route == "verifier"),
                    "Fix: verifier-required scan construct `{id}` must have at least one verifier route."
                );
            }
            other => panic!("Fix: unhandled scan construct tier `{other}` for `{id}`."),
        }
    }

    for required in REQUIRED_SCAN_CONSTRUCT_TIERS.iter().copied() {
        assert!(
            seen_tiers.contains(required),
            "Fix: OP_MATRIX.toml must include at least one scan construct row with tier `{required}`."
        );
    }
}
