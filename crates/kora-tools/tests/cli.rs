//! The command-line tools run end to end. Vulkan-dependent checks accept the documented "no
//! device" outcomes so they pass on machines without a GPU driver.

use std::process::Command;

use kora::json::{self, Value};

fn run(binary: &str, args: &[&str]) -> std::process::Output {
    Command::new(binary).args(args).output().expect("tool runs")
}

fn number(value: &Value, key: &str) -> u64 {
    match value.find(key) {
        Some(Value::Number(text)) => text.parse().unwrap(),
        other => panic!("{key} is not a number: {other:?}"),
    }
}

#[test]
fn kora_info_reports_schema_2_or_no_vulkan() {
    let output = run(env!("CARGO_BIN_EXE_kora-info"), &["--json"]);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("NoVulkanImplementation"), "{stderr}");
        return;
    }
    let report = json::parse(&output.stdout).unwrap();
    assert_eq!(number(&report, "schema_version"), 2);
    assert!(
        !report
            .find("devices")
            .and_then(Value::as_array)
            .unwrap()
            .is_empty()
    );
    assert!(
        !run(env!("CARGO_BIN_EXE_kora-info"), &["--bogus"])
            .status
            .success()
    );
}

#[test]
fn kora_bench_smoke_writes_schema_4() {
    let output = run(env!("CARGO_BIN_EXE_kora-bench"), &["--smoke"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = json::parse(&output.stdout).unwrap();
    assert_eq!(number(&report, "schema_version"), 4);
    let results = report.find("results").and_then(Value::as_array).unwrap();
    assert!(results.len() > 50);
    for result in results {
        assert!(number(result, "median") <= number(result, "p95"));
        assert_eq!(result.find("samples").and_then(Value::as_array).unwrap().len(), 3);
    }
    assert_eq!(
        run(env!("CARGO_BIN_EXE_kora-bench"), &["--iterations", "0"])
            .status
            .code(),
        Some(2)
    );
}

#[test]
fn kora_top_demo_verifies_results_or_skips() {
    let output = run(
        env!("CARGO_BIN_EXE_kora-top"),
        &["--demo", "--frames", "4", "--no-color"],
    );
    match output.status.code() {
        Some(77) => {}
        Some(0) => {
            let frames = String::from_utf8_lossy(&output.stdout);
            assert!(frames.contains("kora-top"), "{frames}");
            assert!(frames.contains("0 mismatches"), "{frames}");
        }
        code => panic!(
            "kora-top --demo exited with {code:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        ),
    }
}

#[test]
fn kora_worker_rejects_bad_arguments() {
    assert_eq!(
        run(env!("CARGO_BIN_EXE_kora-worker"), &["--bogus"]).status.code(),
        Some(2)
    );
    assert!(
        run(env!("CARGO_BIN_EXE_kora-worker"), &["--help"])
            .status
            .success()
    );
}
