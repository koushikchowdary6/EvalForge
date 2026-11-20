"""Integration tests for SciFact orchestration and retained evidence.

The fake installed commands isolate the runner's subprocess contract from
retrieval scoring; BEIR scoring itself is covered by separate CLI tests.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_scifact.py"


def _setup(tmp_path, comparison_exit=0):
    dataset = tmp_path / "dataset"
    (dataset / "qrels").mkdir(parents=True)
    for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        (dataset / name).write_text("fixture\n", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # The stub records the actual CLI invocation and writes the expected
    # artifact, so this tests the real subprocess and file-system boundary.
    stub = """#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
out = pathlib.Path(args[args.index('--output') + 1])
report = {'argv': args}
if pathlib.Path(sys.argv[0]).name == 'evalforge-benchmark-beir':
    import hashlib
    report['source'] = {'source_sha256': {
        key: hashlib.sha256(pathlib.Path(args[args.index(flag) + 1]).read_bytes()).hexdigest()
        for key, flag in (('corpus', '--corpus'), ('queries', '--queries'), ('qrels', '--qrels'))
    }}
out.write_text(json.dumps(report), encoding='utf-8')
if pathlib.Path(sys.argv[0]).name == 'evalforge-compare-beir':
    sys.exit(int(os.environ.get('COMPARE_EXIT', '0')))
"""
    for name in ("evalforge-benchmark-beir", "evalforge-compare-beir"):
        path = bin_dir / name
        path.write_text(stub, encoding="utf-8")
        path.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
               COMPARE_EXIT=str(comparison_exit))
    cmd = [sys.executable, str(SCRIPT), "--dataset", str(dataset),
           "--output-dir", str(output), "--evalforge-revision", "test-revision",
           "--source-url", "https://example.org/scifact", "--download-date", "2026-11-19"]
    return output, cmd, env


def test_runner_executes_both_tokenizers_and_comparison(tmp_path):
    output, cmd, env = _setup(tmp_path)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert {p.name for p in output.iterdir()} == {
        "manifest.json", "scifact-ascii.json", "scifact-unicode.json",
        "scifact-comparison.json",
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evalforge_revision"] == "test-revision"
    assert set(manifest["input_sha256"]) == {"corpus", "queries", "qrels"}
    assert all(len(digest) == 64 for digest in manifest["input_sha256"].values())
    for tokenizer in ("ascii", "unicode"):
        args = json.loads((output / f"scifact-{tokenizer}.json").read_text())["argv"]
        assert args[args.index("--tokenizer") + 1] == tokenizer
        assert args[args.index("--k") + 1:args.index("--output")] == ["1", "3", "5", "10"]
    comparison = json.loads((output / "scifact-comparison.json").read_text())["argv"]
    assert "--all-k" in comparison
    assert comparison[comparison.index("--max-drop") + 1] == "0.02"


def test_failed_quality_gate_retains_all_evidence(tmp_path):
    output, cmd, env = _setup(tmp_path, comparison_exit=1)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 1
    assert "evidence retained" in completed.stderr
    assert all((output / name).exists() for name in (
        "manifest.json", "scifact-ascii.json", "scifact-unicode.json",
        "scifact-comparison.json",
    ))


def test_success_exit_without_benchmark_artifact_is_error(tmp_path):
    output, cmd, env = _setup(tmp_path)
    benchmark = tmp_path / "bin" / "evalforge-benchmark-beir"
    benchmark.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    benchmark.chmod(0o755)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "did not write" in completed.stderr
    assert (output / "manifest.json").exists()
    assert not (output / "scifact-comparison.json").exists()


def test_success_exit_without_comparison_artifact_is_error(tmp_path):
    output, cmd, env = _setup(tmp_path)
    comparator = tmp_path / "bin" / "evalforge-compare-beir"
    comparator.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    comparator.chmod(0o755)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "without writing JSON evidence" in completed.stderr
    assert (output / "scifact-ascii.json").exists()
    assert (output / "scifact-unicode.json").exists()


def test_invalid_benchmark_json_is_error(tmp_path):
    output, cmd, env = _setup(tmp_path)
    benchmark = tmp_path / "bin" / "evalforge-benchmark-beir"
    benchmark.write_text(
        "#!/usr/bin/env python3\nimport pathlib, sys\n"
        "pathlib.Path(sys.argv[sys.argv.index('--output') + 1]).write_text('{bad')\n",
        encoding="utf-8",
    )
    benchmark.chmod(0o755)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "unreadable JSON" in completed.stderr


def test_non_object_benchmark_evidence_is_rejected(tmp_path):
    output, cmd, env = _setup(tmp_path)
    benchmark = tmp_path / "bin" / "evalforge-benchmark-beir"
    benchmark.write_text(
        "#!/usr/bin/env python3\nimport pathlib, sys\n"
        "pathlib.Path(sys.argv[sys.argv.index('--output') + 1]).write_text('[]')\n",
        encoding="utf-8",
    )
    benchmark.chmod(0o755)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "Expected a JSON object" in completed.stderr
    assert (output / "manifest.json").exists()


def test_nonfinite_comparison_evidence_is_rejected(tmp_path):
    output, cmd, env = _setup(tmp_path)
    comparator = tmp_path / "bin" / "evalforge-compare-beir"
    comparator.write_text(
        "#!/usr/bin/env python3\nimport pathlib, sys\n"
        "pathlib.Path(sys.argv[sys.argv.index('--output') + 1]).write_text('{\\\"score\\\": NaN}')\n",
        encoding="utf-8",
    )
    comparator.chmod(0o755)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "Non-finite JSON constant" in completed.stderr
    assert (output / "scifact-ascii.json").exists()
    assert (output / "scifact-unicode.json").exists()


def test_duplicate_json_keys_are_rejected(tmp_path):
    output, cmd, env = _setup(tmp_path)
    benchmark = tmp_path / "bin" / "evalforge-benchmark-beir"
    benchmark.write_text(
        "#!/usr/bin/env python3\nimport pathlib, sys\n"
        "pathlib.Path(sys.argv[sys.argv.index('--output') + 1]).write_text('{\\\"score\\\": 1, \\\"score\\\": 0}')\n",
        encoding="utf-8",
    )
    benchmark.chmod(0o755)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "Duplicate JSON key" in completed.stderr


def test_runner_rejects_dataset_mutation_during_benchmark(tmp_path):
    output, cmd, env = _setup(tmp_path)
    benchmark = tmp_path / "bin" / "evalforge-benchmark-beir"
    benchmark.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "corpus = pathlib.Path(args[args.index('--corpus') + 1])\n"
        "corpus.write_text('modified during run\\n')\n"
        "pathlib.Path(args[args.index('--output') + 1]).write_text('{}')\n",
        encoding="utf-8",
    )
    benchmark.chmod(0o755)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "corpus input changed during evaluation" in completed.stderr
    assert (output / "manifest.json").exists()
    assert not (output / "scifact-comparison.json").exists()


def test_runner_rejects_dataset_mutation_during_comparison(tmp_path):
    output, cmd, env = _setup(tmp_path)
    comparator = tmp_path / "bin" / "evalforge-compare-beir"
    comparator.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "manifest = json.loads((pathlib.Path(args[0]).parent / 'manifest.json').read_text())\n"
        "corpus = pathlib.Path(sys.argv[0]).parent.parent / 'dataset' / 'corpus.jsonl'\n"
        "corpus.write_text('modified during comparison\\n')\n"
        "pathlib.Path(args[args.index('--output') + 1]).write_text('{}')\n",
        encoding="utf-8",
    )
    comparator.chmod(0o755)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "corpus input changed during evaluation" in completed.stderr
    assert (output / "scifact-ascii.json").exists()
    assert (output / "scifact-unicode.json").exists()


def test_custom_comparison_tolerance_is_recorded_and_forwarded(tmp_path):
    output, cmd, env = _setup(tmp_path)
    completed = subprocess.run(cmd + ["--max-drop", "0.125"], env=env,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["comparison_max_drop"] == 0.125
    comparison = json.loads((output / "scifact-comparison.json").read_text(encoding="utf-8"))["argv"]
    assert comparison[comparison.index("--max-drop") + 1] == "0.125"


def test_invalid_comparison_tolerance_is_rejected_before_writing(tmp_path):
    for value in ("-0.01", "1.01", "nan", "inf"):
        case = tmp_path / value.replace(".", "_").replace("-", "neg")
        case.mkdir()
        output, cmd, env = _setup(case)
        completed = subprocess.run(cmd + ["--max-drop", value], env=env,
                                   capture_output=True, text=True, check=False)
        assert completed.returncode == 2
        assert "--max-drop must be a finite number in [0, 1]" in completed.stderr
        assert list(output.iterdir()) == []


def test_benchmark_timeout_retains_manifest_and_stops_pipeline(tmp_path):
    output, cmd, env = _setup(tmp_path)
    benchmark = tmp_path / "bin" / "evalforge-benchmark-beir"
    benchmark.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(5)\n", encoding="utf-8"
    )
    benchmark.chmod(0o755)
    completed = subprocess.run(
        cmd + ["--timeout-seconds", "0.1"], env=env,
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert completed.returncode == 2
    assert "benchmark exceeded" in completed.stderr
    assert (output / "manifest.json").exists()
    assert not (output / "scifact-comparison.json").exists()


def test_comparison_timeout_retains_completed_benchmarks(tmp_path):
    output, cmd, env = _setup(tmp_path)
    comparator = tmp_path / "bin" / "evalforge-compare-beir"
    comparator.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(5)\n", encoding="utf-8"
    )
    comparator.chmod(0o755)
    completed = subprocess.run(
        cmd + ["--timeout-seconds", "0.2"], env=env,
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert completed.returncode == 2
    assert "Comparison exceeded" in completed.stderr
    assert (output / "scifact-ascii.json").exists()
    assert (output / "scifact-unicode.json").exists()
    assert (output / "manifest.json").exists()


def test_invalid_timeout_is_rejected_before_writing(tmp_path):
    for value in ("0", "-1", "nan", "inf"):
        case = tmp_path / value.replace("-", "neg")
        case.mkdir()
        output, cmd, env = _setup(case)
        completed = subprocess.run(
            cmd + ["--timeout-seconds", value], env=env,
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 2
        assert "--timeout-seconds must be a positive finite number" in completed.stderr
        assert list(output.iterdir()) == []


def test_runner_rejects_mismatched_report_provenance(tmp_path):
    output, cmd, env = _setup(tmp_path)
    benchmark = tmp_path / "bin" / "evalforge-benchmark-beir"
    benchmark.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "pathlib.Path(args[args.index('--output') + 1]).write_text("
        "json.dumps({'source': {'source_sha256': {'corpus': 'incorrect'}}}))\n",
        encoding="utf-8",
    )
    benchmark.chmod(0o755)
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "source SHA-256 does not match" in completed.stderr, completed.stderr
    assert (output / "manifest.json").exists()
    assert (output / "scifact-ascii.json").exists()
    assert not (output / "scifact-comparison.json").exists()
