# Run a real SciFact benchmark in GitHub Actions

The [Real SciFact benchmark workflow](../.github/workflows/real-scifact-benchmark.yml) supports manual runs, a weekly Monday schedule (09:17 UTC), and pushes that modify its workflow definition. It does not run on every source-code push or pull request, and it does not publish fabricated scores.

## Run it

The default uses the publisher-hosted BEIR ZIP. If that host is unavailable,
an operator may manually supply **both** a custom HTTPS `dataset_url` and
the independently established, 64-character hexadecimal `dataset_sha256`
in the **Run workflow** form. The download is rejected if its actual SHA-256
does not match the supplied digest. A custom URL without a checksum is
rejected; scheduled runs still use the official default.

**Important:** A matching checksum proves consistency with the checksum
you supplied, **not** that the alternative archive is the authentic or
equivalent SciFact dataset. Independently establish the expected digest,
verify the data source, licensing, test-split membership, and BEIR schema
before publishing comparative results. This option is not an automatic
mirror fallback and does not guarantee that a usable alternate source exists.

1. Open **Actions → Real SciFact benchmark (manual) → Run workflow** on the default branch.
2. Wait for the job to finish. It downloads the public BEIR SciFact archive from the TU Darmstadt BEIR dataset host, installs EvalForge from the checked-out commit, and executes both ASCII and Unicode BM25 baselines on the test split.
3. Open the run summary to inspect the measured comparison table. Download the **real-scifact-bm25-evidence** artifact for the JSON reports, input SHA-256 fingerprints, comparison decisions, and (on a completed verified run) `RESULTS.md`.
4. Reproduce locally with the documented `evalforge-run-scifact` command using the same source files and commit. The manifest records the source URL and download date; the separate archive digest records the downloaded ZIP's SHA-256.

**What the numbers mean:** This measures lexical BM25 retrieval on SciFact relevance judgments. It is not an embedding-model benchmark, generated-answer evaluation, hallucination assessment, or AI-agent safety result. The 0.02 comparison tolerance is illustrative, not a validated production threshold.

**Failures are informative:** A failed comparison quality gate is preserved as evidence and causes a red workflow result; a download, parsing, or verification error also causes a red result. The workflow retains partial artifacts for debugging, including `scifact-acquisition/dataset-acquisition.txt` when the source download fails. This file records the source URL, failed attempt counts and exception types, without pretending a failed download produced benchmark evidence. Do not describe any failed or incomplete run as a successful benchmark.

**Data and reproducibility:** The workflow fetches an upstream third-party dataset at execution time, so availability and upstream contents can change. It caps the download size, extracts only the three required SciFact files, and records an archive SHA-256; it does not pin or independently authenticate a publisher-signed dataset checksum. Review the upstream dataset's license and terms before redistribution. Artifacts expire after 14 days. To publish reproducible results long-term, preserve the exact source digests, EvalForge revision, measured report, and license-compatible dataset acquisition instructions.

**Status:** Adding this workflow is not evidence that a real SciFact run has executed. Only report actual measured scores after a completed run has produced and verified the evidence.
