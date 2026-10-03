# Full-paper data-contract tests

These tests cover the frozen clean-QA fingerprints, complete CounselBench-100
exclusion, question/response/cluster/source-group isolation, deterministic and
balanced corruption assignment, prompt constraints, unthresholded QC semantics,
the generator record contract, pilot stratification, and non-blocking optional
adapters.

The suite also verifies the separate dev120 corrected-export accounting, the
immutable original artifact hashes, the observed source/candidate integrity
checks, and exact source-offset span supervision.

Run from the repository root:

```bash
python -m pytest -q data_contract_tests
```
