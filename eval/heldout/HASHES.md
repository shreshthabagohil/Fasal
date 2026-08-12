# Held-out set hashes

Run `python eval/heldout/build_test_set.py` to generate `eval/heldout/test.jsonl` from
`dataset/gold/*.jsonl` (filtered per `eval/heldout/LEAKAGE_AUDIT.md`). The script prints
the row count and content hash on completion; `test.jsonl` itself is a build artifact
and is not committed, matching how `dataset/kcc_instruct_v1.jsonl` is also gitignored
and regenerated from `data/clean_kcc.py`.

Expected: 227 rows, hash=0d1f096cc842 (as of 2026-08-12, against
Algo-Nova/fasal-kcc-instruct commit 3d1fa91 / dataset/VERSION
5741495f65f43cef14ae1e5a44e2f54b18560dccdbae94613deb6c33c3441912).
