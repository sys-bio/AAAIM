# Release asset: `benchmark-phase2-v1` candidate-generation caches

The aggregate Phase 2 tables are committed. The per-model resumability caches that
produced them are not: regenerating them costs a ~3-day pass. This archive preserves
those caches so `--assemble-only` and `--check-reassembly` remain reproducible from a
fresh clone.

## Asset

| Field | Value |
| --- | --- |
| File | `aaaim-benchmark-phase2-v1-candidate-caches.zip` |
| Contents | 74 JSON files (`benchmark/data/_candidates_cache/*.json`) |
| Uncompressed size | 27.0 MB (26,996,311 bytes) |
| Compressed size | 0.68 MB (683,535 bytes) |
| SHA-256 (this zip) | `84d9b87b29ade56096f3d4ab114ec1ad86e9783f9759403e46c14a95f9157e21` |
| `config_id` | `86938b48ab88` |
| File count | 74 |
| Proposed tag | `benchmark-phase2-v1` |
| Candidate-generation commit | `dbf15d6` |
| Analysis/artifact commit | `b5065e0` |

The archive is not committed: `benchmark/data/_candidates_cache/` is gitignored and
`benchmark/dist/*.zip` is gitignored. Per-file SHA-256 digests live in
`benchmark/manifest/candidate_cache_registry.json`.

Zip archives embed timestamps, so the zip's own digest is not reproducible even though
its contents are. Verify the contents against the registry, not the zip bytes. The zip
SHA-256 above identifies the uploaded asset.

## Create

```powershell
python benchmark/scripts/freeze_phase2.py --pack-caches
```

## Restore and verify

```powershell
# Extract at the repository root so files land in benchmark/data/_candidates_cache/.
python -c "import zipfile; zipfile.ZipFile('benchmark/dist/aaaim-benchmark-phase2-v1-candidate-caches.zip').extractall('.')"
python benchmark/scripts/freeze_phase2.py --verify-caches

# Or restore into a temp tree, verify files, and reassemble without touching the working copy:
python benchmark/scripts/freeze_phase2.py --verify-cache-archive
```

`--verify-cache-archive` must report byte-identical `candidates.csv`,
`candidate_status.csv` and `candidate_generation_failures.csv` against the committed
tables.

## Upload

Attach the zip when creating the GitHub release for tag `benchmark-phase2-v1`. The `gh`
CLI is not assumed:

```bash
gh release create benchmark-phase2-v1 \
  benchmark/dist/aaaim-benchmark-phase2-v1-candidate-caches.zip \
  --title "Benchmark Phase 2 v1 (frozen candidates and caches)" \
  --notes "74 models, 5,816 evaluable reactions, 91,802 candidate rows, zero pipeline failures. Per-model generation caches so the 3-day pass need not be repeated."
```

Without `gh`, attach the file through the GitHub web UI. Do not commit the zip.
