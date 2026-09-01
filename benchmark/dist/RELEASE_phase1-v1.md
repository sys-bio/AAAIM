# Release asset: `benchmark-phase1-v1` SBML inputs

Checksums in `benchmark/manifest/model_registry.json` detect future BioModels drift but
cannot reconstruct the original bytes if upstream files change. This archive preserves
the exact inputs the Phase 1 benchmark was built from.

## Asset

| Field | Value |
| --- | --- |
| File | `aaaim-benchmark-phase1-v1-sbml-inputs.zip` |
| Contents | 75 SBML files (`benchmark/models/*.xml`), 63.3 MB uncompressed |
| Compressed size | 3.67 MB |
| SHA-256 | `320ad7ad97b11c0411374369fd932daf79ca2d255048fb08b9ccce5934505560` |
| Tag | `benchmark-phase1-v1` (commit `2129bc5`) |

The archive is not committed: `benchmark/models/` is gitignored and the zip is a build
artifact. Regenerate it byte-for-byte from a clean checkout with:

```powershell
python benchmark/scripts/download_biomodels.py
Compress-Archive -Path "benchmark/models/*.xml" `
  -DestinationPath "benchmark/dist/aaaim-benchmark-phase1-v1-sbml-inputs.zip" `
  -CompressionLevel Optimal
```

Note that zip archives embed timestamps, so the archive's own digest is not reproducible
even though its contents are. Verify the contents, not the zip: every extracted file must
match the `sha256` recorded for it in `benchmark/manifest/model_registry.json`.

## Upload

The `gh` CLI is not installed in the current environment, so this step is manual:

```bash
gh release create benchmark-phase1-v1 \
  benchmark/dist/aaaim-benchmark-phase1-v1-sbml-inputs.zip \
  --title "Benchmark Phase 1 v1 (frozen SBML inputs)" \
  --notes "74 models, 5,838 ground-truth reactions, 5,816 evaluable. Exact SBML inputs as downloaded from BioModels, verified against upstream SHA-256."
```

Without `gh`, attach the file to a new release for tag `benchmark-phase1-v1` through the
GitHub web UI.

## Verifying a restored archive

```powershell
python benchmark/scripts/verify_snapshot.py --archive benchmark/dist/aaaim-benchmark-phase1-v1-sbml-inputs.zip
```
