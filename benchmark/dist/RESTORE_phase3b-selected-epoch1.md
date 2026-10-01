# Restore the Phase 3B selected epoch-1 inference model

Expected asset: `aaaim-phase3b-selected-epoch1-inference.zip`

Expected SHA-256:
`3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53`

The archive is a self-contained local Hugging Face model. It has
`model.safetensors`, configuration and tokenizer files, pooling/normalization and
formatting metadata, an inference fixture, environment records, loading guidance,
and per-payload SHA-256 values. It has no optimizer, scheduler, gradient scaler,
training cursor, source checkpoint, or model-cache dependency.

## Verify and restore

Place the downloaded asset at `benchmark/dist/aaaim-phase3b-selected-epoch1-inference.zip`,
then run from the repository root:

```powershell
$expected = "3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53"
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath "benchmark/dist/aaaim-phase3b-selected-epoch1-inference.zip").Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "archive SHA-256 mismatch: $actual" }

benchmark/phase3/_phase3b_env/Scripts/python.exe -m benchmark.scripts.phase3b_release verify-archive
benchmark/phase3/_phase3b_env/Scripts/python.exe -m benchmark.scripts.phase3b_release restore --batch 64
```

The second command extracts into a fresh temporary directory, forces Hugging Face
offline mode, loads only from the extracted tree, regenerates the fixed CPU
fixture, and regenerates all 969 validation Top-100 rankings over the frozen
12,312-reaction catalog. Ranked KEGG IDs must exactly equal
`benchmark/phase3/phase3b_full/rankings_epoch_1.jsonl`.

To load without the repository-specific scientific check:

```powershell
New-Item -ItemType Directory -Force -Path "benchmark/phase3/_restored_epoch1" | Out-Null
python -c "import zipfile; zipfile.ZipFile(r'benchmark/dist/aaaim-phase3b-selected-epoch1-inference.zip').extractall(r'benchmark/phase3/_restored_epoch1')"
python -c "from transformers import AutoModel,AutoTokenizer; p=r'benchmark/phase3/_restored_epoch1'; AutoTokenizer.from_pretrained(p,local_files_only=True); AutoModel.from_pretrained(p,local_files_only=True); print('local load passed')"
```

The extraction directory begins with `_` and is gitignored by the Phase 3 policy.
See the archive's `README.md` and `inference_config.json` for the exact CLS pooling,
L2 normalization, maximum length 256, query/document templates, and tie-breaking.
