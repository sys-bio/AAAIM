# Release asset: Phase 3B selected epoch-1 inference model

The ZIP itself is intentionally gitignored. Commit the registry, restoration
record, these instructions, and the scientific fusion artifacts; upload only the
ZIP as a binary release asset after separate authorization.

| Field | Value |
| --- | --- |
| File | `aaaim-phase3b-selected-epoch1-inference.zip` |
| Compressed size | 111,163,286 bytes |
| Uncompressed size | 134,416,825 bytes |
| SHA-256 | `3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53` |
| Selected epoch | 1 |
| Initializer | `BAAI/bge-small-en-v1.5` |
| Immutable revision | `5e62ea33e012fda8c02802b906664c915ebd1bb1` |
| Source checkpoint SHA-256 | `8773b04f09916889b74c956e708044fe2c653fa764fe73c422ecc376ecae81c1` |
| Proposed release tag | `benchmark-phase3b-selected-epoch1` |

## Deterministic creation and verification

```powershell
benchmark/phase3/_phase3b_env/Scripts/python.exe -m benchmark.scripts.phase3b_release build-archive
benchmark/phase3/_phase3b_env/Scripts/python.exe -m benchmark.scripts.phase3b_release verify-archive
benchmark/phase3/_phase3b_env/Scripts/python.exe -m benchmark.scripts.phase3b_release restore --batch 64
```

Creation independently stages and builds the ZIP twice, fixing member ordering,
timestamps, permissions, compression method, and compression level. Both builds
must be byte-identical. The resulting ZIP must match the checksum above.

## Exact upload procedure (not executed)

After a separately authorized release/tag operation, run these exact commands from
the repository root. The first command refuses to upload altered bytes; the second
creates the proposed release and attaches only the ignored ZIP.

```powershell
$expected = "3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53"
$archive = "benchmark/dist/aaaim-phase3b-selected-epoch1-inference.zip"
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "refusing upload: SHA-256 $actual != $expected" }
gh release create benchmark-phase3b-selected-epoch1 $archive --title "AAAIM Phase 3B selected epoch-1 inference model" --notes-file benchmark/dist/RELEASE_phase3b-selected-epoch1.md
```

Then download the release asset into a fresh clone and follow
`benchmark/dist/RESTORE_phase3b-selected-epoch1.md`. Confirm the downloaded SHA-256
before extraction. Do not commit the ZIP, source checkpoint, or restored model
directory.
