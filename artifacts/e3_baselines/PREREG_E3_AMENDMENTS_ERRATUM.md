# Erratum to PREREG_E3_AMENDMENTS.md (append-only; the amendment file itself is NOT modified)

Written 2026-09-26 (KST), after E3 finished. This file only corrects a timestamp typo. It changes no method, grid, selection
rule, analysis rule, or result.

## Erratum E1 — header time of amendment A2

`code/PREREG_E3_AMENDMENTS.md` (sha256 `13d323cffaacdec008bb6b91bb98e01f62e93605e2bd4592210b3be068132806`, unchanged) has this
header for A2:

> `## A2 — 2026-09-26 00:45 KST (before any E3 computation on E1b adapters; ...)`

**"00:45 KST" is a typo. A2 was finalized and hashed at 00:35:54 KST.** The file was hash-locked in the state it is in now,
so the header cannot be corrected in place without breaking the hash.

Evidence (all times KST, 2026-09-26):

| evidence | location | time |
|---|---|---|
| mtime of the amendment file on the analysis box, where it was written | `/workspace/lora-paper/e3/PREREG_E3_AMENDMENTS.md` | 00:35:54.154 |
| mtime of the hash file on the box | `/workspace/lora-paper/e3/PREREG_E3_AMENDMENTS.sha256` | 00:35:54.154 |
| mtime of both files on the GPU machine (mtime kept on copy) | `artifacts/e3_baselines/code/PREREG_E3_AMENDMENTS.{md,sha256}` | 00:35:54 (inode created 00:36:15 by the copy) |
| hash file content | `code/PREREG_E3_AMENDMENTS.sha256` | `13d323cf…2806  PREREG_E3_AMENDMENTS.md`, which equals the current sha256 of the file |
| E3 launch record listing the same hash | `e1b_run/code_at_launch.sha256` (mtime 00:39:05.187), line 6: `13d323cffaacdec008bb6b91bb98e01f62e93605e2bd4592210b3be068132806  PREREG_E3_AMENDMENTS.md` | 00:39:05 |
| E3-on-E1b run start | `e1b_run/run.log` line 1: `2026-09-26 00:39:05 KST \| === e3.py ['e3.py', '--setting', 'e1b', '--stage', 'all', ...]` | 00:39:05 |
| first E3 result on E1b adapters | `e1b_run/run.log` line 2 (`cola-sst2: ...`) | 00:40:07 |

So the hashed A2 text existed at 00:35:54, 3 min 11 s before the E3 launch (00:39:05), and the launch recorded that exact
hash. A time of 00:45 would be after the launch and after the first E1b pair had been computed (00:40:07). That would contradict
both the header's own claim ("before any E3 computation on E1b adapters") and the launch record. The correct header time is
**00:35 KST** (to the minute; hash at 00:35:54).

## Erratum E2 — header time of amendment A1 (found while checking E1; same kind of typo)

The A1 header reads `2026-09-25 22:50 KST` and says A1 was written "before the pilot run". The pilot (`pilot_e1/run.log` line 1)
started at **2026-09-25 22:48:31 KST**, which is earlier than the header time. The A1 text is nevertheless older than the pilot:

- `e3_code.tgz` (sha256 `6e00d573…22d8`, same on box and GPU machine; mtime on the GPU machine 22:47:02.687) contains
  `PREREG_E3_AMENDMENTS.md` (931 bytes, mtime 2026-09-25 22:43:09, sha256 `4cc0e41e…7b05`), which holds only A1.
- Those 931 bytes are byte-identical to the first 931 bytes of the current hash-locked amendment file (checked with `cmp`).

So A1 existed in its final form by 22:43:09 KST, before the smoke_e1_v2 run (22:47:04) and the pilot (22:48:31). "22:50" is a
typo; the supported time is **≤ 22:43 KST**. Unlike A2, A1 was not hashed on its own before the pilot (the pilot
`code_at_launch.sha256` lists no amendment file). Its pre-pilot existence rests on the tarball mtime and content, which is weaker
evidence than a hash in a launch record.

The sha256 of this erratum is in `PREREG_E3_AMENDMENTS_ERRATUM.sha256`.
