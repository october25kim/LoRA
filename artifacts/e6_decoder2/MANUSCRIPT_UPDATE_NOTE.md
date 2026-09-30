# Manuscript update note — E6 Qwen2.5-1.5B integration (2026-09-29 KST)

## What changed

### `paper/MANUSCRIPT_v4.md` (source of truth; main text ~10,986 words)
- **Title / framing:** four backbones (two encoders + two Qwen decoders); date 2026-09-29.
- **Abstract, Intro, Contributions, “What we do not claim”:** second decoder E6 primary H1/H2 INCONCLUSIVE (weaker than 0.5B); unlabeled λ / merge-decision rules called out; exploratory vs confirmatory labelled.
- **§3 Study design:** E6 stage; Table 1 column for Qwen2.5-1.5B (196 LoRA layers; d_out 1536/256/8960; lr 3×10⁻⁴; K=14); disclosure that E6 was written after E4b/E5 with frozen analysis choices.
- **Table 2 / §4.2:** Qwen-1.5B rows; primary P O_A ρ=0.123 / tv_cos ρ=0.150, Holm p=0.2824, INCONCLUSIVE; gate inactive in P (min θ_min=31.7°); secondary S1/pooled PASS noted as descriptive only.
- **New §4.8:** exploratory e6_lambda insert (three prior backbones) + confirmatory E6 H-U1 / H-U1b / H-M3 SUPPORTED (H-M3acc not); explicit exploratory vs confirmatory labels.
- **Box 1 Steps 4–5:** unlabeled agreement for λ; λ≈0.7 no-data default; merge/no-merge via unlabeled disagreement; confirmatory support from 1.5B noted.
- **Discussion / FL / Limitations / Conclusion:** “one decoder” → two decoders; overlap still not PASS on primary P for 1.5B; FL paragraph mentions unlabeled λ / λ≈0.7.

### `paper/CLAIMS_v4.md`
- Added **C46–C52** (E6 primary INCONCLUSIVE; secondary descriptive; H-U1/H-U1b/H-M3; exploratory e6_lambda; process disclosure). Updated withdrawals / must-check notes.

### Venue rebuild (`paper/venues/`)
- Updated `venue_text.py` (ABSTRACT, INTRO_LEAD, CONTRIB, HIGHLIGHTS, IJMLC four-backbone sentence) and `build_venues.py` title / Step-4 clause matching.
- Ran `python3 build_venues.py all` → both `neurocomputing/main.pdf` and `ijmlc/main.pdf` rebuilt (pdflatex+bibtex rc 0; 2026-09-29 ~10:35 KST).
- Bare Unicode `τ` in §4.8 replaced with `$\tau$` so pdflatex succeeds.

### Provenance
- `python3 paper/trace_numbers_v4.py` → **0 unmatched** on `MANUSCRIPT_v4.md`, `venues/neurocomputing/main_body.md`, and `venues/ijmlc/main_body.md`.
- Evidence: `e6_decoder2/VERDICT_E6_DECODER2.md`, `e6_lambda/VERDICT_E6.md` / `MANUSCRIPT_INSERT.md` (already globbed by the tracer).

### Not done (per brief)
- No git push, no upload, no journal submit.
- Supplement v4 not expanded for E6 tables (main-text integration only).
