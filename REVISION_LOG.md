# Revision Log — Innovation CUSUM (Bernoulli submission)

**Paper:** Martingale Prediction Innovations from Recurrent Networks for Sequential Change-Point Detection
(formerly: *Martingale Innovations from Contractive Recurrent Networks and Dimension-Robust Change-Point Detection*)
**Target journal:** Bernoulli (imsart `bj`, author-year)
**Last updated:** 2026-07-19

## Round 9 — Submission package (2026-07-19)

Built `P1_bernoulli_submission_2026_0719/` (3.3 MB, self-contained, untracked
local deliverable). Dated files: `P1_bernoulli_main_2026_0719` (26 pp — the file
to upload), `_supp_` (27 pp), `_cover_letter_` (2 pp), each with `.tex`/`.pdf`,
plus `.bbl`, the two `-xrefs.tex` snapshots, `P1_references.bib`,
`imsart.cls`/`.sty`/`-nameyear.bst`, the 11 referenced figures under
`code/figures/`, and `README_2026_0719.md`. The `\inputcompanionaux` companion
references were rewritten to the new filenames (main↔supp, verified pointing at
each other). Clean-room compiled: 0 undefined, 0 errors, 0 unresolved `??`;
**isolated compilation re-verified** (each file alone, companion `.aux` removed,
resolves via its snapshot). Note: figures are PNG; regenerate as PDF from `code/`
if production prefers vector artwork.

## Round 8 — Optimality overclaim fixed (2026-07-19, commit `9fbb18a`)

A math/stat review found the only substantive overclaim: three places labeled the
**general-case** delay optimality "first-order rate-optimal" / "asymptotically
optimal in Lorden's minimax sense". The detector's constant is the detection
efficiency $2\kappa\mu/\sigma^2$, which equals the KL information $I$ — and hence
gives first-order/exact optimality — **only** in the Gaussian efficient-score case
(Thm optimality (ii)); in general only **order**-optimality $O(\log\gamma)$ holds.
Fixed in the optimality-section intro, Thm optimality (i), and the Conclusion,
making them consistent with the abstract, intro, and `cor:tradeoff` remark, which
already said "minimax-optimal detection *rate* / order-optimal … exactly so in the
Gaussian case". Everything else verified sound — including a line-by-line check of
the Thm ARL renewal proof (supermartingale valid exactly to $\theta^\ast=2\kappa/\sigma^2$;
per-excursion bound by optional stopping + Fatou; geometric domination ⇒
$\E_0[\tau_h]\ge e^{\theta^\ast h}$ with unit constant).

## Round 7 — Final-check retitle (2026-07-19, commit `167a08d`)

Pre-submission final check surfaced a title-vs-body contradiction: the title
claimed "A Distribution-Free Foundation" while the body (Remark after
Cor. tradeoff) states the guarantee "is **not** distribution-free in the
assumption-free sense... it requires the... tail conditions of
Assumption (B1)–(B3)." Retitled across main, supplement, and cover letter:

  *Martingale Innovations from Contractive Recurrent Networks: A
   Distribution-Free Foundation for Sequential Monitoring*
  → **Martingale Innovations from Recurrent Networks for Sequential
     Change-Point Detection**

Drops the "Distribution-Free" overclaim and "Contractive" (which undersold the
architecture-agnostic tier 2); foregrounds change-point detection to route to
the sequential-detection community and separate from the forward/reverse
companion (arXiv:2606.08934). Running head → "Martingale Innovations for
Change-Point Detection". Also verified: cross-refs resolve live *and* isolated
(0 undefined, 0 `??`); all 11 figures present in `code/figures/`; assumptions
(B1)–(B3) consistent; no overfull boxes. Deferred (user, when ready): build the
clean dated submission folder; decide whether to add the forward/reverse
companion note to the paper body (currently cover-letter only).

---

---

## Authoritative files (submission pair)

| File | Role |
|------|------|
| `P1_bernoulli_20260701_Bernoulli_revised.tex/.pdf` | Main paper (26 pp) |
| `P1_bernoulli_supp_20260701_Bernoulli_revised.tex/.pdf` | Supplementary Material (25 pp) |
| `P1_bernoulli_cover_letter.tex/.pdf` | Cover letter (1 p) |
| `P1_references.bib` | Shared bibliography |
| `*-xrefs.tex` | Auto-generated cross-reference snapshots — upload with the source like a `.bbl` |

Older generations (`P1_bernoulli.tex`, `P1_bernoulli＿20260701.tex` — note the
fullwidth underscore — and `P1_bernoulli_supp_20260701.tex`) are superseded and
kept for reference only.

---

## Round 1 — Rebuild against the first external review (pre-2026-07)

The file `Revision suggestion 1` (an AI referee report on an earlier draft)
identified eleven problems, the most serious being: an inconsistent
$\mathcal{I}_t$ definition; a **false claim that the predictable component
converges to a deterministic mean-field fixed point**; treating $(h_t)$ as a
Markov chain under $\beta$-mixing inputs; a misstated contraction condition;
assuming LSTM contractivity; conflating hidden-state MDS with
prediction-residual MDS; and a non-rigorous locally stationary corollary.

The `20260701` generation addressed all eleven: $\mathcal{I}_t$ removed in
favour of prediction innovations $e_{t+1}$; the stationary causal solution
$H_t^*$ introduced with an explicit warning **not** to identify it with a
mean-field fixed point or tail projection; the Markov claim disowned in the
setup; contraction restated as $L_\phi\|W_h\|_{\mathrm{op}}<1$
(Assumption 2.4); the **two-tier architecture** introduced (contractive-RNN
theory for the basic RNN; architecture-agnostic CUSUM theory for the LSTM
detector); the forget-gate result demoted to an explicitly labelled heuristic
derivation in the supplement; and Corollary 2.14 restated as an approximate
block result for locally stationary arrays.

## Round 2 — Second external review and AI revision (2026-07-01, `.patch` on file)

An AI-produced revision (`P1_bernoulli_20260701_Bernoulli_revised.tex`, diff
in `P1_bernoulli_20260701_Bernoulli_revised.patch`) softened and repaired:

1. **Title and abstract**: dropped "Contractive" and "Dimension-Robust";
   "complete, distribution-free guarantee" → "model-free with respect to the
   dynamics, but not assumption-free"; dropped "matches correctly specified
   parametric detectors"; removed the "every false alarm carries a concrete
   operational cost" opener.
2. **New bridge Proposition 2.10** (`prop:rnn-resid-bridge`): residual drift
   $\le L_o\rho^t\Delta_0$ + stationary readout error — connects tier-one
   state stabilization to the tier-two residual condition. Inline proof.
3. **Assumption (B1)–(B4) → (B1)–(B3)**: the a.s. conditional residual-drift
   bound made primary (the old L² accuracy condition does not imply it and is
   now explicitly labelled a diagnostic, not a substitute).
4. **Delay theorem proof** conditioned on $\{\tau_h>\tau\}$ (repairs the
   optional-stopping step).
5. Calibration remark made honest (diagnostics are not a theorem-valid
   estimator of the essential supremum); Ljung–Box caveat; Cor 3.5 aggregate
   drift clarified.

## Round 3 — Reconciliation of the revised pair (2026-07-05, commit `437546a`)

The AI revision had not been reconciled with the supplement. Fixed:

- Main: stale "(B4)" reference → (B1); "implied by the *stronger* uniform
  bound" → "equivalent to" ($\mathbb{E}[\hat e_t\mid\mathcal F_{t-1}]
  = m_{t-1}-\hat m_{t-1}$ identically).
- Supplement (new `_Bernoulli_revised` copy; original untouched): five proof
  citations updated from (B1)/(B4) to the new (B1); Cor 3.5 proof rewritten
  to cite the aggregate a.s. drift bound instead of the removed L² item;
  Table B.1 gained the bridge row, "bias margin" → "drift margin".
- Cross-references regenerated in both directions (bridge insertion shifted
  `prop:whiteness` 2.10 → 2.11 and `cor:ls` 2.13 → 2.14).

## Round 4 — Cross-reference system (commits `714c9c9`, `b7a8ea6`)

Frozen inlined label blocks (which had silently gone stale) replaced. The
`xr` package is **forbidden by the imsart class** (hard error), so both files
now carry a `\inputcompanionaux` macro that (a) reads the companion's `.aux`
live at compile time — renumbering can never go stale — and (b) rewrites a
snapshot `\jobname-xrefs.tex` on every compile, used automatically when the
companion's `.aux` is absent (journal compiling one file in isolation).
No scripts to run; submission = upload the snapshot with the source, like a
`.bbl`. Verified: live pair clean, and each file compiles standalone with the
companion `.aux` removed.
(TeX note for the future: `\def` with `##` parameters must not sit inside
`\IfFileExists` branch arguments — machinery lives in named helper macros.)

## Round 5 — Full evaluation and fixes (commit `8000a2b`)

Complete read of the main paper and load-bearing supplement proofs; the
central chain (Thm 2.5 i–v, FCLT tightness via BDG, ARL supermartingale /
union bound with $\lambda^*=2\kappa/\sigma^2$, two-sided no-halving argument,
delay conditioning, bridge) verified sound. Fixes:

1. **MSC 2020 classifications added** (Bernoulli requirement): primary 62L10,
   60G42; secondary 62M10, 62M45.
2. **Related-work paragraph** added, positioning the paper between the
   residual-based monitoring tradition (Chu–Stinchcombe–White 1996;
   Bai 1994; Lai 1995; Aue–Horváth 2013) and conformal test martingales /
   e-detectors (Vovk et al. 2005; Shin–Ramdas–Rinaldo 2024); six references
   added.
3. **ARL₀ estimation disclosed and corrected to match the code**:
   Study CUSUM-DIM — mean over length-$5T{=}1000$ in-control sequences,
   alarm-free runs right-censored (conservative). Study ETF — the paper said
   "300 randomly drawn in-control subsequences of length 100"; the code that
   produced the table (verified against `results/S8_etf_detection.csv`)
   actually uses **500 block-bootstrapped score sequences of length 500
   (block length 10), right-censored**; text and caption corrected.
4. Study MDS "hidden dimension $d=32$" → $p=32$ (declared $p$/$d$ convention).
5. Remark 2.2 now states Lemma 2.1's scoping role explicitly.
6. Delay proof wording: (O1)–(O2) are a.s. conditions and survive
   conditioning — no extra assumption implied.
7. Billingsley citation aligned to the 1999 second edition (Theorem 13.5;
   15.6 was the 1968 numbering).

Two claims of the second review were verified **false** and not applied:
"missing norms" (a PDF-text-extraction artifact) and an Efron–Stein
$\sqrt{p}$ factor (not needed; concerns the companion RMRNN paper's
proposition in any case).

## Round 6 — Cover letter (commit `52e3c13`)

Rewritten to match the revised manuscript: new title; two-tier framing with
the bridge; stale overclaims removed ("L²-accuracy suffices", "first complete
distribution-free guarantee", "matches correctly specified parametric
detectors"); empirical paragraph mirrors the revised abstract (mechanisms,
not victory); positioning sentence vs the two neighbouring literatures; no
fragile proposition numbers. Companion-manuscript transparency note retained.

---

## Submission checklist

- [x] Main + supplement compile clean (0 errors, 0 undefined; 26 + 25 pp)
- [x] MSC codes render on page 1
- [x] Cross-references verified in live AND isolated compilation
- [x] All items of `Revision suggestion 1` addressed in the current text
- [x] ARL₀ estimation text matches the code and results CSV
- [x] Cover letter consistent with manuscript claims
- [x] Companion arXiv ID **2606.08934 verified** (2026-07-19): resolves to the
      correct companion, "Backward Coherence... A Quasi-Reverse-Martingale
      Theory" (Chang, submitted 2026-06-08)
- [x] Abstract length: **195 words** (already under the ~200 target; the earlier
      244-word count was pre-revision)
- [ ] **User**: final read of both PDFs (newest passages: intro related-work
      paragraph, Assumption (B1), delay proof, the two ARL₀ passages)

**Upload**: main PDF (+ `.tex`, `.bbl`, `-xrefs.tex` if source requested),
supplement PDF (+ likewise), cover letter PDF.

## Remaining housekeeping (not blockers)

- Pre-`20260701` files (`P1_bernoulli.tex`, `P1_bernoulli_supp.tex`, PDFs)
  carry old uncommitted modifications — superseded; sweep or archive.
- Stale legacy `P1_main_xrefs.tex` / `P1_supp_xrefs.tex` are unused since
  Round 4.
