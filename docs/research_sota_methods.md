# Research: accurate "cheap proposal + VLM verification" methods for CCTV rule-violation detection

Date: 2026-10-17. Budget: 20 searches/fetches. Only URLs actually returned by a search or fetch are cited.
Tags: [V] = number read from the paper text/page in this session; [S] = secondary summary or a number quoted by a different paper (treat as approximate); [U] = not verified / my inference.

Our setting recap: overhead CCTV, 464x848, clips of 145 s or less, about 40 labeled clips, local open VLM of 8B or smaller on 8 GB, false negatives unacceptable, false positives minimal, timecode within about 1 s, new camera angle and unseen types at test time.

---

## Q1. Candidate / proposal generation (keyframes, event proposals, agentic search)

### Findings
- **Keyframe selection beats uniform sampling, but only by a few points.** Training-free methods typically add about 1 to 4 points over uniform sampling [S]:
  - AdaQ: +3.1 (LLaVA-Video-7B) and +4.0 (Qwen3-VL-8B) on MLVU vs uniform [S] - https://arxiv.org/pdf/2606.24187
  - AdaRD-Key: Qwen2-VL on LongVideoBench 60.8% at 32 frames, only +0.3 over AKS (CVPR 2025) [S] - https://arxiv.org/pdf/2510.02778 ; AKS: https://cvpr.thecvf.com/virtual/2025/poster/34055
  - Q-Frame: VILA-1.5 MLVU 46.3 -> 54.4 at 8 frames [S, via alphaxiv] - https://alphaxiv.org/overview/2506.22139v3
  - These are query-relevance selectors (CLIP-style). They help most when the video is long and the frame budget is tight. Our clips are at most 145 s, so 1 fps is only about 145 frames. Frame selection is not our bottleneck; precision of the verdict is.
- **Agentic search (Deep Video Discovery, DVD, NeurIPS 2025)** reaches 74.2% on LVBench (76.0% with transcripts) vs 57.1% for o3 with 256 frames, 48.9% GPT-4o, and 28.8% VideoTree / 29.3% VideoAgent as listed in its table [V] - https://arxiv.org/html/2505.18079v4 (also https://neurips.cc/virtual/2025/poster/116039).
  - Ablation [V]: removing Clip Search costs 12.3 points, Frame Inspect 8.4, Global Browse 2.9.
  - The reasoning LLM drives the result [V]: o3 76.0, DeepSeek-R1 68.5, GPT-4o 62.3, Qwen3-32B thinking 57.3. Frame-inspection with GPT-4.1-mini instead of o3 drops 76.0 to 72.3.
  - Implication: agentic search gains rely on a frontier-class planner. An 8B local model is not a credible agent controller. Do not build a DVD-style agent locally.
  - The paper has no uniform-sampling ablation of its own, so "how much search helps" is only inferable from the o3 baseline row.
- **Older agent/tree methods** (original papers): VideoAgent 54.1% EgoSchema and 71.3% NExT-QA using about 8 frames on average (https://arxiv.org/abs/2403.10517v1) [S]; VideoTree 61.1% EgoSchema test, +7.0 over baseline, 40% less inference time (https://arxiv.org/pdf/2405.19209) [S]. Both rely on GPT-4-class LLMs and captioners.
- **Cascades for surveillance specifically: Cerberus** (https://arxiv.org/html/2510.16290v1) [V]:
  - Stage 1 is a CLIP-based filter plus frame-differencing motion gating (motion proportion threshold, optimum 7e-4; red-circle/square motion-mask overlays on the frame). Stage 2 is Qwen2.5-VL-7B for fine reasoning.
  - Coarse filter top-k=5 keeps over 95% of anomaly recall while discarding over 50% of ShanghaiTech frames.
  - Cascade: 82.73 AUC at 0.23 h vs VLM-only 84.24 AUC at 0.35 h (original proportion); at 1% anomalies the VLM-only run takes 14.63 h vs 0.69 h for the cascade. About 97% of the relative AUC at 150x the speed of AnomalyRuler.
  - Coarse filtering alone gives only 67.85 AUC. The cheap stage must be recall-oriented, and the VLM must decide.
  - Also relevant: a zero-shot CCTV pipeline with YOLO11 + BoT-SORT tracking + Qwen2.5-VL (https://arxiv.org/pdf/2608.08867) [only the abstract-level description seen; numbers not read, U].

### Recommendation
- Use domain proposals (person/door/zone events from a detector + tracker, plus frame-difference motion in fixed ROIs), tuned for near-100% recall on the 40 labeled clips. Over-generate, because the VLM stage removes false positives.
- Because clips are short, also run an exhaustive 1 Hz scan of cheap state questions. This removes proposal-miss risk, which matters since false negatives are unacceptable. Pure motion proposals will miss static violations (cash left on desk, safe left open while nobody moves).
- Skip LLM-agent frameworks (VideoAgent/VideoTree/DVD). They need o3/GPT-4-class planners and show weak numbers with small ones.

---

## Q2. Temporal grounding of small open VLMs: can they give timecodes within about 1 s?

### Findings
- **Direct timestamp output is poor.**
  - Qwen2.5-VL-7B on the original Charades-STA: R1@0.3/0.5/0.7 = 59.4/38.2/18.1, mIoU 43.6 [V] (TimeLens table 7, https://arxiv.org/pdf/2512.14698). On the re-annotated Charades-TimeLens: 59.7/37.8/16.6, mIoU 39.3 [V].
  - Qwen3-VL-8B on Charades-TimeLens: 69.2/53.4/27.5, mIoU 48.3 [V]. Qwen3-VL-235B: mIoU 47.8, no better than the 8B [V]. ActivityNet-TimeLens: Qwen3-VL-8B mIoU 46.8; Qwen2.5-VL-7B 31.4 [V].
  - Training helps: TimeLens-8B (RLVR on Qwen3-VL-8B) reaches 76.6/63.0/35.2, mIoU 55.2 on Charades-TimeLens [V]. TimeLens notes legacy benchmarks have noisy labels (about 20% of Charades-STA samples violate query clarity checks) and that rankings change on re-annotation [V].
  - Qwen3-VL replaced T-RoPE with explicit textual timestamps ("<3.0 seconds>") [S] (https://www.themoonlight.io/es/review/qwen3-vl-technical-report); the abstract (https://arxiv.org/abs/2511.21631) lists dense 2B/4B/8B/32B models but no Charades number was visible there [V]. I did not find InternVL3 Charades-STA numbers [U].
  - Context: Charades videos are about 30 s and the events are 8 s long; mIoU of about 48 means errors of seconds. At 145 s clips and events that are instantaneous (door opens, cash appears), expect worse. Not sufficient for a 1 s tolerance.
- **Asking "yes/no per clip" and scanning is much better than asking for timestamps** (FV-Action, https://arxiv.org/pdf/2608.08315) [V]:
  - Same frozen weights: InternVL2-8B 3.76 -> 53.39 R@0.5, Qwen2.5-VL-7B 28.41 -> 56.77 on Charades-STA. Training-free SOTA 56.8% R@0.5, vs the previous training-free best 40.0%.
  - 77 to 80% of wrong timestamp predictions are low-entropy (the model is confidently wrong); entropy-based error detection has F1 below 0.36, below a random classifier. So do not use token confidence to trust the timestamps the VLM emits.
  - Recipe: one fixed prompt "Does this clip show {query}? Answer Yes or No.", 5 frames per probe, 448x448, probes every 2.5 s (Charades) to 9 s (150 s videos) on a coarse scan, then a fine scan around the top peaks, and only the ranking of P(Yes) first-token probabilities is used (not calibrated values).
  - Caveat from the paper itself: output window width limits IoU, and the located peak lands inside the event for up to 95% of cases even where IoU fails. For us, "peak inside the event" is what counts.
- NumPro (CVPR 2025) overlays a frame number on every frame so video LLMs can read time like manga panels (https://arxiv.org/abs/2411.10332v3, https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Number_it_Temporal_Grounding_Videos_like_Flipping_Manga_CVPR_2025_paper.html) [S]: "significantly boosts" training-free grounding; no numbers read [U]. Applies to a pure-VLM fallback.

### Recommendation
- Timecodes must come from the detector/tracker and from a 1 Hz binary-probe scan, not from VLM-generated timestamps. VLM judges "is state X true in this frame/clip", the code produces the time.
- Refine boundaries by probing at 1 s steps (or 0.5 s) around each candidate and taking the first sustained rise (hysteresis), using P(Yes) as a ranking. Direct timestamp generation is a last-resort fallback.

---

## Q3. Reducing hallucination / false positives in verification

### Findings (the evidence is thinner than the question; several items are practitioner knowledge and unverified)
- **Binary probes are the reliable interface** (POPE-style yes/no): FV-Action shows the ground-truth clip wins 80.5% of paired comparisons for InternVL2-8B (AUROC 0.782), and P(Yes) works as a ranking even when poorly calibrated [V] (https://arxiv.org/pdf/2608.08315). Implication: choose thresholds per scene or per type on validation data, compare against that video's own baseline (contrast Yes-probability vs other windows of the same clip), and do not trust absolute probabilities.
- **Direct, un-decomposed questioning of an LLM has very low recall.** On ShanghaiTech, "ask LLM directly" gets accuracy 52.1, precision 97.1, recall 6.2; with an explicit rule procedure (AnomalyRuler) recall 64.3 at precision 90.2 [V] (https://arxiv.org/html/2407.10299v2). Given our "false negatives unacceptable" constraint, this is the main warning: an open "is there a violation?" prompt is conservative and misses cases. Decompose into explicit per-rule/state questions.
- **Rule/state decomposition and perception smoothing help**: removing perception smoothing costs 3.3 AUC, removing rule aggregation 15.8 (AnomalyRuler, ShT) [V].
- **Few-shot examples are double-edged**: a surveillance evaluation of small open VLMs found few-shot examples help some models but add false positives (https://arxiv.org/html/2510.23190v1, via a search summary) [S]. Do not rely on in-context image examples without A/B testing on the 40 clips.
- **Raw token log-probs are miscalibrated** in VLMs (HalLoc, https://arxiv.org/pdf/2506.10286) [S]; hence fit thresholds on validation data and prefer ranks. VideoHEDGE reports sampled-output clustering beats other uncertainty scores for video-VLM reliability (https://arxiv.org/abs/2601.08557v1) [S].
- **Zoom/crop**: ZoomEye (training-free tree search over zoomed crops) improves LLaVA-v1.5-7B by +34.57 on V* and +17.88 on HR-Bench; Qwen2.5-VL-3B reaches 68.38% on HR-Bench 8K [S] (https://arxiv.org/html/2411.16044v3, https://alphaxiv.org/overview/2411.16044v4). I did not see a V* number for Qwen2.5-VL-7B [U]. Relevant to us because the frame is only 464x848 and objects (cash, laptop screen) are small; cropping an ROI at native resolution and upscaling is the cheap version of this.
- **Self-consistency / multi-subset voting**: I found no paper that quantifies frame-subset voting for yes/no video verification [U]. Treat as an engineering choice and calibrate the vote threshold on validation clips.
- **Chain-of-thought / describe-then-decide**: no direct evidence found for small VLMs on verification [U]. Related: LAVAD (caption then LLM scoring) reaches 80.28 AUC on UCF-Crime [S] (https://arxiv.org/html/2404.01014v1; number from https://arxiv.org/pdf/2511.17094), but caption-only reasoning is weak for localization (LLM4VG-style caption reasoning gets 11.8 R@0.5 on Charades-STA [V], quoted in FV-Action).

### Recommendation
- Per violation type, ask 1 to 3 narrow, physically observable binary questions on tight crops ("Is the safe door open?", "Is there paper cash visible on the desk surface?", "Is a person seated at the cashier position?"). Read P(Yes) from first-token logits.
- Calibrate: choose thresholds on the 40 labeled clips to hit recall target first (e.g. 100% of train violations), then minimize false positives. Make threshold relative to the clip's own distribution where the camera is new.
- Use asymmetric voting: recall-critical states OR across 2-3 frame subsets/crops; require persistence (state true for at least N consecutive probes) to cut false positives.
- Use an open "describe what is happening / other" call only on residual unexplained motion segments to produce the "other" class, not as the main verdict.

---

## Q4. Rule-based reasoning for anomaly detection (AnomalyRuler, VERA, LAVAD)

### Findings
- **AnomalyRuler** (ECCV 2024, https://arxiv.org/html/2407.10299v2) [V]:
  - Induction: a VLM (CogVLM-17B) describes a few normal frames; GPT-4 writes normal and anomaly rules (strategies: Normal & Anomaly, Abstract & Concrete, Human & Environment); rules from 10 batches are aggregated by voting. Deduction: Mistral-7B-Instruct-v0.2 applies the rules per frame.
  - AUC (one-class, image-only): Ped2 97.9, Avenue 89.7, ShanghaiTech 85.2, UBnormal 71.9. Rule-based reasoning "doubly-right" 99% vs 73% for ask-GPT-4-directly (no perception errors); with perception errors 83% vs 57%.
  - Ablation: dropping "Normal and Anomaly" rules -18.8 AUC, dropping rule aggregation -15.8, dropping perception smoothing -3.3, dropping Human & Environment -2.4.
  - Cost: 0.38 fps (Cerberus table), far too slow for naive use, but our clips are short and offline.
- **VERA** (CVPR 2025, https://arxiv.org/pdf/2412.01095): learns the guiding questions as text parameters with a learner and an optimizer VLM, using coarse video-level labels; the VLM stays frozen; segment scores are fused with scene and temporal context. Used InternVL2-8B in the case study [S]. I did not see AUC numbers [U]. Borrowable: iterate question wording against the 40 labeled clips with an LLM proposing rewrites, scoring each by recall/precision.
- **LAVAD** (CVPR 2024): caption every frame, clean captions via cross-modal similarity, LLM scores anomalies; 80.28 AUC on UCF-Crime [S] (https://arxiv.org/html/2404.01014v1). Frame-level anomaly scores, not typed events with timecodes; low relevance beyond the caption-cleaning idea.
- **Cerberus** adds rule-based deviation scoring on top (recall 27.13 -> 48.24, AUC 77.93 -> 82.73 on SHTech) [V]; its ablation shows removing action captioning costs 18.56 AUC [V].

### Recommendation
- Our rules are given by the task spec, so skip rule induction and write them as explicit predicates (state questions) plus a small rule engine over the state timeline (sequence/ordering checks for procedural errors).
- Borrow: (1) the rule decomposition itself, (2) aggregation by voting across prompts/crops, (3) perception smoothing over time, (4) VERA-style prompt optimization on the labeled clips, (5) an "other" detector based on deviation from the clip's normal scene description.
- For a new camera angle, use scene-specific ROI re-localisation and a short normal-baseline per clip (first N seconds as reference), analogous to AnomalyRuler's few normal frames, rather than fixed pixel coordinates.

---

## Q5. Combining detector/tracker evidence with VLM judgement; Set-of-Mark (SoM)

### Findings
- SoM overlays numbered/lettered marks on segmented regions; it gave GPT-4V large grounding gains (https://arxiv.org/pdf/2310.11441) [S].
- **Small open VLMs do not reliably read marks out of the box**: "List Items One by One" reports other MLLMs struggle with visual tags, and fine-tuning on 10k-30k tagged images fixes it (https://arxiv.org/html/2404.16375v2) [S].
- **Marker design matters greatly**: marker size/shape/color/position swings Qwen2.5-VL accuracy by up to 13%, and a slightly larger marker pushes InternVL3-8B alongside Gemini 2.5 Pro on visually prompted benchmarks (https://arxiv.org/html/2512.17875v2) [S].
- No source I found tests numbered boxes on Qwen2.5-VL or InternVL3 against unmarked images for a verification task [U]. Cerberus uses motion-mask overlays (red circle for subtle motion, red square for large subjects) with Qwen2.5-VL-7B and reports it as a contributing factor [S] (https://arxiv.org/html/2510.16290v1), so simple single-color marks on the moving object are plausible; numbered multi-object marks are not demonstrated.

### Recommendation
- Prefer crops and text over numbered overlays: crop the ROI (safe, desk, door) or the tracked person and pass tracker facts as text ("person A at cashier seat 0 to 38.0 s, left at 38.0 s; door open since 51.2 s"). Use the VLM to judge the physical state, not to bind track IDs.
- If using marks, use one thick, high-contrast box and A/B it on the 40 clips against no mark; do not assume benefit.

---

## Recommended verification recipe

1. **Per-clip setup (new angle safe).** Decode at 1 fps (and 4 fps near candidates). Locate ROIs for door, cashier seat, desk, safe, laptop on the first frames with the VLM's grounding or a detector, stored as boxes per clip, never as hard-coded pixels.
2. **Cheap proposals, tuned for recall.** Person detector + tracker (YOLO-class + BoT-SORT/ByteTrack): enter/exit events, counts, dwell in zones; frame-difference motion inside each ROI (Cerberus-style gating); plus an exhaustive 1 Hz state scan (step 3). Union of all proposals.
3. **State timeline with binary probes.** For each ROI crop and each second, ask fixed Yes/No questions (safe open? cash visible? cashier present? laptop screen unlocked? door open? person not in staff set? personal item/cup present?). Use 3 to 5 frames per probe, crop upscaled to 448x448 or higher. Read first-token P(Yes).
4. **Smooth and threshold.** Median/hysteresis smoothing over time (AnomalyRuler perception smoothing). Thresholds fitted on the 40 labeled clips, recall first, then precision; use per-clip relative thresholds for the new camera angle.
5. **Rule engine over the timeline.** Violations as logic over states, e.g. cashier absent AND safe open AND/OR cash on desk for at least T seconds; next person enters while cash visible; door open for at least T seconds; outsider present; ordering errors as sequence checks. Output type from the closed list.
6. **Timecodes from code.** Violation start = first sustained state change (from tracker or probe rise), refined by 0.5 to 1 s probes. Evidence frame = highest-P(Yes) frame in the window. Do not ask the VLM for timestamps (Qwen2.5-VL-7B direct: mIoU about 39 to 44; Qwen3-VL-8B about 48; far from 1 s).
7. **Verification voting.** For each proposed violation, re-ask the type-specific question on 2-3 different frame subsets/crops; accept if the vote passes the validated threshold. Use OR-voting only for types where misses are costly, AND-voting plus persistence otherwise.
8. **"Other" class.** For motion or tracker anomalies not explained by any rule, run one open describe-then-decide call with a closed answer list plus "other"; require a higher bar before reporting.
9. **Prompt tuning on labels (VERA-style).** Use an LLM to propose rewrites of the probe questions; keep those that improve recall and precision on the 40 clips; hold out a few clips to avoid overfitting to the camera.
10. **Hold-out check for domain shift.** Evaluate on clips from the most different camera setup you have, and report recall per type; keep a small review queue for low-margin cases rather than silently dropping them.

## Unverified / gaps (be careful)
- No source found for: frame-subset voting gains, describe-then-decide CoT for small-VLM verification, numbered-box SoM on Qwen/InternVL3 with a baseline, ZoomEye V* number for Qwen2.5-VL-7B, InternVL3 Charades-STA, Qwen3-VL-4B grounding numbers, VERA AUCs.
- Items tagged [S] come from search-result summaries or numbers quoted by other papers and should be rechecked before citing externally.
- Steps 3 to 7 (state timeline + rule engine + probe-rise timestamps) are my synthesis of the cited findings, not a published, benchmarked pipeline [U].
- The 2608.* / 2606.* / 2607.* arXiv IDs came from search results dated after my training data; I read them as returned.
