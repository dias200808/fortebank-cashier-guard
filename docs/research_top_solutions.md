# Research: how winning teams solved analogous problems

Budget: 20 searches/fetches. Only URLs actually returned by search/fetch are cited. Several PDFs were unreadable via the fetch tool (binary), so for those I only have abstract-level or search-snippet facts. Items marked **[unverified]** come from secondary summaries or snippets.

**Coverage gaps (be honest):**
- Russian/CIS hackathons (item 4): one search plus one fetch found no write-up of a video-violation, cashier, or SOP solution. The one Habr article I fetched was a team-story with no video pipeline.
- Industrial SOP products (item 5): only patents were found, not vendor engineering write-ups.
- The AI City 2025 winner write-ups were not retrieved.

---

## 1. NVIDIA AI City Challenge

### 1.1 AI City 2024 Track 5, helmet rule violation (the closest "rule on a fixed CCTV scene" task)
- Links:
  - Overview: https://arxiv.org/html/2404.09432v1
  - UIT (1st) paper: https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Vo_Robust_Motorcycle_Helmet_Detection_in_Real-World_Scenarios_Using_Co-DETR_and_CVPRW_2024_paper.pdf (the page returned 403 for me; details are from the overview and the search snippet)
  - China Mobile (2nd): https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Chen_An_Effective_Method_for_Detecting_Violation_of_Helmet_Rule_for_CVPRW_2024_paper.html
  - 3rd place (VNPT): https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Van_Luong_Motorcyclist_Helmet_Violation_Detection_Framework_by_Leveraging_Robust_Ensemble_and_CVPRW_2024_paper.html
- **What won:** UIT 0.4860 mAP, China Mobile 0.4824, VNPT 0.4792, per the overview. This is a detection task: the "violation" is a fine-grained class label (no helmet, 1 or 2 riders, etc.).
- Pipeline:
  - Co-DETR (Swin-L) as the main detector. YOLOv7/v8 and DETA were used as ensemble members.
  - Weighted Box Fusion, NMS and test-time augmentation over detector outputs.
  - Class-imbalance handling: UIT's "Minority Optimizer" and virtual boxes with calibrated confidence to lift recall on rare classes. VNPT used confidence correction.
  - VNPT also added an explicit association module (person-motorbike-head) and tracking for vehicle direction, i.e. rules on top of detections.
- **Lesson:** Winners were detector-centric, with the rule expressed as a class or as geometric association between detections. A large strong detector, ensembling, and fixing rare-class recall mattered more than a clever architecture. Gaps between ranks 1-3 are about 0.007 mAP, so these are all variants of the same recipe.

### 1.2 AI City 2023/2024 Track 3, naturalistic driving action recognition with temporal localization (timecode output)
- Link: https://arxiv.org/html/2404.09432v1 (2024 overview). The 2023 overview is https://ar5iv.labs.arxiv.org/html/2304.07500 (search result only; the winner was not named in what I saw).
- **What won (2024):** TeleAI 0.8282 activity-overlap score. SKKU-AutoLab 0.7798 (5th), MCPRL 0.6080 (8th).
- Pipeline:
  - Clip-level classification with a fine-tuned video backbone, run in a sliding window.
  - Ensemble across views and models, then boundary fusion. TeleAI used an "Augmented Self-Mask Attention" module.
  - Multi-step post-processing to refine start and end times: thresholds, merging of adjacent candidates, removal of short segments.
  - A different team (DeepLocalization) used graph-based change-point detection for proposals and a Video-LLM only to classify them.
- **Lesson:** For "one timecode" output, the top teams all separated proposal/localization from classification and spent most of their effort on post-processing the boundaries. Temporal precision comes from fusion and smoothing, not from the backbone.

### 1.3 AI City 2024 Track 2, traffic safety description (VLM end-to-end)
- Link: https://arxiv.org/html/2404.09432v1
- **What won:** AliOpenTrek 33.43 (avg of 4 metrics), AIO_ISC 32.89, Lighthouse 32.30.
- Pipeline:
  - All VLM teams used CLIP ViT-L/14 as the vision encoder.
  - AliOpenTrek used LLaVA-1.6-34B / Qwen-VL / Video-LLaVA with global plus local views and a visual-prompt schema, and fine-tuned on the task's instruction data. Their RLHF attempt performed poorly.
  - Lighthouse used Vid2Seq-style temporal modeling with a T5 decoder.
  - HCMUS_AGAIN put multi-view information into a rule engine.
- **Lesson:** VLMs win when there is a lot of in-domain fine-tuning data (WTS has many captioned videos). We have about 40 clips, so the end-to-end VLM route is not available to us as a trained system. It is only usable zero-shot as a verifier/describer.

### 1.4 AI City 2025 Track 2 (traffic-safety VQA)
- Links: https://arxiv.org/html/2508.13564v1 (overview; I did not read the leaderboard), https://arxiv.org/html/2510.11907v1 (a 10th-place dual-model team).
- Facts seen: the 10th-place team paired VideoLLaMA (captioning) with Qwen2.5-VL (VQA), each trained separately, scoring 45.76. The track is multi-camera, with a partially held-out test set. Top teams' methods were not retrieved.
- **Lesson (weak evidence):** even 10th place used separate models for "describe" and "answer", i.e. task-specific heads rather than one general prompt.

---

## 2. Procedural mistake detection (Assembly101, HoloAssist, CaptainCook4D, EPIC-Tent, IndustReal)

### 2.1 Review: Vision-Based Mistake Analysis in Procedural Activities
- Link: https://arxiv.org/pdf/2510.19292 (binary PDF; I could not extract the text. Cited only as a pointer to the survey.)

### 2.2 PREGO and TI-PREGO (one-class, online, symbolic)
- Links: https://arxiv.org/pdf/2404.01933 (PREGO), https://arxiv.org/pdf/2411.02570 (TI-PREGO). Binary PDFs, so details below come from the search-result descriptions only.
- Idea: recognise the current step, then predict the next expected step from the history of previous steps, and flag a mistake when the observed step disagrees with the prediction. TI-PREGO uses an LLM with chain-of-thought and in-context learning for the next-step prediction. Training uses only correct procedures (one-class, no mistake labels). Built Assembly101-O and Epic-tent-O for evaluation.
- **Lesson:** the mistake is a symbolic check ("is this step allowed after those steps?") layered over a step recogniser. This maps directly to our procedural violations (counting after the client left, safe left open) if we first extract an event sequence and then apply an allowed-order check.

### 2.3 Differentiable Task Graph Learning
- Link: https://arxiv.org/pdf/2406.01486 (appeared in search; not read). Learns a task graph of step dependencies from sequences and flags missing/out-of-order steps. Same lesson as 2.2: a small explicit graph of procedures is learnable or hand-writable and needs few labels.

### 2.4 HoloAssist challenge, 2nd place
- Link: https://arxiv.org/abs/2506.06174 (abstract only fetched).
- Facts: online framework handling procedural errors (action order) and execution errors, with an LLM generating the explanatory feedback. Placed second on mistake detection. A search snippet reported a leaderboard of MR-CAS (CVPR25) 57.0 F1, this report's RGB-only method 55.0, GazeCompl 51.0, TSformer baseline 35.1 **[unverified, from search snippet]**.
- **Lesson:** best mistake-detection F1 on HoloAssist is only about 55-57. Even the winners are weak, so precision on open-ended error types is hard for everyone. Using an LLM for the explanation (not the decision) matches our "regulation citation" output requirement.

### 2.5 CaptainCook4D
- Link: https://arxiv.org/pdf/2312.14556 (appeared in search). Snippet: supervised per-step error classification baselines were weak (one reported row: recall 14.86, F1 24.26). Error types include order, timing, preparation, missing-step, measurement and technique errors **[from search snippet]**.
- **Lesson:** trained generic video models barely beat chance on rare errors. Do not expect to learn violation types from about 40 clips.

### 2.6 Newer zero-shot / VLM mistake detection (titles only)
- https://arxiv.org/pdf/2606.21579 ("The Unreasonable Effectiveness of VLMs for Zero-shot Procedural Mistake Detection"), https://arxiv.org/pdf/2512.03474 (action-effect modeling), https://arxiv.org/pdf/2503.22405 (AMNAR, multiple normal action representations), https://arxiv.org/pdf/2609.20638 (PROVIA, procedure state tracking). I saw titles and a few snippets only, not the methods.
- Title-level lesson only: state/effect tracking (what is the state of the world after the action: safe open or closed, cash present or absent) is the direction recent work takes, and zero-shot VLMs are claimed to be competitive.

---

## 3. Video anomaly detection, training-free or few-shot

| Method | Link | Pipeline | Numbers |
|---|---|---|---|
| LAVAD (CVPR 2024) | https://openaccess.thecvf.com/content/CVPR2024/html/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.html ; code lucazanella/lavad | Per-frame VLM captions; cross-modal similarity cleans captions; an LLM summarises temporal windows and scores anomaly; scores refined by VLM similarity | UCF-Crime AUC 80.28 **[unverified, from VERA summary]**; beats unsupervised and one-class methods |
| AnomalyRuler (ECCV 2024) | https://arxiv.org/html/2407.10299v2 | Induction: a VLM describes about 10 normal reference frames and an LLM writes normal/anomaly rules (3 prompt strategies, voted). Deduction: VLM describes each test frame, perception smoothing (exponential majority) over time, an LLM rechecks against the rules | AUC Ped2 97.9, Avenue 89.7, ShanghaiTech 85.2, UBnormal 71.9. CogVLM-17B + GPT-4 (induction) + Mistral-7B (deduction). More than 10 reference frames hurt (85.9 to 72.2 at n=20) |
| Holmes-VAD / Holmes-VAU | https://arxiv.org/pdf/2406.12235 , https://arxiv.org/pdf/2412.06171 | Instruction-tuned multimodal LLM with a weakly supervised anomaly scorer for frame selection, plus explanation | UCF-Crime AUC 89.51 (trained) **[unverified]** |
| VERA (CVPR 2025) | https://arxiv.org/abs/2412.01095 | Frozen VLM; guiding questions are the learnable parameters, optimised by a learner/optimizer VLM pair using coarse labels; segment scores fused with scene and temporal context | UCF-Crime AUC 86.55 vs Holmes-VAD (no tuning) 84.61, VADor 85.90, LAVAD 80.28 **[unverified, from search summary]** |

- Not found in my searches: VAD-R1 (no result). LATERN (https://arxiv.org/pdf/2605.15054) appeared as a 2026 test-time context-aware method; contents not read.
- **Lessons:**
  - The best training-free systems all look alike: caption or describe, then reason over text with an LLM, then smooth in time. The text layer is where rules live.
  - AnomalyRuler is the closest match to us. It takes about 10 normal frames from the target camera, turns them into readable rules, and this is explicitly meant to transfer across domains. It supports our new-camera-angle risk because normal reference frames can be swapped per camera.
  - Perception noise from the VLM is the failure mode; the cure used by everyone is temporal smoothing and rule voting.
  - Generic anomaly scores answer "something odd happened", not "which regulation". We need specified rules.

---

## 4. Russian/CIS hackathons

- Searched "Цифровой прорыв" video violations/PPE. Result: no solution write-up found.
- https://habr.com/ru/companies/leadersofdigital/articles/655767/ is a finalists' team story from Digital Breakthrough 2021. It had no video pipeline (IoT, web prototype, pitch).
- https://habr.com/ru/news/857484 (winners announcement of "Цифровой прорыв. Сезон: ИИ" 2024) appeared in search; I did not fetch it.
- https://www.business-gazeta.ru/article/531895 describes a Tatarstan video-analytics hackathon on city cameras (garbage/containers). No pipeline details.
- **Lesson:** none technical. Honest answer: I could not find a transferable CIS winner write-up within budget.

---

## 5. Industrial SOP / retail products

- Links (patents, https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7631808 and .../9218580 among the family).
- Publicly described design: an overhead camera over the cashier; analytics of hand motion, trajectory, item pick-up and drop; the video-derived event stream is **correlated with transaction/POS log data** to flag discrepancies (sweethearting); behaviour models per person help separate operator error from fraud.
- No vendor engineering write-ups for kitchen hygiene or SOP products were found.
- **Lesson:** production systems detect low-level events (hand enters zone, item moves, person present) and then run a rule engine that joins them with other state. They do not ask a model "is this a violation?". We have no POS log, but the same structure applies with the safe, desk and door as the zones.

---

## Common patterns among winners

- Decompose: perception (detect, track, describe, classify short clips) then a separate decision layer. Even the "end-to-end VLM" teams used separate models for describing and answering.
- Violation logic is expressed as rules or a class over structured outputs (association of detections, allowed-next-step checks, LLM-written text rules), not learned directly from few labels.
- Strong pretrained detector plus ensembling (WBF, TTA) beats architecture novelty on detection tasks; top-3 gaps were under 1% mAP.
- Temporal accuracy comes from post-processing: sliding windows, boundary fusion, merging, minimum durations, smoothing (exponential majority in AnomalyRuler).
- Rare-class recall is fixed with explicit tricks (minority optimiser, confidence correction), because violations are rare.
- Training-free methods work best when told what normal looks like for this camera (about 10 reference frames), and degrade when given too many.
- LLMs are used for rule writing and explanation, while the verdict is gated by perceptual evidence.
- Absolute performance on open-ended mistake detection is modest (best HoloAssist F1 about 55-57), so expect noise and design for precision.
- Training on few labels does not generalise to new error types (CaptainCook4D supervised baselines were weak).

## Implications for our pipeline

1. **Structured perception layer, local and cheap.** Per-frame (about 1-2 fps) person detection and tracking, plus detection of hand/cash/laptop/cup/door, plus zone masks (safe, desk, door, client side) drawn per camera. Output a timestamped event table: who is in which zone, door open/closed, safe open/closed, cash visible on desk, laptop state.
2. **Rule engine on the event table.** One explicit rule per violation type with a minimum duration and the regulation id attached. This gives type, precise timecode (rule firing time minus smoothing delay), and the citation for free. Unseen violation types are handled by a fallback below.
3. **Small VLM as describer/verifier, not judge.** Run a local 7B-class VLM (it must fit in 8 GB, quantised) on the candidate windows only (AnomalyRuler/LAVAD-style captions), asking narrow yes/no questions ("is the safe open?", "is cash on the desk?"). This also provides the evidence frame and makes the system tolerant to the new camera angle, since zones can be re-annotated and the VLM is angle-agnostic.
4. **Rule induction from normal clips for unseen types.** Use the AnomalyRuler idea: about 10 normal frames per camera, have a local LLM write "normal / not normal" rules, and apply them as a low-priority flag with explanation. Keep this separate from the high-precision rule engine.
5. **Temporal post-processing.** Smooth per-frame states, merge adjacent detections, drop events shorter than a minimum duration, and take the first stable frame as the timecode. Winners spent their effort here.
6. **Use the 40 labels only for calibration.** Tune thresholds, minimum durations and the verdict combination; do not train a model. Optimise precision first, since the best published mistake-detection scores are low.
7. **Budget.** A 145 s clip at 1-2 fps is 145-290 frames. Run the detector on all frames, the VLM only on a few windows. This fits within 5 minutes on one 8 GB GPU.
