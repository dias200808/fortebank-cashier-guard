# Similar cases and reusable approaches

Research notes for the cashier-room CCTV violation detector (fully local, commercial-use licenses only).
Method: ~20 web searches/fetches, 2026-10-17. Only URLs actually returned by search/fetch are listed. "License (verified)" = seen on the repo page or an official-ish page; "unverified" = from memory or a third-party aggregator, so check the LICENSE file and the HF model card before relying on it.

Honest headline: we found NO public hackathon/Kaggle/DrivenData task for bank-teller or cash-discipline violation detection. The task is close to a niche of (a) retail "sweethearting" research, (b) training-free / rule-based VLM anomaly detection, (c) classic zone/abandoned-object analytics. The competition space (item 1) is therefore weak; use the adjacent literature.

---

## 1. Prior competitions / hackathons / similar tasks

| Name | Link | What it is / what we borrow |
|---|---|---|
| Search for CIS hackathons on teller/cashier violation detection | (no direct hit) | A Russian-language search ("видеоаналитика банк нарушения кассир ... хакатон") returned only retail articles and an unrelated "Цифровой прорыв" regional hackathon for RZD (railways). No evidence of a prior Sber/Kaspi/Halyk/T-Bank/Alfa/VTB competition on this. Do not claim in the pitch that this is a known benchmark. |
| Retail cashier-fraud video analytics (RU) - cleverence.ru | https://www.cleverence.ru/articles/roznichnaya-torgovlya/-zashchita-ot-poter-na-kassah-samoobsluzhivaniya-videoanalitika/ | Industry article on loss prevention at (self-)checkouts. Borrow the violation taxonomy (voided receipt, fake return, cash drawer opened without sale) as vocabulary for our rule list. |
| Cashier-error detection (RU) - new-retail.ru | https://new-retail.ru/business/kak_bystro_obnaruzhit_oshibki_kassirov_i_ne_poteryat_pribyl9887 | Describes the classic approach: join camera footage with POS event data. For us: no POS log, so visual-only; note it as a limitation and future integration (event timestamps from the cash machine/AWS). |
| Bobbitt, Connell et al. (IBM), "Detecting sweethearting in retail surveillance videos" (WACV 2011) | https://researcher.ibm.com/publications/detecting-sweethearting-in-retail-surveillance-videos | Closest classic academic analogue to ours. Motion-based candidate segmentation at the POS, spatio-temporal features to filter false alarms, then validate a whole transaction by combining primitive events under temporal-ordering constraints. Borrow: "primitive events + temporal ordering rules" architecture, i.e. our rule engine over detected states. No code/license. |
| Sweethearting patent family "Method and apparatus for detecting suspicious activity using video analysis" (StopLift) | https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7631808 (others in the family, e.g. /8132725, /9202117) | Patents: describe covering UPC with hand, moving item out of scanner range. Useful as a source of how vendors define event primitives; also a freedom-to-operate caution for the pitch (do not copy claims). |

## 2. Academic papers (2020-2026)

| Name | Link | What it is / what we borrow | Code / license |
|---|---|---|---|
| LAVAD - "Harnessing LLMs for Training-free Video Anomaly Detection" (CVPR 2024) | https://arxiv.org/html/2404.01014v1 ; project page https://lucazanella.github.io/lavad | Training-free: VLM captions each frame, noisy captions cleaned by cross-modal similarity, LLM summarizes a temporal window and scores anomaly, scores refined by aggregating similar-summary frames. Borrow: caption-then-reason pipeline that works with zero labels and is robust to camera-angle change (text is view-agnostic); temporal-summary scoring to get a timecode. | Repo https://github.com/lucazanella/lavad (linked from project page); code license not verified. Uses open captioners and LLMs; swap to Apache models. |
| AnomalyRuler - "Follow the Rules: Reasoning for VAD with LLMs" (ECCV 2024) | https://arxiv.org/pdf/2407.10299v1 ; repo https://github.com/Yuchen413/AnomalyRuler | Two stages: induction (LLM derives rules from few normal samples) and deduction (rules applied to test frames, with perception smoothing and robust re-check). Borrow: explicit rules + majority-vote smoothing of per-frame descriptions + a second "recheck" pass; fits our "rule citation" requirement and few-shot data. | MIT (verified on repo page). Default code calls an OpenAI-style API in `openai_api.py`; must be replaced with a local LLM/VLM. |
| Holmes-VAD (2024) | https://arxiv.org/pdf/2406.12235 ; repo https://github.com/pipixin321/HolmesVAD ; project page holmesvad.github.io (seen in search snippet only) | Multimodal-LLM VAD with explanations, plus VAD-Instruct50k instruction dataset built semi-automatically. Borrow: explanation-generation format (timecode + rationale), and the idea of generating pseudo-instruction data from few labels. | Repo MIT (verified). Released HolmesVAD-7B weights and underlying base model (repo has a `videollava` folder, not confirmed) may carry separate terms; check HF card. Trained on UCF-Crime-style data, domain mismatch with our room. |
| VadCLIP (AAAI 2024) | https://arxiv.org/html/2308.11681v3 ; repo https://github.com/nwpu-zxr/VadCLIP | Weakly-supervised VAD with frozen CLIP and text-prompt classification, video-level labels only. Borrow: weak (clip-level) supervision is exactly our label regime (yes/no per clip); frozen CLIP features + light temporal head trains on ~40 clips. | Apache-2.0 (verified). CLIP weights themselves are MIT (from memory, unverified). Trained for UCF-Crime/XD-Violence categories, so a head must be retrained. |
| CompAgent - agentic visual compliance verification (AWS, 2025) | https://arxiv.org/pdf/2511.00171 | Single-image compliance checking where an MLLM plans which vision tools (detectors, etc.) to call per policy. Borrow: the "policy text -> tool plan -> detector evidence -> verdict" pattern for rule-by-rule checks on keyframes. | License/code not checked. |
| Post-Completion Mistake Detection (procedural mistake detection, Univ. of Trento record) | https://iris.unitn.it/handle/11572/484810 | Reports that end-to-end VLMs struggle with fine-grained temporal logic (ongoing vs completed actions). Supports our design choice: do NOT ask a VLM "was the safe left open for 10 s"; extract per-frame states and do the temporal logic in code. | n/a |
| Unattended-luggage system, Sensors 2025 (adapted YOLOv11-s + person-bag tracking logic) | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12074375/ | Detection + owner-object distance/time logic. Borrow: owner-leaves-object state machine for "cash/bag left on desk when cashier leaves". Benchmarks named: ABODA, PETS2007, AVSS AB2007. | Paper; check for code. |

Not found in this research: arXiv papers specifically on bank-teller behavior analysis or cash-handling anomaly detection. Treat that as a gap (and a pitch point: novel domain), not as proof none exist.

## 3. Open-source tools to reuse (license noted)

| Name | Link | Role for us | License |
|---|---|---|---|
| Roboflow `supervision` | https://github.com/roboflow/supervision | Zones (polygon regions), tracking utilities, dwell/time-in-zone, annotators. Use to implement "person in cashier zone", "object on desk zone", "door zone" and timers. Tutorial on dwell time exists in its docs. | MIT (verified) |
| Grounding DINO | https://playground.roboflow.com/models/idea-research/grounding-dino (aggregator page; official repo not fetched) | Open-vocabulary detection of "banknotes", "cash bag", "mug", "phone", "keys", "laptop" without labeled data; robust to new camera angle. | Apache-2.0 per aggregator; verify on the official repo |
| OWLv2 / OWL-ViT | https://playground.roboflow.com/models/google/owl-vit | Alternative open-vocab detector; supports image-conditioned queries (give it a crop of "our" cash bag). | License NOT confirmed in this research (from memory Apache-2.0). Verify. |
| YOLO-World | https://roboflow.com/model-licenses/yolo-world | Fast open-vocab detector. | GPL-3.0 (per Roboflow); commercial license needed. AVOID for a commercial product unless licensed. |
| Florence-2 | https://playground.roboflow.com/models/compare/florence-2-vs-yolo-world | Small captioner/grounder/OD model (0.23B/0.77B), good for dense captions and phrase grounding on frames. | MIT (per aggregator) |
| SmolVLM / SmolVLM2 | https://playground.roboflow.com/models/hugging-face/smolvlm2 | 256M-2.2B VLMs, image + video; cheap local frame QA ("is the safe door open?"). | Apache-2.0 (per replicate/SmolVLM release note seen in search) |
| Qwen3-VL (2B-32B dense, MoE variants) | https://freeapihub.com/ai-models/qwen3-vl (secondary source) | Strongest open general VLM family for local frame/clip QA; 2B/4B/8B fit commodity GPUs. | Apache-2.0 per search results (secondary); confirm on HF card per size |
| Qwen2.5-VL, InternVL3 | (no primary source fetched) | Also candidates. | Licenses NOT confirmed here; Qwen2.5-VL 3B/72B historically have non-Apache "Qwen" licenses for some sizes, so check each card. |
| AJV009/risk-package-detection | https://github.com/AJV009/risk-package-detection | YOLOv8s (COCO) + OpenVINO + DeepSORT; flags a person moving away from a bag for a set time, with grace period if owner returns. Direct template for "cash left on desk, cashier gone". | License not shown; also note YOLOv8 is AGPL-3.0 (from memory), which is a commercial blocker; use an Apache/MIT detector (e.g. RT-DETR, D-FINE, YOLOX) or open-vocab above. |
| romain420/abandoned_luggage | https://github.com/romain420/abandoned_luggage | Simple abandoned-object demo with a custom YOLO weights file. Reference only. | not checked |
| navyverma/AbandonObject | https://github.com/navyverma/AbandonObject | Classical OpenCV background subtraction + stationary-time tracking. Idea: background-model "static object appears on desk" works independent of object class (good for cash piles that no detector knows). Old code. | not checked |
| LAVAD / AnomalyRuler / Holmes-VAD / VadCLIP | see section 2 | Reference implementations. | see section 2 |

Design implications (inference from the above, not from a single source):
1. Do not train an end-to-end video classifier on 40 clips; fixed-camera features will not transfer to the new angle.
2. Layer: person detection+tracking (role: cashier/client/collector via zone and dwell) -> open-vocab object states (cash on desk, bag, phone, mug, safe door open, key in lock, laptop screen state) -> temporal rule engine (supervision zones + timers) -> local VLM only for borderline keyframe confirmation and for the textual rationale.
3. Use the 40 clips to tune thresholds/rules and measure precision/recall of the whole pipeline, not to train weights. Use viewpoint-robust cues (relations, e.g. "person absent while cash present") rather than pixel-location rules; make zones annotatable per camera in a 1-minute calibration step.
4. Beware AGPL/GPL components (Ultralytics YOLO, YOLO-World).

## 4. Commercial products (for the pitch)

| Vendor | Link | Claims (as seen) |
|---|---|---|
| StopLift (checkout vision) | https://www.securityinfowatch.com/video-surveillance/video-analytics/article/21267424/can-ai-help-put-a-dent-in-shoplifting (search snippet summarised StopLift elsewhere; the fetched article itself covers only Veesion) | Per search summary: detects sweethearting, i.e. cashier fakes scanning, covers barcodes, friend/colleague collusion; visually checks every transaction. Later part of NCR (from memory, not verified here). |
| Everseen | https://www.cbinsights.com/company/everseen/alternatives-competitors | AI/computer vision for checkout lanes, aisles, inventory; retail shrink reduction and ROI. Listed with StopLift as competitors. Only listing-level detail was seen. |
| Veesion | https://veesion.io/en/our-solution/ | Real-time shoplifting gesture detection on existing CCTV; alerts via WhatsApp/Telegram; no facial recognition (GDPR-friendly). Retail only. Good precedent for "alert clip with evidence to a phone". |
| TRASSIR for Banks | https://trassir.com/marketing-materials/files/industries-eng/TRASSIR for Banks.pdf (link from search; the fetch returned HTTP 404, contents NOT verified) | Marketed as smart video surveillance for the banking sector, incl. Kazakhstan, per search snippet. Details unverified. |
| AxxonSoft (ITV) Axxon Next | https://www.retail.ru/rbc/pressreleases/axis-communications-axis-communications-i-itv-axxonsoft-predstavili-sovmestnoe-reshenie-dlya-riteyla-na-forume-rossiyskaya-rozni | VMS with built-in AI analytics, tens of thousands of sites; retail solution with Axis. No cashier-room specifics seen. |
| Macroscop | https://habr.com/en/companies/macroscop/articles/304490 | Russian VMS+analytics vendor; search showed no bank/cash analytics; says niche tasks need custom development. |
| NtechLab (FindFace) | https://www.cbinsights.com/company/vocord (page shows Vocord; NtechLab data came from the search summary) | Face, silhouette, action, vehicle recognition for business and government; reported #1 in a US Commerce Dept test, Oct 2025 (per CB Insights via search). Face-ID oriented, not procedure compliance. |
| Vocord | https://www.cbinsights.com/company/vocord | Face-recognition vendor in the same test. |
| Sergek (Korkem Technologies, Kazakhstan) | https://www.lada.kz/kazakhstan-news/152270-v-kazakhstane-sergek-poluchil-novye-funktsii-chto-teper-vidiat-kamery-na-samom-dele.html | City-scale smart-camera system (traffic, now behavior analytics such as fight detection). Nothing found about banks. Astana operation is moving to Presight (G42) per bes.media/orda.kz. |

Pitch angle: incumbents either do retail POS fraud (StopLift/Everseen/Veesion) or generic security events/face recognition (Axxon, Macroscop, NtechLab, Vocord, Sergek). Nothing we found targets bank cash-discipline procedures with rule citation and evidence frame, running fully on-prem. State this as "we did not find", not as "does not exist".

---

## Top 5 most useful for us and why

1. **AnomalyRuler** (https://github.com/Yuchen413/AnomalyRuler, MIT). Rule induction/deduction with perception smoothing and a recheck pass maps directly onto "rule citation + few labels"; swap the OpenAI calls for a local VLM/LLM.
2. **LAVAD** (https://arxiv.org/html/2404.01014v1). Training-free caption->summary->score with temporal refinement; view-independent because it goes through text, which matters for the unseen camera angle.
3. **Roboflow supervision** (https://github.com/roboflow/supervision, MIT). Zones, tracking and dwell-time make the temporal rule engine (safe open, desk occupied, door open duration) quick to build under a permissive license.
4. **Open-vocabulary detectors + small VLMs** (Grounding DINO, OWLv2, Florence-2, SmolVLM2, Qwen3-VL; mostly Apache/MIT, verify each; avoid GPL YOLO-World and AGPL Ultralytics). Zero-shot object/state perception without labeled classes, local and commercial-safe.
5. **Sweethearting work (Bobbitt/Connell IBM, StopLift patents) and the abandoned-object pattern (risk-package-detection, Sensors 2025)**. Closest domain precedents: primitive events + temporal ordering rules, and the owner-leaves-object state machine for "cash left on desk when cashier leaves". Also the evidence that end-to-end VLMs are weak at temporal logic (Trento thesis record) justifies doing time logic in code.

Open gaps worth a follow-up if time allows: verify licenses (OWLv2, Qwen2.5-VL, InternVL3, LAVAD repo, Holmes-VAD base weights); look for datasets for hand-object interaction from overhead views; look for any CIS bank-vendor case (TRASSIR bank PDF could not be fetched).
