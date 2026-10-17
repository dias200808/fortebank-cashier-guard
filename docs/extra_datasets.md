# Extra datasets for cashier-room CCTV cash-discipline project

Research date: 2026-10-17. Method: web search + page fetches (limited budget). Rules:
- Only URLs actually seen in search/fetch results are listed.
- "verified" = license text seen on the page or in a search snippet of the page. "unverified" = not seen, or only second-hand, or from memory. Re-check every license on the dataset page before putting data into a model that ForteBank will own. Roboflow Universe license badges can change and many Roboflow sets are scraped from third-party images.
- Rule of thumb: CC0/CC-BY/MIT/Apache/ODbL = commercial OK (with attribution). CC-BY-NC / "research only" = NOT OK for the commercial deliverable. Using NC data only to prototype is a grey zone: weights trained on it are arguably derivative, so avoid it in the final model.
- Face recognition is forbidden. Person detection is fine. Skip face-centric datasets.
- Usefulness: 1 = marginal, 5 = directly on target. All targets are overhead, 464x848, so most web images (frontal, close-up) are mainly useful for pre-training or detector warm-up. Our own footage will dominate.

---
## 1. Banknotes / cash / money in hands / counting machines

| Name | Link | Size | Contents | License | Commercial OK? | Use (1-5) |
|---|---|---|---|---|---|---|
| Dataset of Kazakhstan Banknotes with Annotations (Sadyk, Bozshina, 2023) | https://data.mendeley.com/datasets/dny3dgvvw8/1 | 4,200 images, 14 classes (1 to 20,000 KZT), YOLO format, 70:30 train/val | Tenge notes, YOLO boxes | CC BY 4.0 (verified, fetched) | Yes (attribution) | 4 (only KZT detection set found; likely clean studio-style shots, so expect a domain gap to overhead CCTV) |
| Kazakh Banknote Image Dataset (Yerzhan, 2025) | https://data.mendeley.com/datasets/gyw87k93yb | 500/1000/2000/5000/10000/20000 folders (>=100 imgs each, 5000 split into new/old) + Mix folder (>1000 imgs) | Classification folders, Mix has several notes per image; no boxes | CC BY 4.0 (verified, fetched) | Yes (attribution) | 3 (classification only, but includes the old/new 5000 design and multi-note images) |
| Paper using the 4,200-image set | https://beei.org/index.php/EEI/article/view/8004 | n/a | Baselines (reported 94% classic, 99.11% hybrid) | n/a | n/a | 2 (reference only) |
| Roboflow "Banknote detection" (BIN) | https://universe.roboflow.com/bin-mibxd/banknote-detection-2amdt | 1,595 images | Banknote boxes, currency unclear | CC BY 4.0 (search snippet) | Yes, verify | 3 (generic "is this cash", good for pre-training) |
| Roboflow "C" (cash-detect) | https://universe.roboflow.com/cash-detect/c-tzdlb | unverified | Classes "Money", "100" | CC BY 4.0 (snippet) | Yes, verify | 2 |
| Roboflow "moneyt" | https://universe.roboflow.com/project-pswjo/moneyt | small | Single class "money" | CC BY 4.0 (snippet) | Yes, verify | 2 |
| Roboflow "MoneyDetection" | https://universe.roboflow.com/banknote-recognition/moneydetection-pclz2 | unverified | Banknote, real vs fake workflow | CC BY 4.0 (snippet) | Yes, verify | 2 |
| Roboflow "Banknote" (workkk) / "Banknotes 2" | https://universe.roboflow.com/workkk-quf76/banknote-br9ax , https://universe.roboflow.com/machine-learning-hgbat/banknotes-2 | unverified | Notes 20/100/1000, non-KZT | CC BY 4.0 (snippet) | Yes, verify | 2 |
| Roboflow "money" (9try) | https://universe.roboflow.com/9try/money-veb5w | small, ~2 downloads | Classes hand, coin, banknote | unverified | unverified | 2 (hand + note co-occurrence) |
| Roboflow "money counting machine" (forensic) | https://universe.roboflow.com/forensic-1dit3/money-counting-machine | unverified (page returned 403) | Counting-machine images | unverified | unverified | 3 (rare topic, check manually) |
| Roboflow "Money count" (fke) | https://universe.roboflow.com/fke/money-count-3u4sm | unverified | Money counting | unverified | unverified | 2 |
| Roboflow browse page "money" (lists "Cash Counter" ~1.3k imgs, "How much money" 541 imgs) | https://universe.roboflow.com/browse/money | n/a | Index only | per set | per set | 2 |
| Roboflow "holding hand only" (lazenca) | https://universe.roboflow.com/lazenca/holding-hand-only-4riph | unverified | Classes hand, hand_holding (no money) | unverified | unverified | 2 |
| Roboflow "EgoHands Public" | https://universe.roboflow.com/muazs-workspace/hands-eqar1 | unverified | Hands, egocentric | unverified | unverified | 2 |

Gaps: no public set found with overhead views of cash stacks in hands or on a desk, and no tenge in hands. Film our own (section "Recording" below) and use the KZT sets as pre-training or for copy-paste augmentation.

---
## 2. Bags (cash-in-transit, sealed bags, handbags, backpacks on a desk)

| Name | Link | Size | Contents | License | Commercial OK? | Use (1-5) |
|---|---|---|---|---|---|---|
| Roboflow "CashBag" (novo) | https://universe.roboflow.com/novo/cashbag | 208 images (snippet; page returned 403) | Name suggests cash bags; the workflow example said "Parcel", so contents are unconfirmed | CC BY 4.0 (snippet) | Yes, verify | 3 (inspect images first) |
| Roboflow "bags" (ewasince) | https://universe.roboflow.com/ewasince/bags-lsgea | unverified | Bags | unverified | unverified | 2 |
| Roboflow "bag" (count-coin) | https://universe.roboflow.com/count-coin/bag-8sf43 | unverified | Bags | unverified | unverified | 2 |
| Roboflow "bag-det" | https://universe.roboflow.com/new-workspace-l2a1a/bag-det | unverified | Bags | unverified | unverified | 2 |
| Roboflow "bags detection" | https://universe.roboflow.com/classification-ozqwr/bags-detection-6e4ui | unverified | Bags | unverified | unverified | 2 |
| Roboflow "bagdetection()" (GingerAlpha) | https://universe.roboflow.com/gingeralpha/bagdetection-oxads | unverified | Bags | Public domain (snippet) | Yes, verify | 2 |
| Roboflow "Bag" (bercades) | https://universe.roboflow.com/bercades/bag-pjfxp | 71 images | Bags | unverified | unverified | 1 |
| COCO (handbag, backpack, suitcase classes) | no URL seen in this research | 330k images | Generic bags in everyday scenes | annotations CC BY 4.0, Flickr images mixed (unverified here) | Mostly yes, verify | 3 (a pretrained COCO detector already knows these classes; no extra download needed) |

Note: nothing found for sealed or collector bags. Collector bags are a custom class and need our own footage. Colour and tamper-seal features matter, so record these.

---
## 3. Safe / vault door and room door open vs closed

| Name | Link | Size | Contents | License | Commercial OK? | Use (1-5) |
|---|---|---|---|---|---|---|
| Roboflow "Open & Closed door Detection 2" | https://universe.roboflow.com/open-closed-doors-detection/open-closed-door-detection-2 | unverified | Classes open, closed, semi | Public Domain (snippet) | Yes, verify | 3 |
| Roboflow "ClosedDoors_val" | https://universe.roboflow.com/mvisimageprocessing/closeddoors_val | unverified | Closed doors | CC BY 4.0 (snippet) | Yes, verify | 2 |
| Roboflow "door" (pji) | https://universe.roboflow.com/pji/door-s53ai | 1,216 images | Open/closed doors | unverified | unverified | 3 |
| Roboflow "door_open_close" | https://universe.roboflow.com/-utfyu/door_open_close-42gum | unverified | Open/closed | unverified | unverified | 2 |
| Roboflow "door-open-closed" (elevator) | https://universe.roboflow.com/elevator-omucw/door-open-closed-mabek | unverified | Open/closed | unverified | unverified | 2 |
| Roboflow "door-status-detector" | https://universe.roboflow.com/viskom-9e73p/door-status-detector | unverified | Door status | unverified | unverified | 2 |
| Paper: door state detection with Roboflow | https://eprints.ugd.edu.mk/37502/ | n/a | open/closed/semi-open | n/a | n/a | 1 |

Gap: no safe or vault dataset found at all. Open/closed state is a fixed-camera, fixed-location problem. For a known camera, a simple ROI-based classifier or frame-difference trained on our ~40 clips is likely better than generic door models. Door datasets are only useful for pre-training, and the test camera angle is new.

---
## 4. Laptop/monitor state, keys, ID cards/documents

| Name | Link | Size | Contents | License | Commercial OK? | Use (1-5) |
|---|---|---|---|---|---|---|
| Roboflow "Withboundings" (romera) | https://universe.roboflow.com/romera/withboundings-cdbo8 | 1,227 images (snippet; page 403) | laptop_off, laptop_on, monitor_off, monitor_on | unverified | unverified | 4 (exactly the screen-state classes; check license) |
| Roboflow "My First Project" (ar-system) | https://universe.roboflow.com/ar-system/my-first-project-4ftyq | unverified | hand_on_power_button, laptop_off, laptop_on | unverified | unverified | 3 |
| Roboflow "laptop screen detection" | https://universe.roboflow.com/projectfyp/laptop-screen-detection-hoof8 | unverified | Single class screen | unverified | unverified | 2 |
| Roboflow "desk_setup" | https://universe.roboflow.com/campus-gvwxo/desk_setup | unverified | mouse, keyboard, monitor, headset, chair | unverified | unverified | 2 |
| Roboflow "Desktop Setup" | https://universe.roboflow.com/riiixch-u3aeh/desktop-setup | unverified | laptop, mouse, keyboard | unverified | unverified | 2 |
| Roboflow "DK-OFFICE" | https://universe.roboflow.com/ispengsiqigmailcom/dk-office | unverified | Class "key" | unverified | unverified | 2 |
| Roboflow "Office Computer" | https://universe.roboflow.com/1-cvbtv/office-computer | unverified | Office computers | unverified | unverified | 1 |

Gap: no ID card or document-on-desk dataset found, and no keys-in-lock dataset. COCO already has laptop/tv/cell phone/keyboard classes. Screen on/off from an overhead camera can be approximated by brightness inside a monitor ROI, with no model needed.

---
## 5. Phone use, eating/drinking, workplace behaviour

| Name | Link | Size | Contents | License | Commercial OK? | Use (1-5) |
|---|---|---|---|---|---|---|
| Roboflow "mobile detection" (zero) | https://universe.roboflow.com/zero-fck3y/mobile-detection-l2iov-bompa | unverified | mobile use, drinking, eating, smoking | CC BY 4.0 (snippet) | Yes, verify | 3 |
| Roboflow "mobile detection" (object-detection) | https://universe.roboflow.com/object-detection-ek5ii/mobile-detection-l2iov-hsjdh | unverified | Mobile phone | unverified | unverified | 2 |
| Roboflow "Driver_Distraction_noeye" | https://universe.roboflow.com/395841716-qq-com/driver_distraction_noeye | unverified | Phone, Danger, Drink, Safe | CC BY 4.0 (snippet) | Yes, verify | 2 (in-car view) |
| Roboflow "Abnormal Driver Behaviour" (pfe) | https://universe.roboflow.com/pfe-lcrvh/abnormal-driver-behaviour-wkmvr | unverified | phone, cigarette, drinking, seatbelt, eating | CC BY 4.0 (snippet) | Yes, verify | 2 |
| Roboflow "Abnormal Driver Behaviour" (aoop) | https://universe.roboflow.com/aoop-vgwve/abnormal-driver-behaviour-nttpp | unverified | Same classes | CC BY 4.0 (snippet) | Yes, verify | 2 |
| Roboflow "driver-distraction-classifier" | https://universe.roboflow.com/testworking-u71qf/driver-distraction-classifier | unverified | drinking, calling | CC BY 4.0 (snippet) | Yes, verify | 2 |
| Roboflow "Certus-02" / "CertusDrive-02" | https://universe.roboflow.com/drowsinessfocus-idpgk/certus-02 | unverified | drinking, eating, holding phone | unverified | unverified | 2 |
| Kaggle State Farm Distracted Driver | no URL seen in this research | ~22k images, 10 classes | Texting, phone, drinking, etc. in-car | Competition data terms, unverified | Probably NO (competition rules) | 1 |

Driver-cabin datasets are frontal and close up, so the domain gap to overhead is large. For overhead, hand-to-mouth and phone use have to be learned from our own clips or from pose-based rules (COCO pretrained "cell phone", "cup", "bottle" detectors plus wrist/head keypoint distance).

---
## 6. Action recognition and surveillance datasets (hand-overs, passing, theft, left objects)

| Name | Link | Size | Contents | License | Commercial OK? | Use (1-5) |
|---|---|---|---|---|---|---|
| MERL Shopping | https://merl.com/research/downloads/MERL_Shopping_Dataset ; license page https://merl.com/research/license/MERL_Shopping_Dataset ; HF mirror card https://docs.voxel51.com/dataset_zoo/datasets_hf/merl_shopping_dataset.html | 106 videos, ~2 min each, fixed overhead camera | Shoppers reaching to shelf, inspecting, retracting hand, hand in basket (5 actions) | Non-commercial research only (per HF card; MERL notice not opened) | NO | 4 for relevance (overhead hand actions), but license blocks the final model; fine for studying methods only |
| MEVA (Multiview Extended Video with Activities) | https://registry.opendata.aws/mevadata/ | ~328 h video (516 GB, 4,259 clips) + 4.6 h UAV; annotations for ~184 h per registry (the paper says 22 h public; conflicting) | Staged person and person-vehicle activities (carrying, handing over object, opening doors, etc.) from many CCTV cameras | CC BY 4.0 (Kitware/IARPA, per NIST ActEV page; verified via search) | Yes (attribution) | 4 (has door open/close, object transfer, carrying bag from CCTV angles; angles are elevated, not directly overhead) |
| VIRAT | https://actev.kitware.com/pub/VIRAT-Video-Data-Set-Protection-Agreement-1-4-11.pdf | Release 2.0 ground/aerial CCTV | Person-object activities (loading, opening door, carrying, etc.) | Usage agreement states lawful research and commercial use allowed; no PII; redistribution only to parties accepting the same agreement (2011 doc, verify current terms) | Probably yes, verify | 3 |
| UCF-Crime | https://www.crcv.ucf.edu/chenchen/datasets/ (fetch failed, TLS error) ; frame copy https://huggingface.co/datasets/hibana2077/UCF-Crime-Dataset | 1,900 videos, ~128 h, 13 anomaly classes (incl. shoplifting, robbery, burglary) | Real CCTV anomalies, weak video-level labels | "Research purposes only" per UCF page snippet; mirror says "Other" | NO / unverified | 3 (shoplifting/robbery clips; legal risk for commercial) |
| DCSASS (UCF-Crime clips relabelled) | https://www.kaggle.com/datasets/mateohervas/dcsass-dataset | unverified (page content not retrieved) | Clips from UCF-Crime sorted by class | unverified, derived from UCF-Crime so inherit its restrictions | Treat as NO | 2 |
| XD-Violence | https://service.tib.eu/ldmservice/dataset/xd-violence | 4,754 videos, ~217 h | Violence in movies/YouTube with audio | unverified | unverified | 1 (violence, wrong domain) |
| PoseLift / RetailS shoplifting benchmark | paper https://arxiv.org/pdf/2501.06591 ; repo https://gitee.com/soon14/PoseLift | pose sequences, not raw video | Real-world shoplifting as skeleton data | unverified | unverified | 2 (could train a pose-based anomaly head; check license) |
| HICO-DET | https://ml.paperswithcode.com/dataset/hico-det | ~47k images, 600 HOI classes | Human-object interaction boxes | CC0 (per CrossHOI-Bench paper, https://arxiv.org/pdf/2508.18753; second-hand) | Yes, verify | 3 (hold/carry/hand-over style verbs, but web photos, not overhead) |
| V-COCO | paper listing: https://arxiv.org/pdf/2508.18753 | ~10k images, 26 actions | HOI on COCO images | CC BY (per same paper; second-hand) | Yes, verify | 2 |
| EgoHOS | https://github.com/owenzlz/EgoHOS | train/val/test_indomain/test_outdomain splits; sizes not stated | Egocentric hand and interacting-object segmentation, with contact labels | Code MIT; dataset license not stated on page (unverified) | unverified | 2 (egocentric, not overhead) |
| Charades, AVA, Something-Something, Assembly101, ABODA, PETS | no URL seen in this research | n/a | Charades: indoor household actions; AVA: movie atomic actions (annotations CC BY 4.0 from memory); SSv2: hand-object clips (from memory: non-commercial research licence); Assembly101: multi-view assembly with hands (from memory: non-commercial); ABODA: abandoned-object video clips; PETS: left-luggage scenarios | all unverified | assume NO until checked | ABODA 3, PETS 3 (left-behind objects), others 1-2 |

Takeaway: MEVA is the only large CCTV activity set found that is clearly CC BY and commercial-friendly. MERL Shopping is the closest viewpoint match but is non-commercial.

---
## 7. Retail checkout / cashier

| Name | Link | Size | Contents | License | Commercial OK? | Use (1-5) |
|---|---|---|---|---|---|---|
| MERL Shopping (see section 6) | https://merl.com/research/downloads/MERL_Shopping_Dataset | 106 videos | Overhead store hand actions | Non-commercial | NO | 3 |
| IBM sweethearting work (Fan et al., Bobbitt et al.) | https://researcher.ibm.com/publications/detecting-sweethearting-in-retail-surveillance-videos | n/a | Paper on fake-scan detection; data not shown as public | n/a | n/a | 2 (method ideas only: scan-motion + POS alignment) |
| NVIDIA TAO action recognition whitepaper | https://developer.nvidia.com/tao-toolkit-usecases-whitepaper/5-action-recognition | n/a | Retail action recognition recipe on MERL | n/a | n/a | 2 (recipe) |
| Patents on overhead scan detection | https://patents.google.com/patent/US12014544 | n/a | Method only | n/a | n/a | 1 (do not copy; reference only) |

Gap: no public cashier or sweethearting video set with a permissive license was found. Everything for this domain must come from our own footage.

---
## 8. Synthetic data options

| Name | Link | Contents | License | Commercial OK? | Use (1-5) |
|---|---|---|---|---|---|
| BlenderProc (DLR-RM) | https://pypi.org/project/blenderproc | Procedural Blender pipeline, outputs RGB, depth, normals, segmentation, COCO/BOP annotations | GPL-3.0 (per PyPI/mirrors) | Rendered images are our own data. Tool is GPL, so do not embed or redistribute its code in the product; running it to make training images is fine. Verify the asset licences (3D models, textures) separately | 3 |
| NVIDIA Omniverse Replicator | https://developer.nvidia.com/blog/build-custom-synthetic-data-generation-pipelines-with-omniverse-replicator/ | Domain randomisation, annotators and writers for synthetic labelled data | unverified (NVIDIA terms) | unverified | 3 (needs an RTX GPU; heavy setup for a hackathon) |
| Unity Perception | no URL verified | Unity package for labelled synthetic images (boxes, segmentation, keypoints) | unverified | unverified | 2 |
| Existing synthetic CCTV datasets | none found | n/a | n/a | n/a | n/a |

Honest take: for a 1-2 day hackathon, a full synthetic pipeline is probably not worth it. Cheaper alternatives are (a) copy-paste augmentation of cropped banknotes, bags and phones onto our real frames (tenge dataset gives good crops), (b) strong geometric augmentation (flip, rotate, perspective, scale) to simulate the new camera angle, and (c) generative inpainting of objects into real frames only if time is left.

---
## Recommended shortlist (max 8)

1. Dataset of Kazakhstan Banknotes with Annotations (Mendeley, CC BY 4.0). The only KZT detection set, with 4,200 images and YOLO labels. Use it for note-detector pre-training and as source crops for copy-paste augmentation.
2. Kazakh Banknote Image Dataset (Mendeley, CC BY 4.0). Adds the old/new 5000-tenge designs and multi-note images. Use for classification or crop mining.
3. MEVA (AWS Open Data, CC BY 4.0). The best commercial-safe CCTV activity data: door open/close, carrying, object transfer. Take a small subset of clips for the "hand-over" and "door" behaviours and for background and negative examples.
4. Roboflow "Banknote detection" (BIN, CC BY 4.0, 1,595 images). Generic cash detection to pre-train a "money" class. Check images before use.
5. Roboflow "Withboundings" (laptop/monitor on/off, 1,227 images). Only candidate for screen-state classes. License unverified, so check the page. If it is not permissive, use ROI-brightness logic instead.
6. Roboflow "CashBag" (CC BY 4.0, 208 images). A small candidate for collector bags. Inspect it first, and expect to supplement with our own bag footage.
7. HICO-DET (CC0 per second-hand source). Pre-training for person-object interaction (hold, carry, hand-over), and a source of negatives. Domain is web photos, so low weight.
8. VIRAT (Kitware agreement permits commercial use, verify current terms). Person-object CCTV activities, as a fallback if MEVA coverage is thin.

Avoid for the final model: MERL Shopping (non-commercial), UCF-Crime/DCSASS (research only), State Farm (competition terms), Something-Something and Assembly101 (assumed non-commercial). They are fine for reading papers or sanity checks only.

---
## Recording our own clips (the most valuable data)

The test uses a new camera angle, so variation in viewpoint matters more than clip count.

Camera and geometry
- Film each scenario from at least 3 mounting variants: the known layout (overhead, 464x848 portrait), plus about +/-15-30 degrees pitch and yaw, and 2 heights. Include one wider crop and one tighter crop to vary scale.
- Match the target format: 30 fps, portrait 464x848, similar compression (H.264 at a low bitrate). Recording at higher quality and then re-encoding is OK and gives control.
- Vary lighting (daylight, lamps only, dim), desk clutter, clothing colours and number of people (1, 2, 3).

Scenarios (each positive plus a hard negative)
- Cash: counting notes at the desk, cash on the desk left unattended, cash passed hand to hand, cash put in a pocket or bag, cash in the counting machine, tenge props (print or buy fake or souvenir notes; use real ones only with approval).
- Bags: collector bag brought in, sealed vs unsealed, bag on the desk vs the floor, personal handbag/backpack/coat on the desk, bag swapped.
- Safe and doors: safe open vs closed (any cabinet or locker as a stand-in if there is no safe), key in lock vs not, room door open vs closed, someone entering/leaving.
- Desk items: laptop/monitor on, off, locked screen, left on while the person is away, ID card or documents on the desk, keys left on the desk.
- Behaviour: phone use (hold, text, talk), eating, drinking from cup, two people close together or hand-over across the desk, person standing up and leaving the workplace with items on the desk.
- Negatives: normal work with the same objects present, so the model learns the violation, not the object.

Practice
- Short clips of 5-15 s, at least 5-10 repetitions per scenario with different actors, plus clips that start and end with nothing happening.
- Use consent from every participant, no face-recognition use, keep raw footage local.
- Label with the same schema and timestamps as the 40 provided clips. Hold out one camera angle entirely for validation to simulate the unseen test view.
- Cheap multipliers: horizontal flips, rotations of +/-15 degrees, perspective warps, colour jitter, and copy-paste of tenge or bag crops.
