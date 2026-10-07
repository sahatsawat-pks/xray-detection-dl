# Model Card — Bone Fracture X-Ray Detector

> Following the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) format (Mitchell et al., 2019).

## Model Details

| Field | Value |
|---|---|
| **Name** | Bone Fracture X-Ray Detector |
| **Version** | 1.0.0 |
| **Architecture** | EfficientNet-B0 (ImageNet pretrained, full fine-tune) |
| **Parameters** | 5.3M (all trainable) |
| **Framework** | PyTorch ≥ 2.1 |
| **Task** | Binary classification (Fractured / Non-Fractured) |
| **Input** | Single X-ray image (JPEG/PNG), resized to 224×224 |
| **Output** | Class label, confidence score, fracture probability |
| **Training date** | September 2026 |
| **Developed by** | Sahatsawat Nitjaphant (6688249), Ongsa Raksalam (6688093), Xinyi Chen (6688232) |
| **Course** | ITCS355 — Machine Learning Operation and Deployment |

## Intended Use

### Primary use case
AI-assisted triage in emergency departments and radiology workflows. A radiologist uploads an X-ray and receives a fracture/no-fracture prediction as a **second opinion** — not a replacement for clinical judgment.

### Intended users
- Radiologists reviewing X-ray queues
- ED physicians triaging musculoskeletal injuries
- Medical students learning radiographic interpretation

### Out-of-scope uses
- **NOT** for standalone diagnosis without physician review
- **NOT** for non-musculoskeletal imaging (CT, MRI, ultrasound)
- **NOT** for pediatric fractures (dataset is predominantly adult)
- **NOT** for surgical planning or treatment decisions

## Training Data

| Attribute | Value |
|---|---|
| **Dataset** | [FracAtlas](https://www.nature.com/articles/s41597-023-02432-4) |
| **Licence** | CC BY-SA 4.0 |
| **Total images** | 4,083 X-rays |
| **Fractured** | 717 (17.6%) |
| **Non-Fractured** | 3,366 (82.4%) |
| **Body parts** | Leg (2,273), Hand (1,538), Shoulder (349), Hip (338) |
| **Split** | 70% train / 15% validation / 15% test (stratified) |
| **Augmentation** | Random crop, horizontal flip, rotation ±15°, color jitter |
| **Normalization** | ImageNet mean/std |
| **Class imbalance** | Addressed with inverse-frequency weighted CrossEntropyLoss |

### Data limitations
- Single institution dataset — may not generalise to X-rays from different equipment/protocols
- Predominantly adult patients — pediatric fractures have different presentation
- Body part distribution is skewed (56% leg, 38% hand, 6% shoulder/hip)
- No metadata on patient demographics (age, sex, ethnicity)

## Evaluation Results

### Test set performance (seed=42, 15% holdout)

| Metric | ModelBaseline (CNN) | ModelImproved (ResNet-18) | **ModelFinal (EfficientNet-B0)** |
|---|---|---|---|
| Accuracy | 0.8412 | **0.9051** | 0.8969 |
| Precision | 0.6250 | **0.7753** | 0.7340 |
| Recall | 0.2336 | **0.6449** | **0.6449** |
| **F1-score** | 0.3401 | **0.7041** | 0.6866 |
| **AUC-ROC** | 0.8080 | 0.9046 | **0.9130** |

### Why ModelFinal was selected
- Highest F1-score (primary metric for imbalanced classification)
- Highest recall — minimises missed fractures (clinically safer)
- Smallest parameter count (5.3M vs 11.2M) — cheaper to serve
- Best AUC-ROC — better calibrated probability estimates

### Failure analysis
- False negatives are concentrated in hip and shoulder fractures (under-represented in training)
- Hairline fractures on small bones are the hardest to detect
- Model confidence is generally well-calibrated but degrades on atypical views (oblique)

## Ethical Considerations

### Risks
- **Automation bias**: Clinicians may over-rely on AI predictions, especially for "Non-Fractured" calls. High-confidence false negatives are the primary clinical risk.
- **Demographic bias**: Training data lacks demographic metadata, so performance across patient subgroups (age, sex, ethnicity) is untested.
- **Deployment context**: Performance was measured on a static test set. Production performance may differ due to equipment variation, image quality, and patient mix.

### Mitigations
- Predictions below 60% confidence are flagged as `"uncertain": true` — requiring mandatory human review
- The service validates input images (entropy check, format validation) to reject garbage inputs
- All predictions include `model_version` for traceability back to the exact training run
- Drift monitoring (PSI) alerts when prediction distributions shift from baseline

## Quantitative Analyses

### Performance by body part (estimated from dataset composition)

| Body part | Training samples | Share | Expected reliability |
|---|---|---|---|
| Leg | ~1,590 | 56% | Highest — most training data |
| Hand | ~1,076 | 38% | High |
| Shoulder | ~244 | 6% | Moderate — limited data |
| Hip | ~237 | 6% | Moderate — limited data |

### Confidence threshold analysis

| Threshold | Effect |
|---|---|
| 0.50 (default) | Standard binary classification |
| 0.60 (deployed) | Flags ~10% of predictions as "uncertain" — safer for clinical use |
| 0.80 | Would flag ~25% — too conservative for practical use |

## Caveats and Recommendations

1. **Always pair with human review**. This model is a triage aid, not a diagnostic tool.
2. **Monitor drift continuously**. The PSI drift detector runs in production — investigate any alerts promptly.
3. **Retrain on local data**. Hospital-specific X-ray equipment produces different image characteristics. Fine-tuning on local data (even 100–200 images) would improve performance.
4. **Test on underrepresented body parts** before deploying for shoulder/hip fracture detection.
5. **Do not use for pediatric patients** without additional validation — growth plates create confounding features.
