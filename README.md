#  netra.ai — Tumor-Type-Conditioned Multi-Task Brain MRI Segmentation

**Joint tumor classification + FiLM-conditioned segmentation, with counterfactual mask explanations.**

netra.ai is a multi-task deep learning pipeline that takes a single brain MRI slice and, in one forward pass, predicts **what type of tumor is present** and **exactly where it is**, using the classification result itself to actively reshape the segmentation output. It also ships with built-in explainability tools and an automated per-patient reporting system that flags cases worth a second look.

>  **Research preview — not a diagnostic tool.** This project is a technical exploration of multi-task conditioning in medical imaging, not a certified or clinically validated system.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Key Ideas](#key-ideas)
- [System Architecture](#system-architecture)
- [How It Works — End to End](#how-it-works--end-to-end)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Training](#training)
- [Evaluation & Ablation Study](#evaluation--ablation-study)
- [Explainability](#explainability)
- [Automated Reporting](#automated-reporting)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why This Project Exists

Most segmentation pipelines treat "what is this tumor" and "where is this tumor" as two separate problems, often solved by two separate models. In practice, a radiologist's read of *what* a lesion likely is directly informs *how* they trace its boundary. netra.ai tests whether a single network can do the same — using its own tumor-type belief to condition how it draws the mask, rather than just training two heads side by side and hoping they agree.

## Key Ideas

- **One encoder, two tasks.** A shared convolutional encoder feeds both a classification head and a segmentation decoder — no duplicated feature extraction.
- **FiLM conditioning, not just multi-task loss.** The segmentation decoder doesn't just get gradients from a joint loss — its *feature maps are actively modulated* (scaled and shifted) at every decoder stage using the classifier's embedding. Same decoder weights, different behavior per tumor type.
- **Confidence-aware output.** Every prediction ships with a confidence score, not just a binary mask.
- **Counterfactual probing.** You can force the decoder to condition on a *different* tumor type's embedding and watch how the mask changes — a direct, visual test of whether the conditioning is doing anything real.
- **Built-in second-opinion flagging.** An automated report flags slices where the classifier is unsure or where the mask is unstable across competing class hypotheses.

## System Architecture

```
                         ┌─────────────────────────┐
                         │   Input MRI Slice (256×256, 1-ch)  │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │   Shared CNN Encoder     │
                         │  enc1→enc2→enc3→enc4     │
                         │      → bottleneck        │
                         │  (channels: 32→64→128→256→512) │
                         └────────────┬────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
        ┌────────────▼────────────┐      ┌─────────────▼─────────────┐
        │  Classification Head     │      │   FiLM-Conditioned         │
        │  GAP → FC(64) → FC(3)    │      │   Segmentation Decoder     │
        │  → tumor type + embedding│──┐   │   dec4→dec3→dec2→dec1     │
        └────────────┬─────────────┘  │   │  (skip connections from   │
                     │                └──▶│   encoder, gamma/beta      │
                     │  embedding (64-d)   │   modulation per stage)   │
                     │  drives FiLM  ─────▶└─────────────┬─────────────┘
                     │                                    │
        ┌────────────▼────────────┐         ┌─────────────▼─────────────┐
        │ meningioma / glioma /    │         │  Pixel-wise Tumor Mask    │
        │ pituitary + confidence   │         │  + per-class confidence  │
        └──────────────────────────┘         └───────────────────────────┘
                     │                                    │
                     └───────────────┬────────────────────┘
                                      ▼
                     ┌─────────────────────────────────┐
                     │  Explainability & Reporting Layer │
                     │  • Grad-CAM (classifier)          │
                     │  • Saliency map (segmenter)       │
                     │  • Counterfactual mask probing     │
                     │  • Confidence / instability flags  │
                     └─────────────────────────────────┘
```

**FiLM (Feature-wise Linear Modulation):** a small linear layer maps the classifier's embedding to a per-channel `(gamma, beta)` pair for every decoder stage. Each decoder feature map is transformed as `x * (1 + gamma) + beta` before its convolutions run. This means the *same* decoder weights produce different segmentation behavior depending on which tumor type the network currently believes it's looking at — instead of training three separate segmentation heads.

## How It Works — End to End

1. **Input:** a single-channel, 256×256 T1-weighted contrast-enhanced MRI slice.
2. **Shared encoding:** four downsampling `ConvBlock`s + a bottleneck extract hierarchical features, same as a standard U-Net encoder.
3. **Classification branch:** global average pooling on the bottleneck → a 64-dim embedding → a linear layer predicts `meningioma / glioma / pituitary` with a softmax confidence.
4. **FiLM-conditioned decoding:** the same 64-dim embedding is fed into every decoder stage. Each stage upsamples, concatenates the encoder skip connection, applies FiLM's `gamma`/`beta` modulation, then runs its convolutions.
5. **Output:** a pixel-wise tumor mask (sigmoid logits) and the class probabilities, from one forward pass.
6. **Counterfactual check (optional, diagnostic tool):** the decoder can be re-run with `override_embedding` set to the *mean training embedding* of any class, letting you see how the mask changes if the model "thought" the tumor was a different type.
7. **Reporting:** for every slice, the pipeline logs predicted type, confidence, tumor area (px and mm²), and flags the case if confidence is low or the mask is unstable between the top prediction and the runner-up class.

## Dataset

- **Source:** [Figshare Brain Tumor Dataset](https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset) (Cheng et al., 2017), accessed via Kaggle.
- **Size:** 3,064 T1-weighted contrast-enhanced MRI slices from 233 patients.
- **Labels:** each slice has both a tumor-type label (meningioma / glioma / pituitary) **and** a pixel-level segmentation mask.
- **Format:** original MATLAB v7.3 (`.mat`, HDF5-based) files, parsed with `h5py` and cached as image/mask pairs.
- **Split strategy:** patient-level (not slice-level) split into train/val/test to prevent leakage, stratified by each patient's majority tumor type so all three classes are represented proportionally in every split.
- **Augmentation:** resize to 256×256, horizontal flip, random 90° rotation, shift/scale/rotate, brightness-contrast jitter (via `albumentations`); normalization only at eval time.

## Model Details

| Component | Detail |
|---|---|
| Backbone | Custom U-Net-style encoder/decoder, `ConvBlock` = Conv→BN→ReLU ×2 |
| Base channels | 32 (doubling per stage: 32→64→128→256→512) |
| Embedding dim | 64 |
| Conditioning | FiLM (`gamma`, `beta` per channel, per decoder stage) |
| Classification head | AdaptiveAvgPool → Linear(512→64) → ReLU → Dropout(0.3) → Linear(64→3) |
| Segmentation head | 1×1 Conv → single-channel mask logits |
| Ablation baseline | Identical architecture with `use_film=False` (decoder receives no conditioning) |

## Training

- **Loss:** weighted multi-task loss — `0.6 × segmentation loss + 0.4 × classification loss`, where segmentation loss is `0.5 × BCE + 0.5 × Dice`, and classification loss is standard cross-entropy.
- **Optimizer:** Adam, initial LR `1e-3`, with `ReduceLROnPlateau` (factor 0.5, patience 3) tracking a combined validation score.
- **Checkpointing metric:** `0.6 × val_dice + 0.4 × val_accuracy` — the best combined score is what gets saved, not either metric alone.
- **Batch size:** 16, image size 256×256.
- **Two models trained:** the proposed FiLM-conditioned model and a no-FiLM ablation baseline, under identical settings, for direct comparison.

## Evaluation & Ablation Study

Both models are evaluated on the same held-out, patient-disjoint test set for:
- Overall and **per-tumor-type Dice coefficient**
- **IoU**
- Classification accuracy / confusion matrix

The ablation (FiLM vs. no-FiLM) isolates whether conditioning the decoder on tumor type actually improves segmentation, rather than just adding parameters.

## Explainability

Two complementary views are generated for the same slice:
- **Grad-CAM on the classifier** (bottleneck features) — shows *where the network looks to decide what the tumor is*.
- **Gradient-based saliency on the segmenter** — shows *where the network looks to decide where the tumor is*.

Comparing the two exposes whether classification and segmentation reasoning are spatially aligned or genuinely distinct.

**Counterfactual mask probing** goes further: it computes a mean "prototype" embedding per class from training data, then decodes the *same* input slice three times — once under each class's conditioning — and compares the resulting masks. If FiLM conditioning is doing real work, the mask should visibly shift toward what's typical of each tumor type.

## Automated Reporting

For every test slice, the pipeline generates a row containing:
- Predicted tumor type + confidence
- Predicted tumor area (pixels and mm², using a configurable pixel-spacing constant)
- **`flag_low_confidence`** — raised when classification confidence is low
- **`flag_unstable_segmentation`** — raised when the mask predicted under the model's own top class differs substantially in area from the mask predicted under the *runner-up* class's conditioning
- A combined **`needs_review`** flag for cases worth a second look

These are aggregated into a per-patient summary CSV as well as a per-slice CSV.

## Project Structure

```
netra-ai/
├── data/                          # raw / cached dataset (not committed)
├── checkpoints/                   # saved model weights + training history
├── figures/                       # Grad-CAM, saliency, counterfactual plots
├── reports/
│   ├── slice_level_report.csv
│   └── patient_summary_report.csv
├── notebooks/
│   └── brain-mri-multitask-film-segmentation.ipynb
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Me-Rajdip/Netra-Code.git
cd Netra-Code

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
# or, minimally:
pip install torch h5py albumentations kagglehub scikit-learn opencv-python pandas numpy matplotlib
```

**Dataset access** (requires a Kaggle account + API token in `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d ashkhagan/figshare-brain-tumor-dataset -p data --unzip
```

## Usage

Run the full pipeline end to end via the notebook:

```bash
jupyter notebook notebooks/brain-mri-multitask-film-segmentation.ipynb
```

The notebook is organized so each section runs independently after setup:
1. Environment setup → dataset download → `.mat` parsing
2. Patient-level stratified split
3. Dataset/augmentation definitions
4. Model definition (`MultiTaskFiLMUNet`)
5. Train proposed (FiLM) model + no-FiLM ablation baseline
6. Test-set evaluation + ablation comparison
7. Grad-CAM / saliency explainability
8. Counterfactual mask probing
9. Automated per-patient reporting

## Results

> Fill in with your latest run's numbers from `checkpoints/*_history.json` and `reports/`.

| Model | Overall Dice | Meningioma Dice | Glioma Dice | Pituitary Dice | Classification Accuracy |
|---|---|---|---|---|---|
| Proposed (FiLM) | — | — | — | — | — |
| Baseline (No-FiLM) | — | — | — | — | — |

## Limitations

- Trained on a single public dataset (233 patients) — not validated on external cohorts or scanner types.
- 2D slice-based, not volumetric (3D) segmentation.
- Pixel-to-mm² spacing is an approximation unless refined with real DICOM metadata.
- No clinical validation. **Not for diagnostic use.**

## Roadmap

- [ ] 3D/volumetric extension across slice stacks
- [ ] External validation on additional datasets
- [ ] ONNX export for lightweight inference
- [ ] Web front-end for interactive scan review

## License

Dataset usage is subject to the original [Figshare Brain Tumor Dataset](https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset) license and citation terms (Cheng et al., 2017).
