# EmoBot — MELD multimodal emotion recognition (Google Colab GPU)

A **Python-only** prototype (no `.ipynb` files). It predicts one of MELD's 7 emotion categories from utterance text and acoustic/visual signals from a synchronized MP4, then emits a timestamped JSON robot state and a short response.

## What is implemented

- Frozen pretrained **DistilBERT** text embeddings, computed on Colab GPU.
- 12 handcrafted acoustic descriptors (amplitude, spectral statistics, duration, etc.) from FFmpeg-decoded audio.
- 12 handcrafted visual descriptors (frame luminance, contrast, temporal motion, etc.) from sampled video frames.
- Two explicit modality-presence flags, a GPU-trained small PyTorch fusion MLP, and rule-based short responses.
- Official **train** for learning, **dev** for checkpoint selection, and untouched **test** for final metrics.
- CLI and Gradio demonstration, JSON inference, measured full-path latency script, and pytest checks.

**Important limitations:** The video descriptors are coarse appearance/motion features, **not a facial-expression recognizer**. The audio features are simple acoustics, not a speech-emotion pretrained encoder. DistilBERT is frozen; the fusion MLP is the only component trained here. The Gradio demo processes completed utterance clips in sequence, **not live frame-by-frame streaming**. Emotion labels are uncertain interpretations, not ground truth about a person's internal state. No model accuracy or latency is claimed until you run the scripts.

## Colab quick start (T4 GPU recommended)

In Colab, choose **Runtime > Change runtime type > T4 GPU**. Colab is only the execution environment; all inference is local to the Colab runtime, with no remote inference API. Paste these commands into Colab code cells (or a terminal). No notebook needs to be committed.

```bash
!git clone https://github.com/YOUR_USERNAME/emobot-colab.git
%cd emobot-colab
!pip install -r requirements.txt
!apt-get update -qq && apt-get install -y -qq ffmpeg
!nvidia-smi
```

Before you push to GitHub, upload the contents of this project ZIP to your repository and replace `YOUR_USERNAME` above.

### Download the MELD dataset

The official **MELD.Raw.tar.gz is approximately 10.9 GB compressed**. Extraction takes additional disk space and time. Ensure Colab has enough free space; you may need to delete the main archive after extraction, or store files in Drive. The downloader obtains the upstream official archive and unpacks nested train/dev/test media archives.

```bash
!python -m scripts.download_meld --out data
```

If the default download host fails, retry with the official mirror:

```bash
!python -m scripts.download_meld --out data --url https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
```

**Warning:** if a download is interrupted and the archive is corrupted, remove `data/MELD.Raw.tar.gz` before retrying. Do not commit dataset files or clips to GitHub. MELD contains third-party TV footage; verify applicable permissions for any redistribution.

### Prepare all three partitions

```bash
!python -m scripts.prepare --data data/MELD.Raw --cache cache --batch-size 32
```

This caches text embeddings and audio/video descriptors. Video extraction is CPU-heavy and can take substantial time; GPU accelerates the DistilBERT embedding stage. Media availability counts are printed so missing clips are visible. A debug run with `--limit 100` is allowed for testing code but **must not be reported as full-dataset performance**. Delete cached `.npz` files before switching from debug to full training.

### Train on GPU, select with dev only

```bash
!python -m scripts.train --cache cache --out artifacts --epochs 15
```

The selected checkpoint is `artifacts/best.pt`, chosen using **dev macro-F1**, never test. Media standardization is fit **only on train**. Run the final held-out evaluation once the architecture and hyperparameters are fixed:

```bash
!python -m scripts.evaluate --cache cache --checkpoint artifacts/best.pt --out artifacts/test_metrics.json
```

Outputs include accuracy, macro-F1, weighted-F1, per-emotion precision/recall/F1, and confusion matrix. The script checks train/dev/test key disjointness. Do not use test results to repeatedly tune hyperparameters.

### End-to-end inference example

Choose a test row and its corresponding MP4, for example:

```bash
!python -m scripts.demo --checkpoint artifacts/best.pt --text "I cannot believe this!" --video data/MELD.Raw/output_repeated_splits_test/dia0_utt0.mp4
```

The sample filename is only illustrative; **verify that the clip exists and that the text matches its annotation**. To get an actual matched test example from the dataset:

```python
from emobot.data import read_split
from pathlib import Path
r = read_split('data/MELD.Raw', 'test')
r = r[r.clip.map(lambda x: Path(x).exists())]
print(r[['Utterance', 'Emotion', 'clip']].head(3).to_string(index=False))
```

The demo returns JSON with `timestamp`, `emotion`, `confidence`, `tags`, `response`, and `latency_ms`. The output confidence is a softmax score, **not calibrated probability**. Media flags explicitly indicate absent streams.

### Interactive UI

```bash
!python -m scripts.app --checkpoint artifacts/best.pt
```

Colab networking may not expose a local Gradio URL; use `share=True` **only if you understand that it creates a public link** (and do not expose private media). Alternatively run `scripts.demo` in Colab for a reliable demonstration.

### Latency and hardware benchmark

```bash
!python -m scripts.benchmark --checkpoint artifacts/best.pt --video PATH_TO_REAL_TEST_CLIP.mp4 --repeats 10
!pytest -q
```

`benchmark.py` reports warmed-up end-to-end median and p95 latency, process RAM, and GPU peak allocated memory. **Specify a real MP4** to benchmark the actual multimodal path. Record the Colab GPU type from `!nvidia-smi`, RAM from `!free -h`, and disk from `!df -h`. The `evaluate.py` speed figure measures **fusion only**, not the entire inference pipeline.

## Architecture

```text
MELD utterance text -> frozen DistilBERT (768d) ----\
MELD MP4 audio -> FFmpeg + acoustic statistics (12d) -+-> normalize media (train-only) -> fusion MLP -> 7 emotions -> JSON + response template
MELD MP4 video -> OpenCV frame statistics (12d) ----/
                              + 2 presence flags
```

The inference model includes roughly 66M DistilBERT parameters plus a fusion MLP of about 204K parameters. Tokenizer, FFmpeg and handcrafted media features do not add learned parameters. This is **well below 6 billion total learned parameters**; for a formal audit, count the installed encoder and fusion weights with the code below:

```python
from emobot.pipeline import EmoBot
bot = EmoBot('artifacts/best.pt')
print('Total learned parameters:', sum(p.numel() for p in bot.encoder.model.parameters()) + sum(p.numel() for p in bot.model.parameters()))
```

## Definition of real-time

**Turn-level real-time:** accept a completed user utterance (text + a short synchronized video/audio clip) and return a robot state and response before the next conversational turn. This is a *target*, not a demonstrated latency guarantee; report measured median/p95 on the actual GPU. It does **not** process live audio/video chunks continuously or generate token-streaming text. The app can process successive inputs but does not keep dialogue history.

## Data integrity and responsible evaluation

- Official split CSVs: `train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv`.
- Clips matched by `Dialogue_ID` and `Utterance_ID`, e.g. `dia6_utt1.mp4`.
- If a clip is missing/corrupt, its audio/video features are zeros with corresponding presence flags 0; rows are not silently removed.
- Cached test features may be generated in advance, but test labels are not used for optimization or model selection.
- Compare full model with a text-only ablation if time permits (not included as a measured experiment).
- Avoid claiming emotional certainty or presenting robot replies as therapeutic support.

## Files

```text
emobot/                 data.py, features.py, model.py, pipeline.py
scripts/download_meld.py  upstream dataset downloader + extraction
scripts/prepare.py        GPU text embeddings + media feature caching
scripts/train.py          train/dev training + checkpoint selection
scripts/evaluate.py       held-out test metrics (no fitting)
scripts/demo.py           JSON CLI inference
scripts/app.py            Gradio demo
scripts/benchmark.py      full-path runtime measurement
tests/test_contract.py    shape/missing-media tests
requirements.txt
.gitignore
```

## Before submission

1. Execute all commands in a GPU Colab session, fix any environment-specific issues, and save `artifacts/test_metrics.json` plus training history.
2. Record GPU model, CPU/RAM, GPU memory, inference p50/p95, training duration, and observed test metrics in this README. **Do not invent results.**
3. Demonstrate a correctly matched held-out test clip through both JSON and robot response. If possible, add a screenshot/GIF (respect footage rights).
4. Include a model checkpoint via a release or download link if its size permits; the README must explain how to reproduce it. Do not upload MELD videos.
5. Disclose any AI-assisted code and your own verification and edits.

## External components

MELD: Poria et al., *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations*, ACL 2019. Official repository: https://github.com/declare-lab/MELD ; official download: https://affective-meld.github.io/ . DistilBERT: https://huggingface.co/distilbert/distilbert-base-uncased . PyTorch, Hugging Face Transformers, OpenCV, FFmpeg, scikit-learn and Gradio are used. Project code was AI-assisted and must be reviewed and validated by the submitter.
