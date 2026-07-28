<div align="center">

# 🦙 Fine-Tuning TinyLlama with QLoRA

**Teach a 1.1B parameter model new tricks — without needing a data center.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.51-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-blueviolet)
![Colab](https://img.shields.io/badge/Runs%20on-Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-Unspecified-lightgrey)

</div>

---

## ✨ What is this?

A clean, beginner-friendly pipeline that fine-tunes **TinyLlama-1.1B-Chat** on the **Dolly-15k** instruction dataset using **QLoRA** — a technique that fine-tunes huge models by training a tiny set of extra weights instead of the whole thing.

> 💡 **In plain English:** instead of retraining all 1.1 billion parameters (expensive, slow, needs a huge GPU), this freezes the base model, loads it in 4-bit to save memory, and only trains a small "adapter" bolted onto it. You get a customized model using a fraction of the compute.

No PhD required. No massive GPU cluster. Just a free Colab GPU and this repo.

---

## 🚀 Why it's cool

| | |
|---|---|
| 🪶 **Lightweight** | 4-bit quantization means it fits on a free Colab GPU |
| 🧩 **Modular** | Every step — data, model, training, inference — lives in its own clean file |
| 📓 **Notebook-first** | Two ready-to-run notebooks, no setup headaches |
| 📊 **Visual feedback** | Auto-plots your training loss so you can see it actually learning |
| ⚖️ **Compare & contrast** | Runs base model vs. fine-tuned model side-by-side so you can *see* the improvement |

---

## 🗂️ Project Map

```
📦 Fine-Tuning-TinyLlama
├── 📁 src/
│   ├── 📁 data/          → load Dolly-15k → chat format → tokenize → clean
│   ├── 📁 tokenizer/      → loads & configures the tokenizer
│   ├── 📁 model/          → 4-bit quantization + LoRA adapter setup
│   ├── 📁 training/       → training loop (via 🤗 TRL's SFTTrainer)
│   └── 📁 inference/      → generate & compare responses
│
├── 📁 notebooks/
│   ├── 📓 setup.ipynb            ⭐ full step-by-step walkthrough
│   ├── 📓 run_pipelines.ipynb    ⚡ same result, way less code
│
├── 📁 outputs/final_adapter/     → your trained LoRA weights land here
├── 📁 fine_tunning_ploted_graph/ → 📈 training loss curve image
└── 📄 requirements.txt
```

---

## 🔧 How It Works — in 4 Steps

```
 1️⃣  DATA          2️⃣  MODEL              3️⃣  TRAIN            4️⃣  INFER
 ─────────         ─────────              ─────────           ─────────
 Dolly-15k    →    TinyLlama in 4-bit  →   SFTTrainer     →    Compare
 → chat format     + LoRA adapter          fine-tunes          base vs.
 → tokenized        (q/k/v/o proj)         only the             fine-tuned
                                            adapter              response
```

<details>
<summary><b>🔍 Click for the technical details</b></summary>

<br>

**1. Data prep** — Converts each Dolly-15k row (`instruction` + optional `context` → `response`) into a chat message list, applies the tokenizer's chat template, and strips unused columns.

**2. Model setup** — Loads TinyLlama with `BitsAndBytesConfig` (NF4, double quantization, fp16 compute), then attaches a LoRA adapter:
- Rank (`r`): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`

**3. Training** — `trl.SFTTrainer`, 1 epoch, batch size 1 with gradient accumulation of 4, learning rate `2e-4`, fp16 enabled.

**4. Inference** — Loads the saved adapter with `PeftModel`, generates text from both the base and fine-tuned models on the same prompt for comparison.

</details>

---

## 🏁 Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Open a notebook**

| Notebook | Best for |
|---|---|
| `notebooks/setup.ipynb` | Learning — runs every step individually |
| `notebooks/run_pipelines.ipynb` | Speed — one-liner pipeline calls |

**3. Or run it in code:**

```python
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME = "databricks/databricks-dolly-15k"

from src.tokenizer.tokenizer_loader import load_tokenizer
tokenizer = load_tokenizer(model_name=MODEL_NAME)

from src.data.pipeline import pipeline as data_pipeline
data_set = data_pipeline(dataset=DATASET_NAME, tokenizer=tokenizer)

from src.model.pipeline import pipeline as model_pipeline
base_model, lora_model = model_pipeline(model_name=MODEL_NAME)

from src.training.pipeline import pipeline as training_pipeline
trainer = training_pipeline(model=lora_model, dataset=data_set)

from src.inference.pipeline import pipeline as inference_pipeline
inference_pipeline(base_model=base_model, prompt="Explain Machine Learning in simple words.")
```

> ⚠️ A CUDA GPU is required — `bitsandbytes` 4-bit quantization is GPU-only. Both notebooks are built for Google Colab and mount Drive at `/content/drive/MyDrive/tinyllama_qlora`.

---

## 📈 What You Get

- ✅ A trained LoRA adapter in `outputs/final_adapter/` (small — not a full re-saved model)
- ✅ A training loss chart in `fine_tunning_ploted_graph/output.png`
- ✅ Side-by-side text generations so you can see exactly what fine-tuning changed

---

## 🐞 Known Issues

- `save_adapter.py` calls `.save()` on the tokenizer/trainer — use `.save_pretrained()` / `.save_model()` instead (already correct in `setup.ipynb`).
- `generate_basemodel_response.py` and `generate_finetuned_respose.py` are duplicate files.
- `dataset_preperation.ipynb` is an outdated draft with a broken import path.
- Import paths differ slightly between notebooks — run from the repo root with `src/` on the path.

---

<div align="center">

**Built for learning how LoRA fine-tuning actually works, one small file at a time.** 🦙

</div>