# Running Local LLMs Behind Institutional Firewalls

**SIIM 2026 Learning Lab — LL4022**
Hands-on materials for a secure clinical AI learning lab.

Ghulam Rasool, PhD · Aakash Tripathi, PhD · Asim Waqas, PhD — Moffitt Cancer Center

---

This repo contains the scripts and **synthetic** clinical datasets for the
hands-on portion of the Learning Lab. Everything here runs on local hardware
and talks only to a local model server — **no clinical text ever leaves the
machine, and no real patient data or PHI is used anywhere in this repo.**

## What's in here

```
.
├── summarize_report.py      # Lab 3 — radiology report summarization
├── extract_pathology.py     # Lab 3 — pathology -> structured JSON
├── utils/ollama_client.py   # tiny stdlib client for the local Ollama server
├── data/
│   ├── radiology/           # synthetic CT / MRI / X-ray reports
│   └── pathology/           # synthetic path reports (breast, lung, colon, prostate)
├── requirements.txt
└── notebooks/
    └── SIIM_LocalLLMs_Backup.ipynb   # cloud fallback (Ollama + OpenWebUI)
```

## Lab 0 — Setup check

Free accounts: [Hugging Face](https://huggingface.co/join) · [GitHub](https://github.com/signup)

Software:
- **Git**, **Python 3.10+**
- **Ollama** — https://ollama.com/download
- **OpenWebUI** — `pip install open-webui` (needs Python 3.11)
- **vLLM** (optional, for the scaling section) — `pip install vllm`

Clone this repo:

```bash
git clone https://github.com/lab-rasool/SIIM.git
cd SIIM
```

## Lab 1 — Your first local model

```bash
ollama pull llama3.2          # ~2 GB, one time
ollama run llama3.2           # chat in your terminal
curl localhost:11434/api/tags # confirm it's running locally
```

Turn off Wi-Fi — the model keeps answering. That's the point.

## Lab 2 — A private ChatGPT on your laptop

```bash
open-webui serve              # then open http://localhost:8080
```

OpenWebUI auto-detects everything Ollama has pulled. Accounts and chats stay
on your machine.

## Lab 3 — Clinical text workflows

These scripts use **only the Python standard library** — no `pip install`
needed. They read the synthetic reports in `data/` and send them to your local
Ollama server.

**Radiology — summarization:**

```bash
python summarize_report.py                          # all reports, default model
python summarize_report.py --report data/radiology/ct_chest_001.txt
python summarize_report.py --compare llama3.2 qwen2.5:7b   # small vs larger
```

**Pathology — structured extraction:**

```bash
python extract_pathology.py                         # all reports -> JSON
python extract_pathology.py --report data/pathology/path_lung_002.txt --out lung.json
```

The pathology script asks the model for valid JSON and then **validates** it:
it checks the expected fields are present and well-typed, and warns you when
the model leaves something out — a small but realistic taste of the guardrails
you need before trusting extraction in production.

Point either script at a different server with `--host` or the `OLLAMA_HOST`
environment variable.

## Scaling up with vLLM (slide 16)

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct
curl localhost:8000/v1/chat/completions ...   # OpenAI-compatible
```

## Backup: cloud Colab instance

If a laptop won't cooperate during the workshop, `notebooks/SIIM_LocalLLMs_Backup.ipynb`
spins up Ollama **and** OpenWebUI on a free Google Colab GPU and gives you a
public URL to a private ChatGPT-style interface — plus it can run the Lab 3
clinical workflows in the cloud. See the notebook's intro cell for the
one-thing-to-know caveat: a Colab instance is **for demos with synthetic data
only** — it is a public cloud machine and must never see real PHI.

---

*All clinical text in this repository is synthetic. No patient data or PHI is
used or shared.*
