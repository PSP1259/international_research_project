# Confidence without Accuracy? 
### A Comparative Analysis of the Calibration Reliability of Large Language Models across English and German Knowledge Domains

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)

## 📌 About The Project

While the **MMLU-Pro** benchmark provides a robust evaluation of factual accuracy across Large Language Models (LLMs), existing results rarely assess the *calibration of model confidence* or investigate *language-dependent calibration effects*. 

LLMs are predominantly trained on English data. This project investigates the hypothesis that LLMs transfer their high self-confidence learned in English to the German language, without maintaining the same level of factual accuracy, leading to systematic overconfidence (Calibration Error) in German.

**Central Research Question:**
*To what extent does the calibration reliability of large language models differ between English and German language contexts?*

Additionally, this project tracks the **business impact of API costs**, specifically comparing the token usage (input/output) between English queries and their German translations.

## 🚀 Features & Methodology

- **Bilingual Evaluation Pipeline:** Automated processing of English MMLU-Pro queries and deep-translated German equivalents (via DeepL API).
- **Domain Selection:** Tests models across 14 professional/academic categories, grouping them into STEM (logical & universal) and Humanities (contextual & cultural).
- **Model Benchmarking:** Comparative analysis between top-tier models (e.g., OpenAI's GPT series vs. Google's Gemini series).
- **Calibration Tracking:** Measures both *Self-Reported Confidence* (0-100%) and *Logit-based Confidence* against actual answer correctness.
- **Cost & Lateny Analysis:** Tracks exact token count and response time (latency) to evaluate the economic impact of language on API usage.

## 📊 Dataset & Sample Size

The pipeline uses a scientifically validated sample of the **MMLU-Pro (Massive Multitask Language Understanding – Professional)** dataset.
- **Total Sample Size:** ~1,260 questions
- **Categories include:** Math, Physics, Chemistry, Law, Engineering, Economics, Health, Psychology, Business, Biology, Philosophy, Computer Science, and History.

## 🧮 Tracked Variables

| Variable | Type | Description |
| :--- | :--- | :--- |
| `Question_ID` | Nominal | Unique ID for traceability. |
| `Domain` | Nominal | Subject category (e.g., Math, Law). |
| `Language` | Nominal | English (Original) vs. German (DeepL). |
| `Tool` | Nominal | Selected LLM (ChatGPT / Gemini). |
| `Answer` | Nominal | Final choice of the model (A, B, C...). |
| `Correct` | Binary | 1 = correct, 0 = incorrect (matched with Gold label). |
| `Conf_Self` | Metric | Self-reported model confidence (0–100%). |
| `Conf_Logit` | Metric | Probability derived from Logprobs. |
| `Token_Count` | Discrete | Processed tokens (`input_tokens` + `output_tokens`). |
| `Latency_Sec` | Metric | Response time in seconds. |
| `Reasoning_Chain` | Text | Model's standardized justification (1-2 sentences). |

## 📁 Repository Structure

- `/smoke/` - Initial testing scripts and short trial runs.
- `/mini_run/` - Small sample execution to validate the pipeline stability.
- `/real_run/` - Final production scripts for the full ~1,260 question dataset.
- `/data/` - Input CSVs, MMLU-Pro JSONL files, and final output datasets.
- `.env.example` - Template for required API keys.
- `requirements.txt` - Python dependencies.

## 🛠️ Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PSP1259/international_research_project.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Setup:**
   Rename `.env.example` to `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   DEEPL_API_KEY=your_key_here
   ```

## 📈 Statistical Analysis Plan

The final data is modeled using descriptive and inferential statistics:
- **Descriptive:** Mean Accuracy, Mean Confidence, Token comparison, Latency comparison.
- **Calibration Metrics:** Bin-Plots, Expected Calibration Error (ECE), Brier Score.
- **Inference Statistics:** Paired t-tests for mean differences, Chi-square for accuracy, and ANOVA (Language × Domain). Significance is marked conventionally (* p < 0.10, ** p < 0.05, *** p < 0.01).
