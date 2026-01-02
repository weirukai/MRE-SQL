# 🎯 Multi-Ranker Ensemble for Robust Candidate Selection in Text-to-SQL

This repository contains the code and data for the paper "Multi-Ranker Ensemble for Robust Candidate Selection in Text-to-SQL"

> **(𝑖) We propose multiple LLM-based candidate ranking strategies to enable multi-path reasoning. It leverages complementary signals from all strategies to improve robustness and reduce over-reliance on any single method.
> (𝑖𝑖) We design an uncertainty-aware aggregation mechanism that dynamically weights each ranker according to its confidence and consistency. It improves selection stability and reduces overconfidence in unreliable judgments.**

## 📦 Project Structure Overview

```
Multi-Verifier-Ensemble/
├── data/                     # 🗃️ Datasets & Preprocessed DBs
│
├── run/                      # ⚙️ Configuration & Execution Scripts
│   ├── configs/              # YAML configs for temperatures...
│   │   └── selector_config.yaml
│
├── src/                      # 🧠 Core Logic & Workflow
│   ├── database_utils/       # Database connection & metadata extraction
│   ├── llm/                  # LLM wrappers & inference utilities
│   ├── runner/               # Main execution orchestrator of the generator to produce candidates
│   ├── workflow/             # CHESS integration + ensemble builder
│   ├── ensemble_builder.py    # Multi-verifier ensemble logic
│   ├── selection_builder.py   # Candidate scoring & ranking
│   ├── generation.py          # Query generation (CHESS integration)
│   ├── preprocess.py          # Schema/value preprocessing
│   ├── few_shot.py           # Prompt templates for few-shot learning
│   ├── ......
│
├── templates/                # 📄 Prompt Engineering Hub
├── README.md                 # 📘 This document
└── ......
```




## Running the Code

After preprocessing the databases, generate SQL queries for the BIRD dataset by choosing a configuration:

1. **Generate Candidates with CHESS**: 
> Following CHESS at https://github.com/ShayanTalaei/CHESS

2. **Run selection Method**
```python
python selection_evaluation.py --data_mode dev --data_path dev --result_directory CHESS --base_url http://xx.xx.xx.xx:8000
--config ./run/configs/selector_config.yaml --num_workers 8 --engine_name qwen2.5-coder-7b
```
