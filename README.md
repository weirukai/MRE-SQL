# Multi-Ranker Ensemble for Robust Candidate Selection in Text-to-SQL

This repository contains the code and data for the paper "Multi-Ranker Ensemble for Robust Candidate Selection in Text-to-SQL"



## Running the Code

After preprocessing the databases, generate SQL queries for the BIRD dataset by choosing a configuration:

1. **Generate Candidates with CHESS**: 
> Following CHESS at https://github.com/ShayanTalaei/CHESS

2. **Run selection Method**
```python
python selection_evaluation.py --data_mode dev --data_path dev --result_directory CHESS --base_url http://xx.xx.xx.xx:8000
--config ./run/configs/selector_config.yaml --num_workers 8 --engine_name qwen2.5-coder-7b
```
