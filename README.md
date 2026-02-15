# [ICLR2026] Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty

Official implementation of the paper [**Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty**](https://openreview.net/forum?id=aRzEtK9Ite).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Analysis](#analysis)
- [Export Checkpoints](#export-checkpoints)
- [Notes](#notes)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Overview

We identify the over-reflection phenomenon that reduces reasoning efficiency and show it correlates with problem complexity. Based on this insight, we propose **ARLCP**, a dynamic RL method that encourages LRMs to stop unnecessary reflection via an adaptive reward strategy conditioned on complexity. Experiments on math reasoning benchmarks demonstrate that ARLCP reduces average response length while improving accuracy.


## Installation

This project is built upon the [veRL](https://github.com/volcengine/verl) framework.

1. Clone this repository:
```
git clone https://github.com/ZeweiYu1/ARLCP.git
cd ARLCP
```

2. Create environment:
```
conda create -n arlcp python=3.10
conda activate arlcp
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Dataset

Convert JSON to the parquet format required by veRL:
```
bash scripts/preprocess_dataset.sh
```

For custom repeat counts or paths, see `src/preprocess_dataset.py`.

## Training

We provide training scripts for 1.5B and 7B models:
```
# 1.5B
bash scripts/run_ARLCP_1.5b.sh

# 7B
bash scripts/run_ARLCP_7b.sh
```

## Evaluation

Use `run_eval.sh` to run evaluation only:

```
bash scripts/run_eval.sh
```

Evaluation datasets are controlled by `data.val_files` in the scripts (default: GSM8K, MATH, AIME24/25, AMC23).
During training, veRL will automatically evaluate every `trainer.test_freq` step.

## Analysis

Analyze JSONL evaluation logs for accuracy, average length, and reflection keyword counts:

```
python src/data_analysis.py \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --data_path tests/PROJECT_NAME/EXP_NAME/step.jsonl
```

## Export Checkpoints

Convert FSDP/Megatron checkpoints to HuggingFace format:
```
python scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir /path/to/checkpoints/.../actor \
  --target_dir /path/to/merged_hf_model
```

See `scripts/model_merger.py` header for more usage.

## Notes

- Set `MODEL_PATH` correctly in `scripts/run_ARLCP_*.sh` 
- Training data assumes `<think>...</think>` formatting; ensure your model chat template is compatible.

## Acknowledgement

ARLCP builds upon [veRL](https://github.com/volcengine/verl), [AdaptThink](https://github.com/THU-KEG/AdaptThink) and [DeepScaler](https://github.com/agentica-project/rllm), and utilizes [vLLM](https://github.com/vllm-project/vllm) for inference. The models are trained based on [DeepSeek-R1-Distill-Qwen](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B). We sincerely appreciate their contributions to the open-source community.

## Citation

If you find this repo useful, please cite the paper:

```bibtex
@misc{yu2026stopunnecessaryreflectiontraining,
      title={Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty}, 
      author={Zewei Yu and Lirong Gao and Yuke Zhu and Bo Zheng and Sheng Guo and Haobo Wang and Junbo Zhao},
      year={2026},
      eprint={2602.12113},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.12113}, 
}
```
