# Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official code for the paper:

> **Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence**
>
> *Wanying Ren, Xin Song, Futing Wang, Guoxiu He, Aixin Sun*

## Overview

Parameter-based knowledge editing methods update LLM knowledge via localized weight modifications. This paper shows that such edits are fundamentally limited by **dimensional collapse** in LLM representation spaces: low-variance directions amplify even small perturbations, leading to global interference and reasoning collapse.

We provide:
1. **A geometric theoretical framework** (Section 4) characterizing how localized edits destabilize representations
2. **Comprehensive empirical evaluation** (Section 5) under realistic settings across multiple models, methods, datasets, and edit counts
3. **SCR baseline**: a simple retrieval-based method that consistently outperforms all parameter-editing approaches

## Directory Structure

```
├── easyeditor/                  Core editing framework (pruned to paper methods)
│   ├── editors/                 BaseEditor and editing logic
│   ├── models/                  Method implementations (ROME, MEMIT, MEND, AlphaEdit, WISE, PMET, LoRA, FT, GRACE, IKE)
│   ├── trainer/                 Training infrastructure
│   ├── evaluate/                Evaluation utilities (LLM judge, metrics)
│   ├── dataset/                 Dataset loaders (ZsRE, WikiCounterfact, KnowEdit)
│   └── util/                    Utilities (hparams, nethook, generate)
│
├── configs/                     Hyperparameter configurations (paper models only)
│   ├── ROME/  MEMIT/  MEND/  AlphaEdit/  WISE/  PMET/  LoRA/  FT/  GRACE/  IKE/
│
├── scr/                         SCR retrieval baseline
│   ├── edit_rag.py              Main SCR implementation
│   ├── edit_rag_reasoning.py    Reasoning model experiments
│   ├── edit_rag_event.py        Event knowledge experiments
│   └── prepare_requests.py      Data preprocessing
│
├── experiments/                 Experiment entry points
│   ├── run_param_edit.py        RQ1: Parameter editing (single + sequential)
│   ├── run_scr_baseline.py      RQ1: SCR retrieval baseline
│   ├── run_reasoning_edit.py    RQ2: Reasoning LLM editing
│   ├── run_event_edit.py        RQ3: Event knowledge editing
│   └── run_efficiency.py        RQ4: Time efficiency measurement
│
├── theory/                      Theory analysis code (Section 4 + Appendix B)
│   ├── svd_analysis.py          Singular value decomposition analysis
│   ├── relative_change_rate.py  Rk directional relative change rate
│   ├── principal_stability.py   Principal component stability under edits
│   └── hidden_states.py         Hidden state extraction
│
├── evaluation/                  Standalone evaluation module
│   ├── llm_judge.py             Qwen2.5-72B semantic consistency judge
│   └── metrics.py               Reliability, Generalization, Locality, Portability
│
├── visualization/               Paper figure reproduction
│   ├── fig2_relative_change.py   Figure 2: Rk distribution
│   ├── fig3_editing_performance.py  Figure 3: Editing performance vs. N
│   ├── fig4_event_editing.py     Figure 4: Event knowledge results
│   └── fig5_efficiency.py        Figure 5: Time efficiency trade-off
│
├── scripts/                     Utility scripts
│   ├── serve_judge.sh           Launch LLM judge vLLM service
│   └── data_prepare/            Dataset preparation
│
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

Datasets are available from the [KnowEdit benchmark](https://huggingface.co/datasets/zjunlp/KnowEdit):

```bash
# ZsRE dataset
python scripts/data_prepare/prepare_zsre.py --output_dir ./data/zsre

# WikiData-counterfact dataset
python scripts/data_prepare/prepare_wiki_counterfact.py --output_dir ./data/wiki_counterfact

# ELKEN event dataset
python scripts/data_prepare/prepare_elken.py --output_dir ./data/elken
```

Place the downloaded dataset files in the respective directories.

### 3. Launch LLM Judge Service (for semantic evaluation)

```bash
# Start vLLM serving Qwen2.5-72B-Instruct on port 21910
bash scripts/serve_judge.sh
```

### 4. Run Experiments

```bash
# RQ1: Parameter editing (sequential, N=100, ZsRE)
python experiments/run_param_edit.py \
    --method AlphaEdit \
    --model llama3.1-8b \
    --config_path configs/AlphaEdit/llama3.1-8b.yaml \
    --dataset zsre \
    --data_path ./data/zsre/zsre-test-all.json \
    --N 100 \
    --sequential \
    --eval_mode llm_judge

# RQ1: SCR retrieval baseline
python experiments/run_scr_baseline.py \
    --model_path /path/to/llama3.1-8b \
    --retriever_path /path/to/contriever-msmarco \
    --dataset_path ./data/wiki_counterfact/test_cf.json \
    --memory_path ./data/wiki_counterfact/wiki_counterfact-test-all-sentence.jsonl \
    --top_k 5 \
    --edit_scene sequential

# RQ2: Reasoning LLM editing
python experiments/run_reasoning_edit.py \
    --method AlphaEdit \
    --model deepseek-r1-distill-llama-8b \
    --config_path configs/AlphaEdit/llama3-8b-distill.yaml \
    --data_path ./data/zsre/zsre-test-all.json \
    --N 100 \
    --reasoning_bench gsm8k math gpqa

# RQ3: Event knowledge editing
python experiments/run_event_edit.py \
    --method AlphaEdit \
    --model llama3.1-8b \
    --config_path configs/AlphaEdit/llama3.1-8b.yaml \
    --data_path ./data/elken/test_processed.json \
    --N 100

# RQ4: Efficiency measurement
python experiments/run_efficiency.py \
    --method ROME \
    --model llama3.1-8b \
    --config_path configs/ROME/llama3.1-8b.yaml
```

## Evaluated Methods

| Category | Methods | Description |
|----------|---------|-------------|
| Locate-then-edit | ROME, MEMIT, PMET, AlphaEdit, FT-L | Locate knowledge-associated parameters, then edit |
| Meta-learning | MEND | Learn parameter change patterns via hypernetwork |
| Additional parameter | WISE, LoRA (AdaLoRA) | Store new knowledge in external parameter components |
| External memory | GRACE, IKE, SCR | Retrieve from external store without modifying parameters |

## Evaluated Models

- Llama-2-7B-Chat
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct
- DeepSeek-R1-Distill-Llama-8B
- Llama-2-13B (large-scale verification)
- Qwen3-14B (large-scale verification)

## Evaluation Dimensions

| Dimension | Definition | Metric |
|-----------|-----------|--------|
| **Reliability** | Does the model output the updated target? | LLM judge accuracy |
| **Generalization** | Does it generalize to paraphrased prompts? | LLM judge accuracy |
| **Locality** | Are unrelated behaviors preserved? | Pre/post-edit consistency |
| **Portability** | Does edited knowledge support downstream reasoning? | LLM judge accuracy |

## Theory: Key Results

- **Theorem 4.5** (Relative Amplification): In low-singular-value directions, edit perturbations become disproportionately large: $R_{\min} = \sqrt{n}\varepsilon / \sigma_{\min}$
- **Sequential accumulation**: Under coherent editing, distortion grows linearly: $R^{(T)}_{\min} \approx T \cdot R^{(1)}_{\min}$
- **Empirical validation**: Significant Spearman correlations between $R_k$ statistics and editing performance degradation

Reproduce theory results:
```bash
cd theory/
python svd_analysis.py --model_path /path/to/model
python relative_change_rate.py --data_dir ./ --output_dir ./results/
python principal_stability.py --data_dir ./ --output_dir ./results/
```

## Citation

If you use this code, please cite:

```bibtex
@article{ren2025revisiting,
  title={Revisiting Parameter-Based Knowledge Editing in Large Language Models:
         Theoretical Limits and Empirical Evidence},
  author={Ren, Wanying and Song, Xin and Wang, Futing and He, Guoxiu and Sun, Aixin},
  year={2025},
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This codebase builds on [EasyEdit](https://github.com/zjunlp/EasyEdit), an easy-to-use knowledge editing framework for LLMs. We thank the EasyEdit authors for their excellent work.
