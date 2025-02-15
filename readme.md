# Task-Decomposed Instruction Tuning for Relation Extraction: Auto-Selecting Optimal LLM Data-Training-Inference Combinations

This repository contains the code and resources for the paper **"Task-Decomposed Instruction Tuning for Relation Extraction: Auto-Selecting Optimal LLM Data-Training-Inference Combinations"**. The project focuses on optimizing the combination of data, training, and inference strategies for large language models (LLMs) in relation extraction tasks.

---

## Repository Structure

The repository is organized as follows:

```plaintext
.
├── analysis_exp/          # Results analysis scripts and outputs
├── datas/                 # Datasets used for training and evaluation
├── eval/                  # Code for model evaluation
├── llm_models/            # Pre-trained base LLMs
├── train/                 # Training code and scripts
├── trained_models/        # Checkpoints of trained models
├── README.md              # This file
└── requirements.txt       # List of dependencies
```

### Directory Descriptions

1. **`analysis_exp/`**:
   - Contains scripts and notebooks for analyzing experimental results.
   - Includes visualizations, statistical analyses, and performance metrics.

2. **`datas/`**:
   - Stores datasets used for training and evaluation.
   - Each dataset is organized into subdirectories with clear naming conventions.

3. **`eval/`**:
   - Contains code for evaluating trained models.
   - Includes scripts for computing metrics such as F1 score, precision, and recall.

4. **`llm_models/`**:
   - Houses pre-trained base LLMs used in the experiments.
   - Models are stored in a format compatible with the training pipeline.

5. **`train/`**:
   - Includes scripts and configurations for training models.
   - Supports various training strategies and hyperparameter configurations.

6. **`trained_models/`**:
   - Stores checkpoints of trained models.
   - Each checkpoint is labeled with the corresponding experiment ID and timestamp.

---

## Environment Configuration

| Package            | Version   |
|--------------------|-----------|
| `torch`            | 2.1.2+cu121 |
| `transformers`     | 4.38.1    |
| `datasets`         | 2.17.1    |
| `peft`             | 0.8.2     |
| `deepspeed`        | 0.12.6    |
| `accelerate`       | 0.27.2    |
| `numpy`            | 1.23.4    |
| `scikit-learn`     | 1.2.1     |
| `tqdm`             | 4.64.1    |
| `sentencepiece`    | 0.1.99    |
| `evaluate`         | 0.4.1     |

For the full list of dependencies, see `requirements.txt`.

## Usage

### Training

To train a model, navigate to the `train/` directory and run the appropriate script:

```bash
cd ./train/scripts/bidirection_ade_deepseek_7b_notype
bash 0_model_1.sh
```

### Evaluation

To evaluate a trained model, use the scripts in the `eval/` directory:

```bash
cd ./eval/eval_bidirection
bash run_script.sh
```

### Analysis

For result analysis, explore the notebooks in the `analysis_exp/` directory.
