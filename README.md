# Incremental Mixture of Experts: Continual Learning for Object Detection in Forestry Scenarios

This repository contains the official PyTorch implementation and experimental configurations for our paper, **"Incremental Mixture of Experts: Continual Learning for Object Detection in Forestry Scenarios"**.

Our work introduces **Incre-MoE**, a novel framework designed to mitigate catastrophic forgetting in continual learning for object detection. It is particularly effective in challenging, real-world scenarios that involve both domain increments (e.g., learning from a new sensor type like thermal) and class increments (e.g., learning a new object category within an existing domain).

**[Link to Paper (Coming Soon)]**

## Abstract

The complexity of object detection in forestry scenarios poses significant challenges to continual learning. In this work, we propose the **Incremental Mixture of Experts (Incre-MoE)** framework to alleviate long-term forgetting. Our approach explicitly distinguishes between domain-incremental and class-incremental tasks and applies a specialized strategy for each:
1.  A **Frequency-Domain Router (FDR)** analyzes the statistical signature of new tasks to automatically determine their nature without needing explicit labels.
2.  For domain-incremental tasks, **Adaptive Expert Expansion (AEE)** creates new, isolated expert modules, preventing catastrophic forgetting by design.
3.  For class-incremental tasks, **Spherical Linear Interpolation Guided Vector Fusion (SGVF)** efficiently merges new knowledge into an existing expert while preserving its prior capabilities.

Through extensive experiments, our proposed framework consistently outperforms previous state-of-the-art approaches in continual learning for object detection.

## Reproducibility and Experimental Workflow

To ensure full reproducibility, we provide the complete source code, configuration files, and a detailed description of the experimental process.

### Experimental Workflow Diagram

Our continual learning experiments follow a sequential, task-by-task process. The diagram below illustrates a typical multi-task learning sequence as described in our paper (e.g., Larch -> Smoke -> Car -> Vehicle -> Person).
```
[Start]
   |
   +--> [Initial State]
        - Model: Base Pre-trained Incre-MoE (E₀)
        - experts.json: Contains only Expert 0
        - weights: latest_model.pth (contains E₀ weights)
   |
   +--> [Task 1: Larch (RGB) -> Run train_workflow.sh]
   |    1. FDR analyzes 'Larch' data.
   |    2. Decision: Domain-Incremental (new scene, distinct from base).
   |    3. AEE activates: A new Expert 1 (E₁) is created and trained.
   |    4. Result: latest_model.pth now contains E₀ and E₁ weights.
   |             experts.json is updated with Expert 1.
   |
   +--> [Task 2: Smoke (RGB) -> Run train_workflow.sh]
   |    1. FDR analyzes 'Smoke' data.
   |    2. Decision: Domain-Incremental (distinct from base E₀ and Larch E₁).
   |    3. AEE activates: A new Expert 2 (E₂) is created and trained.
   |    4. Result: latest_model.pth now contains E₀, E₁, E₂.
   |             experts.json is updated with Expert 2.
   |
   +--> [Task 3: Car (Thermal) -> Run train_workflow.sh]
   |    1. FDR analyzes 'Car' data from a new thermal sensor.
   |    2. Decision: Domain-Incremental (new domain).
   |    3. AEE activates: A new Expert 3 (E₃) is created and trained for the thermal domain.
   |    4. Result: latest_model.pth now contains E₀, E₁, E₂, E₃.
   |             experts.json is updated with Expert 3.
   |
   +--> [Task 4: Vehicle (Thermal) -> Run train_workflow.sh]
   |    1. FDR analyzes 'Vehicle' data.
   |    2. Decision: Class-Incremental (most similar to Expert 3, as they share the thermal domain).
   |    3. SGVF activates: A temporary student expert is trained on 'Vehicle' data.
   |    4. Fusion: The student's knowledge is fused into the teacher (E₃).
   |    5. Result: latest_model.pth is updated with the fused weights for E₃.
   |             Total expert count does not change.
   |
   +--> [Task 5: Person (Thermal) -> Run train_workflow.sh]
   |    1. FDR analyzes 'Person' data.
   |    2. Decision: Class-Incremental (again, most similar to the thermal expert E₃).
   |    3. SGVF activates: Another temporary student expert is trained on 'Person' data.
   |    4. Fusion: The new knowledge is fused into the existing teacher (E₃).
   |    5. Result: E₃'s weights are updated again. Total expert count remains unchanged.
   |
   +--> [End of Sequence]
        - Final model contains multiple specialized experts (e.g., for Larch, Smoke, and a comprehensive Thermal expert).
```

## Installation

Follow these steps carefully to set up the environment for **CUDA 12.4** and **Python 3.10**.

### Step 1: Clone Repository and Submodules

```bash
git clone https://github.com/your-username/Incre-MoE.git
```

### Step 2: Environment Setup (Version Control)

To ensure a completely identical software environment, we recommend using a virtual environment.
*   **Python Version:** 3.10
*   **PyTorch Version:** 2.4.0 (for CUDA 12.4)
*   **Framework:** The base model is built upon Grounding DINO.

```bash
# Create and activate a virtual environment using Python 3.10
python3.10 -m venv venv
source venv/bin/activate

# 1. Install PyTorch with CUDA 12.4 support
# Note: Ensure you install the correct PyTorch version for your specific CUDA toolkit.
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 2. Install all other specified dependencies
pip install -r requirements.txt
```

### Step 3: Compile GroundingDINO CUDA Extensions

This step is critical for the model to function correctly.

1.  **Verify CUDA Setup:** The compilation requires your `CUDA_HOME` environment variable to be set correctly. If it's not, you may encounter a `NameError: name '_C' is not defined` error.
    ```bash
    # Check if CUDA_HOME is set
    echo $CUDA_HOME
    # If it's empty, find and set your CUDA path, e.g.:
    export CUDA_HOME=/usr/local/cuda-12.4
    ```

2.  **Build the Modules:** Navigate into the `groundingdino` directory and run the installation.
    ```bash
    cd Incre-MoE
    pip install -e .
    cd ..
    ```

### Step 4: Download Pre-trained Base Weights

Our continual learning process starts from our specifically prepared Incre-MoE base model, which will be released shortly.

```bash
mkdir -p weights

# The link to our pre-trained base model will be provided here upon publication.
# For now, this is a placeholder.
echo "Model weights for Incre-MoE base model (E₀) will be available soon."
# wget [Link_to_Incre-MoE_Base_Weights] -O weights/latest_model.pth
```

## Dataset Preparation

### Directory Structure & Preprocessing

Organize your data as follows:
```
datasets/
└── smoke_task/
    ├── train.csv
    └── train_images/
        └── ...
```

Our framework applies two distinct preprocessing steps:

1.  **For the FDR Module:** To create a consistent domain signature, all images (both RGB and Thermal) are converted to **grayscale** and resized to a fixed resolution of **256x256 pixels**. This ensures the frequency analysis is based purely on intensity patterns, independent of color or original size.

2.  **For the Model Backbone:** Images are processed using standard vision transformer techniques. This includes resizing (e.g., to 800px on the longest side), converting to a PyTorch tensor, and normalizing using standard ImageNet statistics. This is handled automatically by the data loaders in `train_main.py`.

### CSV Annotation Format

The `train.csv` and `test.csv` files must contain the following columns. The column order does not matter, but the names must match exactly.

| Column Name | Description |
|---|---|
| `label_name` | The string name of the object class (e.g., "larch"). |
| `bbox_x` | The top-left x-coordinate of the bounding box. |
| `bbox_y` | The top-left y-coordinate of the bounding box. |
| `bbox_width` | The width of the bounding box. |
| `bbox_height` | The height of the bounding box. |
| `image_name` | The filename of the image in the corresponding folder. |
| `image_width` | The width of the image in pixels. |
| `image_height`| The height of the image in pixels. |

## Hyperparameter Tuning

The key hyperparameters `α`, `β`, and `γ` were determined through empirical evaluation as detailed in our paper.

*   **`α` (FDR Similarity Balance):** This parameter balances the real (magnitude) and imaginary (phase) components in the frequency-domain similarity calculation. We performed a sensitivity analysis by varying `α` from 0.0 to 1.0 (see Fig. 6 in the paper). The value of **`α = 0.6`** was chosen as it consistently yielded the highest mean and median AP scores across tasks.

*   **`β` (Routing Feature Update):** This parameter controls the update rate of an expert's routing feature during class-incremental learning. Instead of a fixed value, we use a **dynamic strategy**: `β = n / (n + 1)`, where `n` is the number of classes the expert has already learned. This makes the expert's domain signature more stable and less susceptible to change as it accumulates more knowledge.

*   **`γ` (SGVF Fusion Ratio):** This parameter balances knowledge retention from the "teacher" expert and knowledge acquisition from the "student" expert during fusion. Our experiments (see Fig. 9 in the paper) show a clear trade-off. The value of **`γ = 0.6`** was selected as it provides an optimal balance, enabling effective learning of the new class while minimizing forgetting of previous classes within the same domain.

## How to Use

The entire training and inference workflows are managed by simple shell scripts. You only need to edit the configuration variables at the top of these scripts.

### Training on a New Task
Configure and run `train/train_workflow.sh` to train the model on a new incremental task. The script will automatically update `weights/latest_model.pth` and `configs/experts.json`.

### Inference with Incre-MoE
Configure and run `inference/inference_workflow.sh` to run inference on a single image. The script will use FDR to select the best expert and perform sparse activation for efficient prediction.

## Project Structure

```
incre-moe/
├── bert/                      # Stores BERT weight files for the model's text encoder
├── datasets/                  # Holds the datasets for all tasks
├── error_analysis/            # Stores results from model error analysis
│   ├── a_0.6/                 # Experiment results under a specific hyperparameter
│   │   └── pt6/
│   │       ├── flir_dataset_car_test/
│   │       └── ... (misclassified samples for other test sets)
│   └── fdr_sample/            # Sample images for FDR analysis or visualization
│       ├── car/
│       └── ... (other categories)
├── eval/                      # Contains model evaluation scripts (e.g., calculating mAP)
├── groundingdino/             # Git submodule for the base GroundingDINO model
├── hyper_parameter_α_3D/      # Stores results from hyperparameter tuning experiments for α
│   ├── 0.0/
│   ├── 0.1/
│   └── ... (other α values)
├── hyper_parameter_γ/         # Stores results from tuning and comparative experiments for γ
├── inference/                 # Contains code and workflows for model inference
├── train/                     # Contains code and workflows for model training
└── weights/                   # Stores trained model weight files
```

*   **`bert/`**: Stores BERT weight files used by the model's text encoder.
*   **`datasets/`**: Holds all datasets required for the different learning tasks.
*   **`error_analysis/`**: Contains scripts and results for analyzing model failures, such as logs of misclassified samples and images used for FDR visualization.
*   **`eval/`**: Includes scripts for evaluating model performance, such as calculating mean Average Precision (mAP).
*   **`groundingdino/`**:The model.
*   **`hyper_parameter_α_3D/`**: Stores experimental results related to the tuning of the `α` hyperparameter.
*   **`hyper_parameter_γ/`**: Stores experimental results related to the tuning and comparison of the `γ` hyperparameter.
*   **`inference/`**: Contains all necessary code and workflows for running inference with a trained model.
*   **`train/`**: Contains all necessary code and workflows for training the model on new tasks.
*   **`weights/`**: Serves as the storage location for trained model weights (e.g., `latest_model.pth`).