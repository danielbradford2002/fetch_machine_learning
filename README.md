# Multi-Task Sentence Transformer for ML Apprentice Role

## Overview

This project implements a multi-task learning framework using a sentence transformer to address two NLP tasks:
- **Task A: Sentence Classification** – Classify sentences into five predefined categories: Sports, Business, Entertainment, Technology, and Politics.
- **Task B: Sentiment Analysis** – Determine sentiment with three categories: Negative, Neutral, and Positive.

The model uses **DistilBERT** as its backbone to produce fixed-length sentence embeddings via mean pooling. Two task-specific linear heads then map these embeddings to the respective outputs.

The project is organized into several modules:
- `src/train.py`: Contains the training loop, data splitting (train/validation/test), and CSV logging.
- `src/evaluate.py`: Provides functions for evaluating the model on validation and test data.
- `src/data_utils.py`: Contains helper functions for splitting data and batching, along with synthetic datasets.
- `models/multi_task_model.py`: Defines the multi-task model architecture.

In addition, the project is containerized with Docker to ensure reproducibility across environments, and the environment is specified via `requirements.txt`.

---

## Task 1: Sentence Transformer Implementation

**Objective:**  
Implement a sentence transformer that encodes variable-length sentences into fixed-length embeddings.

**Approach:**  
- **Backbone:** The project uses a pretrained DistilBERT model due to its balance of speed and performance.
- **Pooling Strategy:** Mean pooling is applied over the transformer’s token embeddings to obtain a fixed-size representation.
- **Testing:** Sample sentences are processed through the model to validate the embedding extraction.

*Key Decisions:*  
- Mean pooling was chosen for simplicity and effectiveness.
- DistilBERT provides a compact yet powerful representation for English text.

---

## Task 2: Multi-Task Learning Expansion

**Objective:**  
Expand the transformer model to handle both sentence classification and sentiment analysis.

**Approach:**  
- **Shared Backbone:** The DistilBERT encoder is shared across tasks.
- **Task-Specific Heads:**  
  - **Classification Head:** A linear layer producing outputs for 5 classes.  
  - **Sentiment Head:** A linear layer producing outputs for 3 sentiment categories.

*Key Decisions:*  
- Sharing the transformer backbone allows the model to leverage common language representations while the separate heads fine-tune task-specific mappings.

---

## Task 3: Training Considerations and Transfer Learning

**Objective:**  
Discuss different training scenarios and the transfer learning approach.

**Scenarios Considered:**

1. **Entire Network Frozen:**  
   - *Implication:* The model relies entirely on pretrained weights, and only a forward pass is performed.  
   - *Pros:* Faster training and less risk of overfitting with very limited data.  
   - *Cons:* No adaptation to the specific tasks, possibly leading to suboptimal performance if the target domain is different.

2. **Transformer Backbone Frozen:**  
   - *Implication:* The DistilBERT layers are fixed while the task-specific heads are trained.  
   - *Pros:* Preserves general language understanding and speeds up training; effective when the tasks are similar to the pretraining data.
   - *Cons:* Limited domain adaptation if the tasks require more specialized knowledge.

3. **One Task-Specific Head Frozen:**  
   - *Implication:* One head (for example, classification) is left unchanged while the other head and the backbone are fine-tuned.
   - *Pros:* Useful if one task already has strong performance and you want to preserve it while adapting the other.
   - *Cons:* May prevent the frozen head from adjusting to subtle changes in the shared representations.

**Transfer Learning Approach:**

- **Pretrained Model Choice:** DistilBERT was selected for its efficiency and strong performance.
- **Layer Freezing Strategy:**  
  - In scenarios with limited data, freezing the backbone and updating only the heads can help avoid overfitting.
  - For tasks that differ significantly from the pretraining domain, selectively unfreezing some top layers may yield better performance.
  
*Key Decisions:*  
- The balance between freezing and fine-tuning depends on available data and task similarity.
- Experimentation with these strategies (e.g., freezing the entire backbone vs. partial unfreezing) can help determine the optimal approach.

---

## Task 4: Training Loop Implementation

**Objective:**  
Develop a training loop that handles multi-task learning, including data splitting and metric tracking.

**Approach:**  
- **Data Splitting:** The synthetic data is split into 70% training, 15% validation, and 15% test sets.
- **Batch Processing:**  
  - The training loop processes classification and sentiment data separately so that each head is updated with its correct labels.
- **Metrics & Logging:**  
  - After each epoch, the average loss for each task is computed.
  - The model is evaluated on the validation set, and metrics (accuracy and loss) are logged to a CSV file.
  - A final evaluation is performed on the test set to gauge generalization.

*Key Decisions:*  
- Processing tasks separately in batches prevents the model from learning with dummy labels.
- Logging metrics enables tracking progress and comparing different training strategies.

---

## Performance Results and Future Improvements

**Current Results:**  
- **Training and validation outputs** show a gradual decrease in loss and improvement in accuracy for both classification and sentiment tasks.
- **Final test performance:** Classification Accuracy = 0.89, Sentiment Accuracy = 0.93

**Potential Improvements:**  
- **Data Quantity & Quality:** Increasing the amount of training data or using real-world datasets could further improve model performance.
- **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and number of epochs.
- **Layer Freezing Experiments:** Test scenarios where different parts of the model are frozen/unfrozen to optimize performance.
- **Pooling Strategy:** Explore alternative pooling methods (e.g., attention-based pooling) to potentially enhance the quality of the sentence embeddings.
- **Fine-Tuning:** Consider fine-tuning the backbone on a domain-specific corpus if the tasks are in a specialized domain.

---

## Docker Containerization & Environment Setup

### Docker Containerization

**Purpose:**  
- Ensures a consistent and reproducible runtime environment.
- Simplifies deployment across different systems.

**How to Build and Run:**

1. **Build the Docker Image:**
bash:
   docker build -t multi-task-nlp .
Run the Docker Container:
bash
docker run --rm multi-task-nlp
Dockerfile Details:

Base Image: Uses python:3.11-slim for a lightweight, up-to-date environment.
WORKDIR: Sets the working directory to /app and sets PYTHONPATH to /app so that the code in models/ and src/ is discoverable.
Dependency Installation: Copies requirements.txt and installs dependencies.
Code Copy: Copies the entire project into the container.
CMD: Defaults to running the training script.
Environment Setup & requirements.txt


**Creating requirements.txt:**

**Activate your virtual environment:**
bash:
    source venv/bin/activate

**Install Dependencies:**
Ensure all required packages are installed (e.g., torch, transformers, etc.).
Generate the File:
bash:
    pip freeze > requirements.txt

This creates a file with pinned versions of all packages.
**Review and Clean:**
Optionally, remove unnecessary packages to keep the file focused on your project.
Example requirements.txt:
certifi==2025.1.31
charset-normalizer==3.4.1
filelock==3.17.0
fsspec==2025.2.0
huggingface-hub==0.28.1
idna==3.10
Jinja2==3.1.5
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==2.8.8
numpy==2.2.3
packaging==24.2
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
safetensors==0.5.2
sympy==1.13.1
tokenizers==0.21.0
torch==2.6.0
tqdm==4.67.1
transformers==4.48.3
typing_extensions==4.12.2
urllib3==2.3.0

## Conclusion

**This project demonstrates:** 

Task 1: Implementation of a sentence transformer using DistilBERT and mean pooling.
Task 2: Expansion to multi-task learning with separate heads for classification and sentiment analysis.
Task 3: Detailed training considerations, including freezing strategies and transfer learning rationale.
Task 4: A robust training loop with proper train/validation/test splits, metric logging, and evaluation.
Future improvements include enhancing data quality and quantity, hyperparameter tuning, further experiments with layer freezing, and exploring alternative pooling strategies.

The project is containerized with Docker for consistency, and the environment is managed via requirements.txt for reproducibility.