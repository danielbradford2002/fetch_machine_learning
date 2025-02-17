# Multi-Task Sentence Transformer for ML Apprentice Role

## Overview

Hi, I’m Daniel and this project is my implementation of a multi-task learning framework using a sentence transformer for the Fetch technical take home challenge. I built it to tackle two natural language tasks:
- **Task A: Sentence Classification** – where I classify sentences into five categories (Sports, Business, Entertainment, Technology, Politics).
- **Task B: Sentiment Analysis** – where I determine if a sentence has Negative, Neutral, or Positive sentiment.

I use **DistilBERT** as the backbone to generate fixed-length sentence embeddings by applying mean pooling on the token representations. Then, two simple linear layers (one for each task) convert these embeddings into predictions.

The project is split into several modules:
- `src/train.py`: Contains the training loop, data splitting (train/validation/test), and CSV logging.
- `src/evaluate.py`: Has functions to evaluate the model on validation and test data.
- `src/data_utils.py`: Provides helper functions for splitting data and batching, along with my synthetic datasets.
- `models/multi_task_model.py`: Defines the overall multi-task model architecture.

I’ve also containerized the project with Docker to ensure it runs smoothly on any system, and I’ve specified all dependencies in a `requirements.txt` file.

---

## Task 1: Sentence Transformer Implementation

**Objective:**  
I needed to implement a sentence transformer that takes variable-length sentences and encodes them into fixed-length vectors.

**What I Did:**  
- **Backbone:** I chose DistilBERT for its balance between performance and efficiency.
- **Pooling Strategy:** I used mean pooling over the token embeddings from the transformer to get a fixed-size vector.
- **Testing:** I ran a few sample sentences through the model to ensure the embeddings looked reasonable.

**My Thoughts:**  
I went with mean pooling because it’s simple and effective, and DistilBERT was a natural choice due to its compact size while still providing strong performance on English text.

---

## Task 2: Multi-Task Learning Expansion

**Objective:**  
I expanded the transformer model to handle both sentence classification and sentiment analysis simultaneously.

**What I Did:**  
- I kept the DistilBERT encoder as a shared backbone.
- Then I added two separate linear heads:
  - One head for the 5-class sentence classification task.
  - Another head for the 3-class sentiment analysis task.

**My Thoughts:**  
By sharing the backbone, I leverage the same underlying language understanding for both tasks, while the separate heads allow each task to fine-tune its own mapping. This setup strikes a balance between efficiency and task specialization.

---

## Task 3: Training Considerations and Transfer Learning

**Objective:**  
This part was all about exploring different training strategies and thinking about transfer learning.

**Scenarios I Considered:**

1. **Entire Network Frozen:**  
   - *What it Means:* No layers are updated; the model uses only the pretrained weights.
   - *Pros:* Training is very fast and there’s minimal risk of overfitting when data is scarce.
   - *Cons:* The model can’t adapt to my specific tasks, which might limit performance if my data is quite different from what DistilBERT was trained on.

2. **Transformer Backbone Frozen:**  
   - *What it Means:* The DistilBERT layers remain fixed while only the task-specific heads are updated.
   - *Pros:* This approach is common when you have limited data; it preserves the general language knowledge while letting the heads learn the task-specific mappings.
   - *Cons:* It might not capture domain-specific nuances if my tasks need more specialized understanding.

3. **One Task-Specific Head Frozen:**  
   - *What it Means:* One of the task heads (say, classification) is kept unchanged while the other head and the backbone are fine-tuned.
   - *Pros:* This can be useful if one task is already performing well and I want to preserve its accuracy while adapting the other.
   - *Cons:* It might limit the model’s ability to adjust to new patterns if the frozen head really needs to learn more.

**Transfer Learning Approach:**

- I chose **DistilBERT** because it’s fast and effective.
- For freezing, the idea is to freeze layers when you have very limited data or when your tasks are similar to the pretrained domain. On the other hand, if my tasks are very different or I have enough data, I might unfreeze more layers (or even the whole model) to allow deeper adaptation.

**My Thoughts:**  
I experimented with different freezing strategies in my head (and I discuss these more in my code comments). Finding the right balance between preserving pretrained knowledge and adapting to new tasks is key, and it depends a lot on the data and the specific tasks.

---

## Task 4: Training Loop Implementation

**Objective:**  
I needed to build a training loop that supports multi-task learning with proper data splitting and metric logging.

**What I Did:**  
- **Data Splitting:** I split my synthetic data into 70% training, 15% validation, and 15% test sets.
- **Batch Processing:** I process classification and sentiment data in separate batches so that each head is updated using its own correct labels.
- **Metrics & Logging:** After each epoch, I compute the average loss for each task and evaluate the model on the validation set. These metrics are logged to a CSV file, and a final evaluation on the test set gives me an unbiased measure of performance.

**My Thoughts:**  
Separating the tasks during training ensures that each head gets accurate signals, and logging the metrics helps me track the model’s progress over time.

---

## Performance Results and Future Improvements

**Current Results:**  
- My training and validation outputs show that loss decreases steadily, with final test performance reaching around:
  - **Classification Accuracy:** 89%
  - **Sentiment Accuracy:** 93%

**Areas for Improvement:**  
- **Data Quality and Quantity:** Using larger, real-world datasets could further improve performance.
- **Hyperparameter Tuning:** There’s room to experiment with different learning rates, batch sizes, and numbers of epochs.
- **Layer Freezing Strategies:** I plan to explore more nuanced freezing/unfreezing strategies to see if further gains can be made.
- **Pooling Methods:** Alternative pooling strategies (like attention-based pooling) might enhance the quality of the embeddings.
- **Fine-Tuning:** More extensive fine-tuning on domain-specific data could yield additional improvements.

---

## Docker Containerization & Environment Setup

### Docker Containerization

**Purpose:**  
Using Docker makes my project environment consistent and reproducible across different systems.

**How to Build and Run:**

1. **Build the Docker Image:**
   bash:docker build -t multi-task-nlp .

Run the Docker Container:
bash: docker run --rm multi-task-nlp
Dockerfile Details:

**Dockerfile Highlights:**

I use python:3.11-slim as the base image for a lightweight, modern environment.
The working directory is set to /app and PYTHONPATH is configured so that my models/ and src/ directories are correctly discovered.
Dependencies from requirements.txt are installed, and the entire project is copied into the container.
The default command runs my training script.
Environment Setup & requirements.txt
Purpose:
The requirements.txt file allows others to recreate the exact Python environment for this project.

**How I Created It:**

I activated my virtual environment:
bash: source venv/bin/activate
I installed all the necessary libraries (like torch, transformers, etc.).
I generated the file with:
bash: pip freeze > requirements.txt
This file lists the exact package versions used, ensuring reproducibility.

## Conclusion
This project showcases my implementation of a multi-task sentence transformer using DistilBERT. I addressed the following:

Task 1: Building a sentence transformer with mean pooling.

Task 2: Expanding the model to handle both sentence classification and sentiment analysis.

Task 3: Exploring training considerations such as various freezing strategies and transfer learning approaches.

Task 4: Implementing a robust training loop with a proper train/validation/test split and logging metrics.

While the current results are promising (with test accuracies of 89% for classification and 93% for sentiment), there’s still room for improvement, especially in terms of data quality, hyperparameter tuning, and exploring alternative pooling and freezing strategies.

The project is fully containerized with Docker for consistency, and the environment is managed via requirements.txt for easy reproducibility.

Feel free to check out the code, and thanks so much for taking the time to review my work. I cannot wait to hear back from the Fetch team!