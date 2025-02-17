import argparse
import csv
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

# Import our multi-task model and helper functions for data splitting and batching
from models.multi_task_model import MultiTaskModel
from evaluate import evaluate_model  # This returns (classification_accuracy, sentiment_accuracy)
from data_utils import split_data_3way, batch_iter, classification_data, sentiment_data

def parse_args():
    """
    Parse command-line arguments for training hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Train a multi-task transformer model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    return parser.parse_args()

def train_multitask(args):
    """
    Train the multi-task model using separate batches for classification and sentiment tasks.
    
    This function:
      - Splits the data into train, validation, and test sets.
      - Initializes the model, tokenizer, and optimizer.
      - Trains the model for a specified number of epochs.
      - Logs metrics (losses and accuracy) to a CSV file.
      - Evaluates the final model on a test set.
    """
    # Retrieve hyperparameters from command-line arguments
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr

    # Split our synthetic data into 70% training, 15% validation, and 15% test sets
    train_data_class, val_data_class, test_data_class = split_data_3way(classification_data, 0.7, 0.15)
    train_data_senti, val_data_senti, test_data_senti = split_data_3way(sentiment_data, 0.7, 0.15)

    # Initialize the model using DistilBERT as the backbone.
    # We have 5 classes for classification and 3 for sentiment.
    model_name = "distilbert-base-uncased"
    num_classes_classification = 5
    num_classes_sentiment = 3

    model = MultiTaskModel(model_name, num_classes_classification, num_classes_sentiment)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define loss functions for each task.
    # CrossEntropyLoss expects integer class labels.
    criterionA = nn.CrossEntropyLoss()  # for classification task
    criterionB = nn.CrossEntropyLoss()  # for sentiment task
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Prepare CSV logging to track our metrics across epochs.
    csv_file = open("training_log.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "AvgClassLoss", "AvgSentiLoss", "ClassAccuracy", "SentiAccuracy"])

    # Begin the training loop.
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        # Shuffle training data to prevent the model from learning order-specific patterns.
        random.shuffle(train_data_class)
        random.shuffle(train_data_senti)

        total_class_loss = 0.0
        total_senti_loss = 0.0
        class_batches = 0
        senti_batches = 0

        # Process classification data in batches.
        # This loop iterates over batches of the training data for the classification task.
        for batch in batch_iter(train_data_class, batch_size):
            # Extract the sentences (text data) from each example in the batch.
            sentences = [ex[0] for ex in batch]
            # Extract the corresponding true labels for classification from the batch.
            labelsA = [ex[1] for ex in batch]  # True classification labels

            # Use the tokenizer to convert the list of sentences into tensors.
            # 'padding=True' ensures that all sentences in the batch have the same length.
            # 'truncation=True' truncates sentences that exceed the model's maximum input size.
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
            
            # Perform a forward pass through the model.
            # We only need the output from the classification head, so we ignore the sentiment output.
            logitsA, _ = model(inputs["input_ids"], inputs["attention_mask"])  # Only classification head output is used

            # Convert the list of true labels to a PyTorch tensor with dtype long, which is required for loss computation.
            labelA_tensor = torch.tensor(labelsA, dtype=torch.long)
            # Calculate the loss for the classification task by comparing the model's logits with the true labels.
            lossA = criterionA(logitsA, labelA_tensor)

            # Zero out gradients from previous batches to ensure they don't accumulate.
            optimizer.zero_grad()
            # Backpropagate the classification loss to compute gradients.
            lossA.backward()
            # Update model parameters using the optimizer (AdamW).
            optimizer.step()

            # Accumulate the classification loss for averaging later.
            total_class_loss += lossA.item()
            # Keep count of the number of classification batches processed.
            class_batches += 1

        # Process sentiment data in batches (minimal comments)
        for batch in batch_iter(train_data_senti, batch_size):
            sentences = [ex[0] for ex in batch]
            labelsB = [ex[1] for ex in batch]

            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
            _, logitsB = model(inputs["input_ids"], inputs["attention_mask"])
            labelB_tensor = torch.tensor(labelsB, dtype=torch.long)
            lossB = criterionB(logitsB, labelB_tensor)

            optimizer.zero_grad()
            lossB.backward()
            optimizer.step()

            total_senti_loss += lossB.item()
            senti_batches += 1

        # Compute average losses for each task
        avg_class_loss = total_class_loss / class_batches if class_batches else 0.0
        avg_senti_loss = total_senti_loss / senti_batches if senti_batches else 0.0

        # Evaluate on the validation set and obtain accuracy for each task
        model.eval()
        class_acc, senti_acc = evaluate_model(model, tokenizer, val_data_class, val_data_senti)

        # Print metrics for the epoch
        print(f"Epoch {epoch+1}: Classification Loss={avg_class_loss:.4f}, "
              f"Sentiment Loss={avg_senti_loss:.4f}, "
              f"Class Acc={class_acc:.2f}, Senti Acc={senti_acc:.2f}")

        # Log metrics to CSV
        csv_writer.writerow([epoch+1, avg_class_loss, avg_senti_loss, class_acc, senti_acc])

    csv_file.close()
    print("Training complete. Metrics logged to CSV.")

    # After training, evaluate on the test set for final performance
    print("Evaluating on test set...")
    test_class_acc, test_senti_acc = evaluate_model(model, tokenizer, test_data_class, test_data_senti)
    print(f"Final Test Performance: Class Acc={test_class_acc:.2f}, Senti Acc={test_senti_acc:.2f}")

if __name__ == "__main__":
    args = parse_args()
    train_multitask(args)
