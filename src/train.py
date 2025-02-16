# src/train.py

import argparse
import csv
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

from models.multi_task_model import MultiTaskModel
from evaluate import evaluate_model  # returns (class_acc, senti_acc)
from data_utils import split_data_3way, split_train_val, batch_iter, classification_data, sentiment_data

def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-task transformer model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    return parser.parse_args()

def train_multitask(args):
    # 1. Hyperparams
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr

    # 2. Split data into train/val
    #    This yields 80% training, 20% validation.
    # train_data_class, val_data_class = split_train_val(classification_data, 0.8)
    # train_data_senti, val_data_senti = split_train_val(sentiment_data, 0.8)
    # example: 70% train, 15% val, 15% test
    train_data_class, val_data_class, test_data_class = split_data_3way(classification_data, 0.7, 0.15)
    train_data_senti, val_data_senti, test_data_senti = split_data_3way(sentiment_data, 0.7, 0.15)
    # 3. Initialize model, tokenizer, optimizer
    model_name = "distilbert-base-uncased"
    num_classes_classification = 5
    num_classes_sentiment = 3

    model = MultiTaskModel(model_name, num_classes_classification, num_classes_sentiment)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    criterionA = nn.CrossEntropyLoss()  # classification
    criterionB = nn.CrossEntropyLoss()  # sentiment
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 4. Prepare CSV logging
    csv_file = open("training_log.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "AvgClassLoss", "AvgSentiLoss", "ClassAccuracy", "SentiAccuracy"])

    # 5. Training Loop
    for epoch in range(num_epochs):
        model.train()

        # Shuffle each epoch so we don't always see data in the same order
        random.shuffle(train_data_class)
        random.shuffle(train_data_senti)

        # We'll track classification and sentiment losses separately
        total_class_loss = 0.0
        total_senti_loss = 0.0
        class_batches = 0
        senti_batches = 0

        # ---- Classification Batches ----
        for batch in batch_iter(train_data_class, batch_size):
            sentences = [ex[0] for ex in batch]
            labelsA = [ex[1] for ex in batch]  # real classification labels

            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
            logitsA, logitsB = model(inputs["input_ids"], inputs["attention_mask"])

            labelA_tensor = torch.tensor(labelsA, dtype=torch.long)

            # Only compute classification loss here
            lossA = criterionA(logitsA, labelA_tensor)

            optimizer.zero_grad()
            lossA.backward()
            optimizer.step()

            total_class_loss += lossA.item()
            class_batches += 1

        # ---- Sentiment Batches ----
        for batch in batch_iter(train_data_senti, batch_size):
            sentences = [ex[0] for ex in batch]
            labelsB = [ex[1] for ex in batch]  # real sentiment labels

            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
            logitsA, logitsB = model(inputs["input_ids"], inputs["attention_mask"])

            labelB_tensor = torch.tensor(labelsB, dtype=torch.long)

            # Only compute sentiment loss here
            lossB = criterionB(logitsB, labelB_tensor)

            optimizer.zero_grad()
            lossB.backward()
            optimizer.step()

            total_senti_loss += lossB.item()
            senti_batches += 1

        # Compute average classification & sentiment loss
        avg_class_loss = total_class_loss / class_batches if class_batches > 0 else 0.0
        avg_senti_loss = total_senti_loss / senti_batches if senti_batches > 0 else 0.0

        # Evaluate on validation
        model.eval()
        class_acc, senti_acc = evaluate_model(model, tokenizer, val_data_class, val_data_senti)
        # after the training loop finishes
        

        # Print separate metrics
        print(f"Epoch {epoch+1}: "
              f"Classification Loss={avg_class_loss:.4f}, "
              f"Sentiment Loss={avg_senti_loss:.4f}, "
              f"Class Acc={class_acc:.2f}, "
              f"Senti Acc={senti_acc:.2f}")

        # Log to CSV
        csv_writer.writerow([epoch+1, avg_class_loss, avg_senti_loss, class_acc, senti_acc])

    # 6. Close CSV file
    csv_file.close()
    print("Training complete. Metrics logged to CSV.")
    print("Evaluating on test set...")
    test_class_acc, test_senti_acc = evaluate_model(model, tokenizer, test_data_class, test_data_senti)
    print(f"Final Test Performance: Class Acc={test_class_acc:.2f}, Senti Acc={test_senti_acc:.2f}")


if __name__ == "__main__":
    args = parse_args()
    train_multitask(args)
