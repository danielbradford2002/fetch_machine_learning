import torch
from transformers import AutoTokenizer
from models.multi_task_model import MultiTaskModel

def evaluate_model(model, tokenizer, eval_data_class, eval_data_senti):
    """
    Evaluate both the classification and sentiment tasks.
    
    This function computes the accuracy for each task and prints the results.
    
    Args:
        model: The trained MultiTaskModel.
        tokenizer: The tokenizer used for encoding input sentences.
        eval_data_class: List of (text, label) for classification.
        eval_data_senti: List of (text, label) for sentiment analysis.
    
    Returns:
        A tuple (accuracy_class, accuracy_senti).
    """
    model.eval()
    correct_class, total_class = 0, 0
    correct_senti, total_senti = 0, 0

    # Disable gradients for evaluation
    with torch.no_grad():
        # Evaluate classification task
        for text, labelA in eval_data_class:
            inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
            logitsA, _ = model(inputs["input_ids"], inputs["attention_mask"])
            predA = logitsA.argmax(dim=1).item()
            if predA == labelA:
                correct_class += 1
            total_class += 1

        # Evaluate sentiment task
        for text, labelB in eval_data_senti:
            inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
            _, logitsB = model(inputs["input_ids"], inputs["attention_mask"])
            predB = logitsB.argmax(dim=1).item()
            if predB == labelB:
                correct_senti += 1
            total_senti += 1

    accuracy_class = correct_class / total_class if total_class else 0
    accuracy_senti = correct_senti / total_senti if total_senti else 0

    print(f"Classification Accuracy: {accuracy_class:.2f}")
    print(f"Sentiment Accuracy: {accuracy_senti:.2f}")
    return accuracy_class, accuracy_senti

def main():
    # Example usage: initialize model and tokenizer, then run evaluation on sample data.
    model_name = "distilbert-base-uncased"
    model = MultiTaskModel(model_name, 5, 2)  # Note: adjust number of classes if needed.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_data_class = [
        ("The senator introduced a bill targeting tax reform.", 4),  # politics
        ("The soccer team won the championship.", 0),               # sports
    ]
    eval_data_senti = [
        ("I hated the service at that restaurant; it was awful.", 1),  # negative (assuming mapping: 0=negative, 1=neutral, 2=positive)
        ("I absolutely love this movie, it's fantastic!", 0),         # positive (if reversed, adjust accordingly)
    ]

    evaluate_model(model, tokenizer, eval_data_class, eval_data_senti)

if __name__ == "__main__":
    main()
