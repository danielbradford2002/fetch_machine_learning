import torch
from transformers import AutoTokenizer
from models.multi_task_model import MultiTaskModel

def evaluate_model(model, tokenizer, eval_data_class, eval_data_senti):
    """
    Evaluate classification and sentiment tasks separately and print accuracy.
    
    :param model: An instance of MultiTaskModel (already loaded)
    :param tokenizer: The tokenizer matching the model
    :param eval_data_class: List of (text, label) for classification
    :param eval_data_senti: List of (text, label) for sentiment
    """
    model.eval()
    correct_class, total_class = 0, 0
    correct_senti, total_senti = 0, 0

    # Turn off gradient calculations for evaluation
    with torch.no_grad():
        # Evaluate classification
        for text, labelA in eval_data_class:
            inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
            logitsA, logitsB = model(inputs["input_ids"], inputs["attention_mask"])
            # Argmax for classification
            predA = logitsA.argmax(dim=1).item()
            if predA == labelA:
                correct_class += 1
            total_class += 1

        # Evaluate sentiment
        for text, labelB in eval_data_senti:
            inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
            logitsA, logitsB = model(inputs["input_ids"], inputs["attention_mask"])
            # Argmax for sentiment
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
    # 1. (Optionally) load a saved model checkpoint or re-initialize the model
    # Example: If you saved your trained model's weights to "model_weights.pt", you can do:
    # model = MultiTaskModel("distilbert-base-uncased", 5, 2)
    # model.load_state_dict(torch.load("model_weights.pt"))

    model_name = "distilbert-base-uncased"
    model = MultiTaskModel(model_name, 5, 2)  # 5 classes for classification, 2 for sentiment
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Example eval data (could also be loaded from a file)
    eval_data_class = [
        ("The senator introduced a bill targeting tax reform.", 4),   # politics
        ("The soccer team won the championship.", 0),                # sports
    ]
    eval_data_senti = [
        ("I hated the service at that restaurant; it was awful.", 1), # negative
        ("I absolutely love this movie, it's fantastic!", 0),         # positive
    ]

    # 3. Call the evaluation function
    evaluate_model(model, tokenizer, eval_data_class, eval_data_senti)

if __name__ == "__main__":
    main()
