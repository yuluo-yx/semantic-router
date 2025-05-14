import os
import json
import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Define a custom cross entropy loss compatible with sentence-transformers
class ClassificationLoss(torch.nn.Module):
    def __init__(self, model):
        super(ClassificationLoss, self).__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, sentence_features, labels):
        # sentence_features is a list of dictionaries for each text
        # We get embeddings for the input texts
        embeddings = self.model(sentence_features[0])['sentence_embedding']
        
        # For each batch element after the first one
        for i in range(1, len(sentence_features)):
            emb = self.model(sentence_features[i])['sentence_embedding']
            embeddings = torch.cat((embeddings, emb.unsqueeze(0)))
        
        # Convert label indices to a tensor
        label_tensor = torch.tensor(labels, dtype=torch.long, device=embeddings.device)
        
        # Calculate and return the loss
        return self.loss_fn(embeddings, label_tensor)
        
    def __call__(self, sentence_features, labels):
        return self.forward(sentence_features, labels)

# Load the MMLU-Pro dataset
dataset = load_dataset("TIGER-Lab/MMLU-Pro")
print(f"Dataset splits: {dataset.keys()}")

# Extract questions and categories from the test split
questions = dataset["test"]["question"]
categories = dataset["test"]["category"]

# Get unique categories and create a mapping
unique_categories = list(set(categories))
category_to_idx = {category: idx for idx, category in enumerate(unique_categories)}
idx_to_category = {idx: category for category, idx in category_to_idx.items()}

print(f"Found {len(unique_categories)} unique categories: {unique_categories}")

# Convert categories to indices
category_indices = [category_to_idx[category] for category in categories]

# Split the data into train and validation sets
train_questions, val_questions, train_categories, val_categories = train_test_split(
    questions, category_indices, test_size=0.2, stratify=category_indices, random_state=42
)

# Create a custom model with classification head
# The base model can be chosen as per requirements, e.g., 'sentence-transformers/all-MiniLM-L6-v2' or 'sentence-transformers/all-MiniLM-L12-v2'
word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L12-v2')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# The dense layer will output logits for each category
dense_model = models.Dense(
    in_features=pooling_model.get_sentence_embedding_dimension(),
    out_features=len(unique_categories),
    activation_function=torch.nn.Identity() # Outputs raw logits
)

# Create the full model pipeline
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

# Prepare training data
train_samples = [(question, category) for question, category in zip(train_questions, train_categories)]

# Define the loss function (custom cross entropy for classification)
train_loss = ClassificationLoss(model)

# Configure the training
num_epochs = 8
batch_size = 16
train_examples = []
for question, category_idx in train_samples:
    train_examples.append(InputExample(texts=[question], label=category_idx))

warmup_steps = int(len(train_examples) * num_epochs * 0.1 / batch_size)

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

output_model_path = "category_classifier_linear_model"
os.makedirs(output_model_path, exist_ok=True)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluator=None,
    output_path=output_model_path,
    show_progress_bar=True
)

# Save the category mapping
category_mapping_path = os.path.join(output_model_path, "category_mapping.json")
with open(category_mapping_path, "w") as f:
    json.dump({
        "category_to_idx": category_to_idx,
        "idx_to_category": {str(k): v for k, v in idx_to_category.items()} # JSON keys must be strings
    }, f)

# Function to predict category using the linear classification model
def predict_category_linear(model, question, idx_to_category_map):
    logits_np = model.encode(question, show_progress_bar=False)
    logits_tensor = torch.tensor(logits_np)

    # Apply softmax to get probabilities. For a single sample, logits_tensor will be 1D.
    probabilities = torch.softmax(logits_tensor, dim=0) 

    # Get the predicted class index and confidence
    confidence_tensor, predicted_idx_tensor = torch.max(probabilities, dim=0)
    predicted_idx = predicted_idx_tensor.item()
    confidence = confidence_tensor.item()

    predicted_category_name = idx_to_category_map.get(predicted_idx, "Unknown Category")
    
    return predicted_category_name, confidence

# Evaluate on validation set using the linear classification model
def evaluate_classifier_linear(model, questions_list, true_category_indices_list, idx_to_category_map):
    correct = 0
    total = len(questions_list)
    
    if total == 0:
        return 0.0

    for question, true_category_idx in zip(questions_list, true_category_indices_list):
        predicted_category_name, _ = predict_category_linear(model, question, idx_to_category_map)
        true_category_name = idx_to_category_map.get(true_category_idx) # true_category_idx is already an int
        
        if true_category_name == predicted_category_name:
            correct += 1
    
    return correct / total

print("\nEvaluating on validation set...")
val_accuracy = evaluate_classifier_linear(model, val_questions, val_categories, idx_to_category)
print(f"Validation accuracy: {val_accuracy:.4f}")


model.save(output_model_path)

# Test the model with examples
print("\nTesting with examples:")
queries = [
    "What is the derivative of x^2?",
    "Explain the concept of supply and demand in economics.",
    "How does DNA replication work in eukaryotic cells?",
    "What is the difference between a civil law and common law system?",
    "Explain how transistors work in computer processors.",
    "Why do stars twinkle?",
    "How do I create a balanced portfolio for retirement?",
    "What causes mental illnesses?",
    "How do computer algorithms work?",
    "Explain the historical significance of the Roman Empire.",
]

for i in range(len(queries)):
    question = queries[i]
    predicted_category, confidence = predict_category_linear(model, question, idx_to_category)
    print(f"Question: {question[:100]}...")
    print(f"Predicted category: {predicted_category}, Confidence: {confidence:.4f}")
    print("---")

print(f"Fine-tuned model and category mapping saved to: {output_model_path}") 