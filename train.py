# Step 4
# I should train a model to actually predict entities from pages
import pandas as pd
import torch
from sklearn.model_selection import train_test_split  # uncomment if you want sklearn split
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
import ast
import datasets
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer
# I was used a bit to the PyTorch Lightning Trainer, but found out about this one too
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_dataset_file', type=str, default='labeled_dataset.csv')
parser.add_argument('--path_to_save_model', type=str, default='furniture_ner_model', required=False)
parser.add_argument('--test_data_size', type=float, default=0.1, required=False)
parser.add_argument('--nb_of_epochs', type=int, default=25, required=False)
parser.add_argument('--batch_size', type=int, default=16, required=False)


def train():
    args = parser.parse_args()

    # Read the labeled dataset
    # df = pd.read_csv('labeled_dataset.csv')
    df = pd.read_csv(args.path_to_dataset_file)
    logging.info("Loaded the csv dataset file!")

    # I didn't like this because you may drop weird indices
    #  and this failed me on other methods
    # train_df = df.sample(frac=0.8,random_state=42)
    # val_df = df.drop(train_df.index)

    # Or sklearn can be used also:
    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=args.test_data_size, random_state=42)

    dataset_train = datasets.Dataset.from_pandas(train_df, preserve_index=False)
    dataset_val = datasets.Dataset.from_pandas(val_df, preserve_index=False)

    id2label = {0: 'O', 1: 'B-PRODUCT', 2: 'I-PRODUCT'}
    label2id = {'O': 0, 'B-PRODUCT': 1, 'I-PRODUCT': 2}

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    model = AutoModelForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=3, id2label=id2label, label2id=label2id)

    # Only get the max length of the model from tokens, no point going further
    train_tokens = [text[:tokenizer.model_max_length].lower().split() for text in train_df['text']]
    val_tokens = [text[:tokenizer.model_max_length].lower().split() for text in val_df['text']]

    # Create some dictionaries for faster (and easier retrieval)
    examples_train = {}
    examples_val = {}

    examples_train['tokens'] = train_tokens
    examples_train['ner_tags'] = [ast.literal_eval(str_list) for str_list in train_df['label'].tolist()]

    examples_val['tokens'] = val_tokens
    # Convert from strings to lists if ints
    examples_val['ner_tags'] = [ast.literal_eval(str_list) for str_list in val_df['label'].tolist()]

    tokenized_train_inputs = {}
    tokenized_val_inputs = {}

    tokenized_train_inputs['tokens'] = tokenizer(examples_train['tokens'], is_split_into_words=True)
    tokenized_val_inputs['tokens'] = tokenizer(examples_val['tokens'], is_split_into_words=True)

    tokens_train = tokenizer.convert_ids_to_tokens(tokenized_train_inputs['tokens']['input_ids'][0])
    tokens_val = tokenizer.convert_ids_to_tokens(tokenized_val_inputs['tokens']['input_ids'][0])

    # Great resource: https://huggingface.co/docs/transformers/tasks/token_classification
    # It provided a method to align label -100 the CLS and SEP tokens +
    #  only label the first token of given word!
    # + https://medium.com/@andrewmarmon/fine-tuned-named-entity-recognition-with-hugging-face-bert-d51d4cb3d7b5
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

        labels = []

        # Loop through labels
        for pos, label in enumerate(examples["ner_tags"]):

            word_ids = tokenized_inputs.word_ids(batch_index=pos)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:  # if index in words not found -> special token
                    label_ids.append(-100)  # Set the special tokens to -100
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                    # We changed to the next word id -> use the label for the first token
                else:
                    label_ids.append(-100)  # assign -100 to other tokens from SAME word
                previous_word_idx = word_idx  # We are still at the same word
            labels.append(label_ids)

        return labels

    tokenized_train_inputs['labels'] = tokenize_and_align_labels(examples_train)
    tokenized_val_inputs['labels'] = tokenize_and_align_labels(examples_val)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    seqeval = evaluate.load("seqeval")

    label_list = ['O', 'B-PRODUCT', 'I-PRODUCT']

    # https://medium.com/@andrewmarmon/fine-tuned-named-entity-recognition-with-hugging-face-bert-d51d4cb3d7b5
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Looks like I also need a custom dataset
    class CustomNERDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.tokens = dataset['tokens']
            self.labels = dataset['labels']

        def __getitem__(self, index):
            # Create a new item based on the tokens at that index
            elem = {k: torch.tensor(v[index]) for k, v in self.tokens.items()}

            # Add the labels also from that same index
            # logging.info(self.labels[index])
            elem['labels'] = torch.tensor(self.labels[index])
            return elem

        def __len__(self):
            return len(self.labels)

    # Create the datasets
    train_set = CustomNERDataset(tokenized_train_inputs)
    val_set = CustomNERDataset(tokenized_val_inputs)
    logging.info("Got the datasets!")

    training_args = TrainingArguments(
        output_dir=args.path_to_save_model,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.nb_of_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting training...")
    trainer.train()

if __name__ == '__main__':
    train()


# I also tried some other things with other datasets created, the result wasn't great,
#  the performance was even worse.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the dataset from CSV
df = pd.read_csv("dataset.csv")
texts = df["text"].tolist()

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)

# Use the IOB2 labeling scheme for product names
labels = [["O"] * len(tokenizer.tokenize(text)) for text in texts]

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tokenizer.convert_tokens_to_ids(["O", "B-PRODUCT", "I-PRODUCT"])))

# Tokenize and convert labels to IDs
train_encodings = tokenizer(train_texts, truncation=True, padding=True, is_split_into_words=True, return_offsets_mapping=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, is_split_into_words=True, return_offsets_mapping=True)

train_labels_encoded = []
val_labels_encoded = []

# Convert from string labels to int
# I simply wanted to check this code suggested to me by ChatGPT.
# Of course it didn't worked, and also tried to debug it.
# It apparently uses the offset mappings from the tokenizer to create the new labels (str -> int)
for i, offset_mapping in enumerate(train_encodings["offset_mapping"]):
    label_ids = [0]  # 0 corresponds to "O"
    for j, (start, end) in enumerate(offset_mapping):
        if j > 0 and start != offset_mapping[j-1][1]:
            label_ids.append(0)  # 0 corresponds to "O"
        label_ids[-1] = 1 if train_labels[i][j].startswith("B-PRODUCT") else label_ids[-1]
    train_labels_encoded.append(label_ids)

for i, offset_mapping in enumerate(val_encodings["offset_mapping"]):
    label_ids = [0]  # 0 corresponds to "O"
    for j, (start, end) in enumerate(offset_mapping):
        if j > 0 and start != offset_mapping[j-1][1]:
            label_ids.append(0)  # 0 corresponds to "O"
        label_ids[-1] = 1 if val_labels[i][j].startswith("B-PRODUCT") else label_ids[-1]
    val_labels_encoded.append(label_ids)

# Create custom dataset
# This is basically what the HuggingFace documentation suggests to do
class ProductNameDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ProductNameDataset(train_encodings, train_labels_encoded)
val_dataset = ProductNameDataset(val_encodings, val_labels_encoded)

# Set up training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training loop
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    logging.info(f"Epoch {epoch + 1}, Training Loss: {average_loss}")

# Save the trained model
model.save_pretrained("product_name_model")

# Inference on validation set
model.eval()
predictions = []

for batch in tqdm(val_loader, desc="Validation Inference"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=2).cpu().numpy())

# Decode predictions back to labels
decoded_predictions = []

for i, offsets in enumerate(val_encodings["offset_mapping"]):
    decoded_labels = []
    for j, (start, end) in enumerate(offsets):
        if j > 0 and start != offsets[j-1][1]:
            decoded_labels.append("O")
        label_id = predictions[i][j]
        decoded_labels.append("B-PRODUCT" if label_id == 1 else "O")
    decoded_predictions.append(decoded_labels)

# Add predicted labels to the DataFrame
df["predicted_labels"] = [" ".join(decoded_labels) for decoded_labels in decoded_predictions]

# Save the DataFrame
df.to_csv(path_or_buf='predicted_dataset.csv',
              sep=',',
              index=False)
"""