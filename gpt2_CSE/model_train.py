import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset

# Define the paths to the dataset files
file_paths = [
    '../CSE-CIC-IDS2018/02-14-2018.csv',
    '../CSE-CIC-IDS2018/02-15-2018.csv',
    '../CSE-CIC-IDS2018/02-16-2018.csv',
    '../CSE-CIC-IDS2018/02-20-2018.csv',
    '../CSE-CIC-IDS2018/02-21-2018.csv',
    '../CSE-CIC-IDS2018/02-22-2018.csv',
    '../CSE-CIC-IDS2018/02-23-2018.csv',
    '../CSE-CIC-IDS2018/02-28-2018.csv',
    '../CSE-CIC-IDS2018/03-01-2018.csv',
    '../CSE-CIC-IDS2018/03-02-2018.csv',
    # Add all other file paths here
]

# Function to load and preprocess each file
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    data['Timestamp'] = pd.to_datetime(data['Timestamp']).dt.hour  # Convert timestamps to hour of the day
    return data

# Load all files and concatenate into a single DataFrame
full_data = pd.concat([load_and_preprocess(fp) for fp in file_paths], ignore_index=True)

# Fill missing values
full_data.fillna(0, inplace=True)

# Combine numerical and categorical features into a single text field
features_to_combine = [col for col in full_data.columns if col not in ['Label']]
full_data['text_data'] = full_data[features_to_combine].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Encode labels
label_encoder = LabelEncoder()
full_data['labels'] = label_encoder.fit_transform(full_data['Label'])

# Split data into training and testing sets
train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize data
def tokenize_function(data):
    return tokenizer(data['text_data'].tolist(), truncation=True, padding='max_length', max_length=512)

train_encodings = tokenize_function(train_data)
test_encodings = tokenize_function(test_data)

# Define a custom dataset class
class GPTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset objects
train_dataset = GPTDataset(train_encodings, train_data['labels'].values)
test_dataset = GPTDataset(test_encodings, test_data['labels'].values)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the GPT-2 model for sequence classification
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=len(label_encoder.classes_))

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()