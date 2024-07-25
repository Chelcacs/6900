import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import GPT2Tokenizer,GPT2ForSequenceClassification, GPT2Config, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# Load the training dataset
train_file_path = '../NSL-KDD/KDDTrain+.txt'
train_data = pd.read_csv(train_file_path, header=None)

# Load the test dataset
test_file_path = '../NSL-KDD/KDDTest+.txt'
test_data = pd.read_csv(test_file_path, header=None)

# Assign column names as per the dataset description
column_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
                "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
                "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
                "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
                "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
                "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
                "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","attack","level"]

train_data.columns = column_names
test_data.columns = column_names

# Convert categorical data to numerical
categorical_columns = ["protocol_type", "service", "flag"]
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    train_data[col] = label_encoders[col].fit_transform(train_data[col])
    test_data[col] = label_encoders[col].transform(test_data[col])



# Convert labels to binary classification (normal vs attack)
train_data['attack'] = train_data['attack'].apply(lambda x: 0 if x == 'normal' else 1)
test_data['attack'] = test_data['attack'].apply(lambda x: 0 if x == 'normal' else 1)

# Split the train dataset into train and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)


# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token 
# Make sure to save and load the tokenizer for use in training to maintain configurations
tokenizer.save_pretrained('./tokenizer')
tokenizer = GPT2Tokenizer.from_pretrained('./tokenizer')

# Updated preprocess function
def preprocess(data):
    # Concatenate row values into a single string per row
    preprocessed_data = data.apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return tokenizer(preprocessed_data.tolist(), return_tensors="pt", padding="max_length", truncation=True, max_length=512)


# Apply preprocessing
X_train = preprocess(train_data.drop(['attack', 'level'], axis=1))
y_train = train_data['attack'].values
X_val = preprocess(val_data.drop(['attack', 'level'], axis=1))
y_val = val_data['attack'].values
X_test = preprocess(test_data.drop(['attack', 'level'], axis=1))
y_test = test_data['attack'].values

# model building
# Define a custom dataset class compatible with PyTorch
class IntrusionDetectionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Initialize the model with GPT-2 configuration
config = GPT2Config.from_pretrained('gpt2', pad_token_id=tokenizer.pad_token_id, num_labels=2)  # Binary classification
model = GPT2ForSequenceClassification(config)

# Load datasets
train_dataset = IntrusionDetectionDataset(X_train, y_train)
val_dataset = IntrusionDetectionDataset(X_val, y_val)
# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # directory for saving logs and model checkpoints
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=128,   # batch size for training
    per_device_eval_batch_size=128,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                # log training info every 10 steps
    evaluation_strategy="epoch",     # evaluate at the end of each epoch
    save_strategy="epoch",           # save model at the end of each epoch
    load_best_model_at_end=True      # load the best model at the end of training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Start training
trainer.train()

# Save the trained model
model_path = "./gpt2_intrusion_detection_model"
model.save_pretrained(model_path)
config.save_pretrained(model_path)
print("Model saved to", model_path)