
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import numpy as np

class CitationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_text = self.data[idx]['query']
        entry_text = self.data[idx]['entry']
        label = self.data[idx]['label']

        query_tokens = self.tokenizer(query_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        entry_tokens = self.tokenizer(entry_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'query_input_ids': query_tokens['input_ids'].squeeze(),
            'query_attention_mask': query_tokens['attention_mask'].squeeze(),
            'entry_input_ids': entry_tokens['input_ids'].squeeze(),
            'entry_attention_mask': entry_tokens['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class DualVectorModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(DualVectorModel, self).__init__()
        self.query_model = BertModel.from_pretrained(model_name)
        self.entry_model = BertModel.from_pretrained(model_name)
        self.pooler = nn.AdaptiveAvgPool2d((1, 768))

    def forward(self, query_input_ids, query_attention_mask, entry_input_ids, entry_attention_mask):
        query_output = self.query_model(input_ids=query_input_ids, attention_mask=query_attention_mask)
        entry_output = self.entry_model(input_ids=entry_input_ids, attention_mask=entry_attention_mask)

        query_pooled_output = self.pooler(query_output.last_hidden_state).squeeze()
        entry_pooled_output = self.pooler(entry_output.last_hidden_state).squeeze()

        return query_pooled_output, entry_pooled_output

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        query_input_ids = batch['query_input_ids'].to(device)
        query_attention_mask = batch['query_attention_mask'].to(device)
        entry_input_ids = batch['entry_input_ids'].to(device)
        entry_attention_mask = batch['entry_attention_mask'].to(device)
        labels = batch['label'].to(device)

        query_vec, entry_vec = model(query_input_ids, query_attention_mask, entry_input_ids, entry_attention_mask)

        loss = criterion(query_vec, entry_vec, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def contrastive_loss(query_vec, entry_vec, labels, margin=1.0):
    distances = (query_vec - entry_vec).pow(2).sum(1)
    losses = labels * distances + (1 - labels) * nn.functional.relu(margin - distances.sqrt()).pow(2)
    return losses.mean()

if __name__ == '__main__':
    # Example data (replace with your actual dataset)
    data = [
        {'query': 'Transformer models in NLP', 'entry': 'BERT: Pre-training of Deep Bidirectional Transformers', 'label': 1},
        {'query': 'Transformer models in NLP', 'entry': 'RoBERTa: A Robustly Optimized BERT Pretraining Approach', 'label': 1},
        {'query': 'Transformer models in NLP', 'entry': 'GPT-3: Language Models are Few-Shot Learners', 'label': 0},
        # Add more data samples here
    ]

    # Training parameters
    model_name = 'bert-base-uncased'
    batch_size = 8
    epochs = 3
    learning_rate = 2e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset and dataloader
    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = CitationDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = DualVectorModel(model_name=model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = contrastive_loss

    # Training loop
    for epoch in range(epochs):
        avg_loss = train(model, dataloader, optimizer, criterion, device)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
