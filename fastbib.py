
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.neighbors import KDTree
import numpy as np

class DualVectorModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(DualVectorModel, self).__init__()
        self.query_model = BertModel.from_pretrained(model_name)
        self.entry_model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.pooler = nn.AdaptiveAvgPool2d((1, 768))

    def forward_query(self, input_text):
        tokens = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
        output = self.query_model(**tokens)
        pooled_output = self.pooler(output.last_hidden_state).squeeze()
        return pooled_output

    def forward_entry(self, input_text):
        tokens = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
        output = self.entry_model(**tokens)
        pooled_output = self.pooler(output.last_hidden_state).squeeze()
        return pooled_output

class BibliographyPreSelector:
    def __init__(self, model_name='bert-base-uncased'):
        self.model = DualVectorModel(model_name=model_name)
        self.tree = None
        self.entries = []

    def train(self, entries, negative_sample_ratio=1):
        entry_embeddings = []
        for entry in entries:
            emb = self.model.forward_entry(entry)
            entry_embeddings.append(emb.detach().numpy())
            self.entries.append(entry)

        entry_embeddings = np.vstack(entry_embeddings)
        self.tree = KDTree(entry_embeddings, metric='euclidean')

    def preselect_bibliography(self, query, k=10):
        query_embedding = self.model.forward_query(query).detach().numpy().reshape(1, -1)
        distances, indices = self.tree.query(query_embedding, k=k)
        selected_entries = [self.entries[i] for i in indices[0]]
        return selected_entries

if __name__ == '__main__':
    # Example usage
    entries = [
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "GPT-3: Language Models are Few-Shot Learners",
        # Add more bibliography entries here
    ]

    selector = BibliographyPreSelector()
    selector.train(entries)

    query = "Transformer models in NLP"
    selected_bibliography = selector.preselect_bibliography(query=query, k=3)

    print("Selected bibliography:")
    for entry in selected_bibliography:
        print(entry)
