import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification

class TransformerTextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = [str(text) for text in texts]
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

class WeightedTransformerEncoder(nn.Module):

    def __init__(self, model_name, n_labels, dropout, encoder_trainable, pos_weight, local_files_only=False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=n_labels, problem_type='multi_label_classification', seq_classif_dropout=dropout, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True, local_files_only=local_files_only)
        if not encoder_trainable:
            base_model = getattr(self.model, self.model.base_model_prefix)
            for param in base_model.parameters():
                param.requires_grad = False
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)
        return {'loss': loss, 'logits': logits}

def make_encoder_loader(texts, labels, tokenizer, max_len, batch_size, shuffle):
    dataset = TransformerTextDataset(texts, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def predict_encoder_scores(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(scores)
