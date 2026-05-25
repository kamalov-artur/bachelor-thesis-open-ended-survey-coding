from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from paths import PROJECT_ROOT
PAD = '<pad>'
UNK = '<unk>'
EMBEDDING_MATRIX_CACHE = {}

class TextVocabulary:

    def __init__(self, max_size=50000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token_to_id = {PAD: 0, UNK: 1}

    def fit(self, texts):
        counts = Counter()
        for text in texts:
            counts.update(simple_tokenize(text))
        for token, count in counts.most_common(self.max_size - 2):
            if count < self.min_freq:
                break
            self.token_to_id[token] = len(self.token_to_id)
        return self

    def encode(self, text, max_tokens):
        ids = [self.token_to_id.get(token, 1) for token in simple_tokenize(text)[:max_tokens]]
        if len(ids) < max_tokens:
            ids += [0] * (max_tokens - len(ids))
        return ids

    def __len__(self):
        return len(self.token_to_id)

    @property
    def id_to_token(self):
        tokens = [None] * len(self.token_to_id)
        for token, idx in self.token_to_id.items():
            tokens[idx] = token
        return tokens

def simple_tokenize(text):
    return text.lower().replace('//', ' ').split()

class TextDataset(Dataset):

    def __init__(self, texts, y, vocab, max_tokens):
        self.x = torch.tensor([vocab.encode(text, max_tokens) for text in texts], dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

class RNNClassifier(nn.Module):

    def __init__(self, vocab_size, n_labels, embedding_dim, hidden_dim, bidirectional, dropout, embedding_weights=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = _embedding_layer(vocab_size, embedding_dim, embedding_weights, freeze_embeddings)
        self.rnn = nn.GRU(self.embedding.embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, n_labels)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        return self.classifier(self.dropout(hidden))

class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, n_labels, embedding_dim, channels, kernel_sizes, dropout, embedding_weights=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = _embedding_layer(vocab_size, embedding_dim, embedding_weights, freeze_embeddings)
        self.convs = nn.ModuleList([nn.Conv1d(self.embedding.embedding_dim, channels, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * len(kernel_sizes), n_labels)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            activated = torch.relu(conv(embedded))
            pooled.append(torch.max(activated, dim=2).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


@dataclass
class TorchTrainResult:
    model: nn.Module
    vocab: TextVocabulary
    val_scores: np.ndarray

def train_torch_text_model(model_name, train_texts, y_train, val_texts, y_val, model_cfg, training_cfg, random_seed):
    torch.manual_seed(random_seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    vocab = TextVocabulary().fit(train_texts)
    embedding_cfg = model_cfg.pop('pretrained_embedding', None)
    embedding_weights = build_embedding_matrix(vocab, embedding_cfg) if embedding_cfg else None
    freeze_embeddings = bool(embedding_cfg.get('freeze', False)) if embedding_cfg else False
    max_tokens = training_cfg['max_tokens']
    train_ds = TextDataset(train_texts, y_train, vocab, max_tokens)
    val_ds = TextDataset(val_texts, y_val, vocab, max_tokens)
    loader = DataLoader(train_ds, batch_size=training_cfg['batch_size'], shuffle=True)
    model_cls = {'rnn': RNNClassifier, 'cnn': CNNClassifier}[model_name]
    model = model_cls(len(vocab), y_train.shape[1], embedding_weights=embedding_weights, freeze_embeddings=freeze_embeddings, **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_loss = float('inf')
    patience_left = training_cfg['patience']
    for _ in range(training_cfg['max_epochs']):
        model.train()
        for xb, yb in loader:
            xb, yb = (xb.to(device), yb.to(device))
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_loss = _loss_on_dataset(model, val_ds, criterion, device, training_cfg['batch_size'])
        if val_loss < best_loss:
            best_loss = val_loss
            patience_left = training_cfg['patience']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores = predict_torch_scores(model, val_texts, vocab, max_tokens, training_cfg['batch_size'])
    return TorchTrainResult(model=model, vocab=vocab, val_scores=val_scores)

def build_embedding_matrix(vocab, embedding_cfg):
    kind = embedding_cfg['type']
    path = Path(embedding_cfg['path'])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cache_key = (kind, str(path), tuple(vocab.id_to_token))
    cached = EMBEDDING_MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if kind == 'glove_text':
        matrix = _load_text_embeddings(path, vocab, has_header=False)
    elif kind == 'word2vec_text':
        matrix = _load_text_embeddings(path, vocab, has_header=True)
    elif kind == 'word2vec_binary':
        matrix = _load_word2vec_binary(path, vocab)
    EMBEDDING_MATRIX_CACHE[cache_key] = matrix
    return matrix

def _embedding_layer(vocab_size, embedding_dim, embedding_weights, freeze_embeddings):
    if embedding_weights is None:
        return nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    weights = torch.tensor(embedding_weights, dtype=torch.float32)
    layer = nn.Embedding.from_pretrained(weights, freeze=freeze_embeddings, padding_idx=0)
    return layer

def _load_text_embeddings(path, vocab, has_header):
    token_to_id = vocab.token_to_id
    vectors = {}
    dim = None
    with path.open('r', encoding='utf-8', errors='ignore') as fh:
        if has_header:
            header = fh.readline().strip().split()
            if len(header) == 2 and header[1].isdigit():
                dim = int(header[1])
        for line in fh:
            parts = line.rstrip().split(' ')
            if len(parts) < 3:
                continue
            token = parts[0].lower()
            idx = token_to_id.get(token)
            if idx is None:
                continue
            try:
                vector = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            dim = dim or len(vector)
            if len(vector) == dim:
                vectors[idx] = vector
    return _finalize_embedding_matrix(vocab, vectors, dim or 0)

def _load_word2vec_binary(path, vocab):
    token_to_id = vocab.token_to_id
    vectors = {}
    with path.open('rb') as fh:
        header = fh.readline().decode('utf-8', errors='ignore').strip().split()
        total, dim = (int(header[0]), int(header[1]))
        for _ in range(total):
            token_bytes = bytearray()
            while True:
                ch = fh.read(1)
                if ch == b' ':
                    break
                if ch == b'\n':
                    continue
                token_bytes.extend(ch)
            token = token_bytes.decode('utf-8', errors='ignore').lower()
            idx = token_to_id.get(token)
            vector = np.fromfile(fh, dtype=np.float32, count=dim)
            if idx is not None and vector.shape[0] == dim:
                vectors[idx] = vector
    return _finalize_embedding_matrix(vocab, vectors, dim)

def _finalize_embedding_matrix(vocab, vectors, dim):
    rng = np.random.default_rng(42)
    matrix = rng.normal(0, 0.05, size=(len(vocab), dim)).astype(np.float32)
    matrix[0] = 0.0
    for idx, vector in vectors.items():
        matrix[idx] = vector
    coverage = len(vectors) / max(1, len(vocab) - 2)
    print(f'Loaded pretrained embeddings: covered {len(vectors)}/{len(vocab) - 2} tokens ({coverage:.1%}), dim={dim}')
    return matrix

def predict_torch_scores(model, texts, vocab, max_tokens, batch_size):
    device = next(model.parameters()).device
    dummy_y = np.zeros((len(texts), model.classifier.out_features), dtype=np.float32)
    ds = TextDataset(texts, dummy_y, vocab, max_tokens)
    loader = DataLoader(ds, batch_size=batch_size)
    model.eval()
    scores = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(scores)

def _loss_on_dataset(model, ds, criterion, device, batch_size):
    loader = DataLoader(ds, batch_size=batch_size)
    losses = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            losses.append(float(criterion(model(xb.to(device)), yb.to(device)).cpu()))
    return float(np.mean(losses))
