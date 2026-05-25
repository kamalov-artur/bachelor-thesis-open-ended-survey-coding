import re
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from paths import PROJECT_ROOT
TOKEN_RE = re.compile("[A-Za-z][A-Za-z']+")

class EmbeddingPoolingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_path, pooling='tfidf_weighted_mean', lowercase=True, normalize_output=True, max_vocabulary=None, min_df=1):
        self.embedding_path = embedding_path
        self.pooling = pooling
        self.lowercase = lowercase
        self.normalize_output = normalize_output
        self.max_vocabulary = max_vocabulary
        self.min_df = min_df

    def fit(self, texts, y=None):
        self.tfidf_ = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None, lowercase=self.lowercase, min_df=self.min_df, max_features=self.max_vocabulary)
        self.tfidf_.fit(texts)
        vocab = set(self.tfidf_.vocabulary_)
        self.embeddings_, self.embedding_dim_, self.coverage_ = _load_embeddings(self.embedding_path, vocab, self.lowercase)
        return self

    def transform(self, texts):
        tfidf_matrix = self.tfidf_.transform(texts)
        feature_names = np.asarray(self.tfidf_.get_feature_names_out())
        output_dim = self.embedding_dim_ * 2 if self.pooling == 'mean_max' else self.embedding_dim_
        rows = np.zeros((len(texts), output_dim), dtype=np.float32)
        for i in range(tfidf_matrix.shape[0]):
            start, end = (tfidf_matrix.indptr[i], tfidf_matrix.indptr[i + 1])
            indices = tfidf_matrix.indices[start:end]
            weights = tfidf_matrix.data[start:end]
            vectors = []
            vector_weights = []
            for idx, weight in zip(indices, weights):
                token = feature_names[idx]
                vector = self.embeddings_.get(token)
                if vector is None:
                    continue
                vectors.append(vector)
                vector_weights.append(weight)
            if not vectors:
                continue
            matrix = np.vstack(vectors)
            if self.pooling == 'mean':
                rows[i] = matrix.mean(axis=0)
            elif self.pooling == 'tfidf_weighted_mean':
                w = np.asarray(vector_weights, dtype=np.float32)
                rows[i] = np.average(matrix, axis=0, weights=w)
            elif self.pooling == 'mean_max':
                w = np.asarray(vector_weights, dtype=np.float32)
                mean = np.average(matrix, axis=0, weights=w)
                maxv = matrix.max(axis=0)
                rows[i] = np.concatenate([mean, maxv])
        if self.normalize_output:
            rows = normalize(rows)
        return rows

def _tokenize(text):
    return TOKEN_RE.findall(text.lower())

def _load_embeddings(embedding_path, vocabulary, lowercase):
    path = Path(embedding_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    embeddings = {}
    with path.open('r', encoding='utf-8', errors='ignore') as fh:
        first = fh.readline().strip().split()
        if not (len(first) == 2 and all((part.isdigit() for part in first))):
            _maybe_add_embedding(first, vocabulary, lowercase, embeddings)
        for line in fh:
            parts = line.rstrip().split(' ')
            _maybe_add_embedding(parts, vocabulary, lowercase, embeddings)
    dim = len(next(iter(embeddings.values()))) if embeddings else 0
    coverage = {'vocabulary_size': float(len(vocabulary)), 'covered_tokens': float(len(embeddings)), 'coverage_share': float(len(embeddings) / len(vocabulary)) if vocabulary else 0.0}
    return (embeddings, dim, coverage)

def _maybe_add_embedding(parts, vocabulary, lowercase, embeddings):
    if len(parts) < 3:
        return
    token = parts[0].lower() if lowercase else parts[0]
    if token not in vocabulary:
        return
    try:
        embeddings[token] = np.asarray(parts[1:], dtype=np.float32)
    except ValueError:
        return
