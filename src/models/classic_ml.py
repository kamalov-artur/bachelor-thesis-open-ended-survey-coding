from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from features.embeddings import EmbeddingPoolingVectorizer
from models.binary_relevance import BinaryRelevanceClassifier

def build_classic_model(name, cfg, random_seed, model_params_override=None):
    model_cfg = cfg['models'][name]
    tfidf_cfg = cfg['tfidf']
    estimator = _base_estimator(name, model_cfg, random_seed, model_params_override)
    if model_cfg['feature_view'] == 'embedding_pooling':
        emb_cfg = model_cfg['embedding']
        steps = [('embedding_pooling', EmbeddingPoolingVectorizer(embedding_path=emb_cfg['path'], pooling=emb_cfg['pooling'], lowercase=emb_cfg.get('lowercase', True), normalize_output=emb_cfg.get('normalize_output', True), max_vocabulary=emb_cfg.get('max_vocabulary'), min_df=emb_cfg.get('min_df', 1)))]
    else:
        vectorizer = TfidfVectorizer(lowercase=tfidf_cfg['lowercase'], min_df=tfidf_cfg['min_df'], max_df=tfidf_cfg['max_df'], max_features=tfidf_cfg['max_features'], ngram_range=tuple(tfidf_cfg['word_ngram_range']), sublinear_tf=True, strip_accents='unicode')
        steps = [('tfidf', vectorizer)]
    if model_cfg['feature_view'] == 'svd_tfidf':
        steps.extend([('svd', SafeTruncatedSVD(model_cfg['svd_components'], random_seed)), ('norm', Normalizer(copy=False))])
    steps.append(('clf', BinaryRelevanceClassifier(estimator)))
    return Pipeline(steps)

def enabled_classic_models(cfg):
    return [name for name, model_cfg in cfg['models'].items() if model_cfg.get('enabled', False)]

def _base_estimator(name, cfg, random_seed, model_params_override=None):
    if name == 'gradient_boosting':
        if cfg.get('implementation') == 'catboost':
            return _catboost_estimator(model_params_override or cfg['search_space'][0], random_seed)
        return GradientBoostingClassifier(n_estimators=cfg['n_estimators'], learning_rate=cfg['learning_rate'], max_depth=cfg['max_depth'], random_state=random_seed)
    estimators = {
        'logistic_regression': LogisticRegression(C=cfg['C'], max_iter=cfg['max_iter'], solver='liblinear', class_weight='balanced', random_state=random_seed),
        'svm': LinearSVC(C=cfg['C'], class_weight='balanced', random_state=random_seed),
        'naive_bayes': ComplementNB(alpha=cfg['alpha']),
        'random_forest': RandomForestClassifier(n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'], class_weight='balanced_subsample', n_jobs=-1, random_state=random_seed),
    }
    return estimators[name]

class SafeTruncatedSVD(TruncatedSVD):

    def __init__(self, requested_components, random_state):
        super().__init__(n_components=requested_components, random_state=random_state)
        self.requested_components = requested_components

    def fit_transform(self, X, y=None):
        self.n_components = min(self.requested_components, max(1, X.shape[1] - 1))
        return super().fit_transform(X, y)

    def fit(self, X, y=None):
        self.n_components = min(self.requested_components, max(1, X.shape[1] - 1))
        return super().fit(X, y)

def _catboost_estimator(params, random_seed):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError('CatBoost is required for tuned gradient_boosting.') from exc
    return CatBoostClassifier(loss_function='Logloss', eval_metric='Logloss', auto_class_weights='Balanced', random_seed=random_seed, verbose=False, allow_writing_files=False, thread_count=-1, **params)
