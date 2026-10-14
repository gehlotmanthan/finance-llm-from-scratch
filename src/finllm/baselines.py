"""Classical baselines the LLM has to beat (Ch 6).

majority     always predict the most common training class
tfidf_lr     TF-IDF word 1-2 grams + multinomial logistic regression; the
             regularisation strength C is picked on the validation split only
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .stats import macro_f1


def majority_predict(train_labels, n: int) -> np.ndarray:
    vals, counts = np.unique(train_labels, return_counts=True)
    return np.full(n, vals[np.argmax(counts)])


def fit_tfidf_lr(train_texts, train_y, val_texts, val_y, grid=(0.25, 1.0, 4.0, 16.0), seed: int = 0):
    best = (-1.0, None, None)
    for C in grid:
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
            LogisticRegression(C=C, max_iter=2000, random_state=seed))
        pipe.fit(train_texts, train_y)
        f1 = macro_f1(np.asarray(val_y), pipe.predict(val_texts), len(set(train_y)))
        if f1 > best[0]:
            best = (f1, C, pipe)
    return best[2], {"C": best[1], "val_macro_f1": best[0]}
