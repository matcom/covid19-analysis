from sklearn.linear_model import Lasso
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit

import numpy as np
import pandas as pd


def extract_features(
    df: pd.DataFrame,
    look_back: int = 1,
    series=("active",),
    target="active",
    grouper=None,
    keep=(),
):
    X = []

    if grouper:
        dfs = [group for name, group in df.groupby(grouper)]
    else:
        dfs = [df]

    for df in dfs:
        for i in range(look_back, len(df)):
            features = {}

            for column in keep:
                features[column] = df[column].values[i]

            for j in range(1, look_back + 1):
                for s in series:
                    features[f"{s}(n-{j})"] = df[s].values[i - j]

            features["target"] = df[target].values[i]

            X.append(features)

    return pd.DataFrame(X)


def features_to_training(data: pd.DataFrame, drop=(), target="target"):
    X = []
    y = []

    for record in data.to_dict("record"):
        for column in drop:
            record.pop(column)

        y.append(record.pop(target))
        X.append(record)

    return X, y


class ForecastModel(Pipeline):
    def __init__(
        self,
        max_iter: int = 1000,
        fit_intercept: bool = False,
        positive: bool = False,
        compute_cross_validation: bool = False,
        cross_validation_steps = 100,
    ):
        self.compute_cross_validation = compute_cross_validation
        self.cross_validation_steps = 100

        super().__init__(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "regressor",
                    Lasso(
                        max_iter=max_iter,
                        fit_intercept=fit_intercept,
                        positive=positive,
                    ),
                ),
            ]
        )

    def _index(self, X, indices):
        return [X[i] for i in indices]

    def fit(self, X, y=None, **fit_params):
        if self.compute_cross_validation:
            self.scores_ = []
            cv = ShuffleSplit(n_splits=self.cross_validation_steps)

            for train, test in cv.split(X):
                xtrain, xtest = self._index(X, train), self._index(X, test)
                ytrain, ytest = y[train], y[test]

                super().fit(xtrain, ytrain, **fit_params)
                ypred = self.predict(xtest)
                print(ytest, ypred)
                self.scores_.extend(np.abs(ytest - ypred) / ytest)

            self.scores_.sort()

        return super().fit(X, y=y, **fit_params)

    @property
    def validation(self):
        if not hasattr(self, 'scores_'):
            return None
        
        return np.mean(self.scores_)

    def confidence(self, p=0.5):
        if not hasattr(self, 'scores_'):
            return None
        
        return self.scores_[int(p * (len(self.scores_) - 1))]

    def feature_importance(self):
        return self.named_steps["vectorizer"].inverse_transform(
            self.named_steps["regressor"].coef_.reshape(1, -1)
        )[0]
