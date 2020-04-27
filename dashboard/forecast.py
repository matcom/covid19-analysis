from sklearn.linear_model import Lasso
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

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
    def __init__(self, max_iter: int = 1000, fit_intercept: bool = False, positive: bool = False):
        super().__init__(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("regressor", Lasso(max_iter=max_iter, fit_intercept=fit_intercept, positive=positive)),
            ]
        )

    def feature_importance(self):
        return self.named_steps["vectorizer"].inverse_transform(
            self.named_steps["regressor"].coef_.reshape(1, -1)
        )[0]
