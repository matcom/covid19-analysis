import pandas as pd
import streamlit as st
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def predict_all_importances(model, threshold, data, all_responses):
    result = []
    progress = st.progress(0.0)

    # thresholds = list(np.arange(0, 1, step=0.5))
    thresholds = [threshold]
    possible_days = list(range(7, 30))

    for i, thr in enumerate(thresholds):
        for j, days in enumerate(possible_days):
            clf, vect = predict_measures_importance(
                model, thr, days, data, all_responses, cv=False
            )
            progress.progress(
                (i * len(possible_days) + j) / (len(thresholds) * len(possible_days))
            )

            for k, v in vect.inverse_transform(clf.coef_)[0].items():
                result.append(dict(measure=k, factor=v, threshold=thr, days=days))

    return pd.DataFrame(result)


def predict_measures_importance(
    model, threshold, days_before_effect, data, all_responses, cv=True
):
    data = data.copy()
    data["safe"] = data["growth"] < -threshold
    features = []

    for i, row in data.iterrows():
        country = row.country
        date = row.date
        growth = row.growth

        try:
            country_responses = all_responses.get_group(country)
        except KeyError:
            continue

        country_responses = country_responses[
            country_responses["Date"] <= date - pd.Timedelta(days=days_before_effect)
        ]

        if len(country_responses) == 0:
            continue

        features.append(
            dict(
                growth=growth,
                **{measure: True for measure in country_responses["Measure"]},
            )
        )

    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(
        [{k: True for k in featureset if k != "growth"} for featureset in features]
    )
    y = [featureset["growth"] < -threshold for featureset in features]

    if model == "Decision Tree":
        classifier = DecisionTreeClassifier()
    elif model == "Logistic Regression":
        classifier = LogisticRegression()

    if cv:
        acc_scores = cross_val_score(classifier, X, y, cv=10, scoring="accuracy")
        f1_scores = cross_val_score(classifier, X, y, cv=10, scoring="f1_macro")

    # st.info(
    #     "**Precisión:** %0.2f (+/- %0.2f) - **F1:** %0.2f (+/- %0.2f)"
    #     % (
    #         acc_scores.mean(),
    #         acc_scores.std() * 2,
    #         f1_scores.mean(),
    #         f1_scores.std() * 2,
    #     )
    # )

    classifier.fit(X, y)
    return classifier, vectorizer
