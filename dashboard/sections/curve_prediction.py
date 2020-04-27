import streamlit as st
import pandas as pd
import altair as alt
from dashboard.data import raw_information
from dashboard.forecast import extract_features, features_to_training, ForecastModel


@st.cache
def filter_data(raw, countries, mode="full"):
    data = []
    for c in countries:
        df = raw[c][["date", "active"]].copy()
        df["country"] = c
        mid = df[df["active"] == df["active"].max()]["date"].min()

        if mode == "up":
            df = df[df["date"] <= mid]
        elif mode == "down":
            df = df[df["date"] > mid]

        data.append(df)

    return pd.concat(data)


def run(tr):
    raw = raw_information()
    countries = list(raw.keys())
    selected_countries = [c for c in countries if raw[c]["active"].max() > 10000]
    selected_countries = st.multiselect("Countries", countries, selected_countries)
    st.info("Por defecto están seleccionados los países con más de 10,000 casos.")

    data = filter_data(raw, selected_countries)

    if st.checkbox("Show data"):
        st.write(data)

    st.altair_chart(
        alt.Chart(data)
        .mark_line()
        .encode(x="date", y="active", color="country",)
        .interactive(),
        use_container_width=True,
    )

    st.write("### Predicción de la parte baja de la curva")

    st.write(
        "Visualizando solo las fechas superiores al dia de máximo número de casos activos."
    )

    data = filter_data(
        raw, selected_countries, mode=st.sidebar.selectbox("Problema", ["full", "up", "down"])
    )

    st.altair_chart(
        alt.Chart(data)
        .mark_line()
        .encode(x="date", y="active", color="country")
        .interactive(),
        use_container_width=True,
    )

    features = []
    target = []

    st.sidebar.markdown("### Features")
    look_back = st.sidebar.slider("Look back days", 1, 15, 1)

    features = extract_features(
        data, look_back=look_back, grouper="country", keep=("country", "date")
    )

    if st.checkbox("Show features"):
        st.write(features)

    Xtrain, ytrain = features_to_training(features, drop=("country", "date"))

    st.sidebar.markdown("### Model hyperparameters")
    model = ForecastModel(
        max_iter=st.sidebar.number_input("Max iterations", 100, 10000, 1000)
    )

    model.fit(Xtrain, ytrain)
    st.write(model.score(Xtrain, ytrain))

    st.write(model.feature_importance())

    st.write("### Predicción")

    country = st.selectbox("Country", countries, countries.index('Cuba'))
    country_data = filter_data(raw, [country])

    country_features = extract_features(country_data, look_back=look_back, keep=('date',))

    Xtest, ytest = features_to_training(country_features, drop=('date',))
    ypredicted = model.predict(Xtest)
    country_features['predicted'] = ypredicted

    if st.checkbox("Show prediction data"):
        st.write(country_features)

    st.altair_chart(alt.Chart(country_features.melt(id_vars=['date'], value_vars=['target', 'predicted'])).mark_line().encode(
        x='date',
        y='value',
        color='variable'
    ), use_container_width=True)
    