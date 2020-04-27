import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from dashboard.data import raw_information
from dashboard.forecast import extract_features, features_to_training, ForecastModel
from dashboard.data import demographic_data, get_responses


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


@st.cache
def max_per_country(data: pd.DataFrame, days_before=0, min_cases=100):
    # esta es la fecha más actual en el dataset
    today = data['date'].max()
    # por cada país, cogemos el máximo de los valores activos
    max_active = data.groupby("country").agg(max_active=('active', 'max')).reset_index()
    # tenemos por cada país el día en que se alcanzó el máximo de activos
    merged = pd.merge(max_active, data, left_on=['max_active', 'country'], right_on=['active', 'country'])
    # queremos aquellos países que alcanzan este máximo antes de la fecha deseada
    merged = merged[merged['date'] <= today - pd.Timedelta(days=days_before)]
    # si hay más de un día en el que se alcancen esos valores de activos, nos quedamos
    # con el último de esos días
    merged = merged.groupby('country').agg(max_date=('date', 'max'), max_active=('max_active', 'min')).reset_index()

    # para contar la cantidad de días que han pasado desde que comenzó la epidemia
    # fijamos el día cero de cada país en el primer día donde hay al menos `min_cases` activos
    start_dates = data[data['active'] > min_cases].groupby("country").agg(start_date=('date', 'min')).reset_index()
    # mezclamos
    merged = pd.merge(merged, start_dates, on='country')
    # calculamos la cantidad de días que han pasado 
    merged['total_days'] = (merged['max_date'] - merged['start_date']).dt.days

    # ahora mezclamos con los datos demográficos
    demographic = demographic_data(as_dict=False).reset_index().rename(columns={'Country': 'country'}).fillna(0)
    merged = pd.merge(merged, demographic, on='country')

    # mezclamos con las medidas
    responses = get_responses()
    # agrupamos por país y categoría para obtener la cantidad de medidas tomadas por cada país en cada categoría
    responses = responses[['Country', 'Category', 'Measure']].groupby(['Country', "Category"]).agg('count').reset_index()
    # al hacer un pivot las categorías se convierten en columnas, con el valor en cada celda
    responses = responses.pivot_table(index='Country', columns='Category', values='Measure').fillna(0).reset_index()
    # y mezclar
    merged = pd.merge(merged, responses.rename(columns={'Country': 'country'}), on='country')


    return merged


def run(tr):
    select_active = st.sidebar.number_input("Preseleccionar países con # activos", 0, 10000000, 1000)

    raw = raw_information()
    countries = list(raw.keys())
    selected_countries = [c for c in countries if raw[c]["active"].max() > select_active]
    st.info(f"Están pre-seleccionados los {len(selected_countries)} países con más de {select_active} casos.")
    selected_countries = st.multiselect("Countries", countries, selected_countries)

    data = filter_data(raw, selected_countries)

    if st.checkbox("Show data"):
        st.write(data)

    # data = filter_data(
    #     raw, selected_countries, mode=st.sidebar.selectbox("Problema", ["full", "up", "down"])
    # )

    st.altair_chart(
        alt.Chart(data)
        .mark_line()
        .encode(x="date", y="active", color="country")
        .interactive(),
        use_container_width=True,
    )

    st.write("## Predicción del pico máximo")

    days_before=st.sidebar.slider("Días después de alcanzar el máximo", 0, 30, 7)
    min_cases=st.sidebar.number_input("Número de casos donde empezar a contar", 0, 100000, 100)

    st.write(f"Se han selección los países que han alcanzado el máximo hace más de {days_before} días.")

    max_values = max_per_country(data, days_before=days_before, min_cases=min_cases).copy()

    X = max_values.drop(columns=['country', 'max_date', 'max_active', 'start_date', 'total_days']).to_dict(orient='records')

    predict_target = st.selectbox("Objetivo a predecir", ['total_days', 'max_active'])
    y = max_values[predict_target]

    st.write(f"Prediciendo la columna `{predict_target}` a partir de los datos demográficos y de la cantidad de medidas tomadas.")

    predictor = ForecastModel(positive=st.sidebar.checkbox("Forzar factores positivos"))
    predictor.fit(X, y)
    max_values['predicted'] = predictor.predict(X).astype(int)
    max_values['predicted'] = max_values['predicted'].apply(lambda v: max(0, v))
    max_values['error'] = (max_values['predicted'] - max_values[predict_target]).abs() / max_values[predict_target]
    
    st.write(f"#### Promedio de error relativo: `{max_values['error'].mean() * 100:.2f} %`")
    
    st.write("### Datos")
    st.write(max_values)
    st.write("### Importancia de las características")
    st.write(predictor.feature_importance())

    return

    st.write("## Estimación de la curva")

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
