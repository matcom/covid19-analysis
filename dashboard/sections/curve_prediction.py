import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from dashboard.data import raw_information
from dashboard.forecast import extract_features, features_to_training, ForecastModel
from dashboard.data import demographic_data, get_responses
from dashboard.similarity import most_similar_countries, most_similar_curves


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
def features_per_country(
    data: pd.DataFrame,
    days_before=0,
    min_cases=100,
    use_demographics=True,
    use_responses=True,
):
    # esta es la fecha más actual en el dataset
    today = data["date"].max()
    # por cada país, cogemos el máximo de los valores activos
    max_active = data.groupby("country").agg(max_active=("active", "max")).reset_index()
    # tenemos por cada país el día en que se alcanzó el máximo de activos
    merged = pd.merge(
        max_active,
        data,
        left_on=["max_active", "country"],
        right_on=["active", "country"],
    )
    # queremos aquellos países que alcanzan este máximo antes de la fecha deseada
    merged = merged[merged["date"] <= today - pd.Timedelta(days=days_before)]
    # si hay más de un día en el que se alcancen esos valores de activos, nos quedamos
    # con el último de esos días
    merged = (
        merged.groupby("country")
        .agg(max_date=("date", "max"), max_active=("max_active", "min"))
        .reset_index()
    )

    # para contar la cantidad de días que han pasado desde que comenzó la epidemia
    # fijamos el día cero de cada país en el primer día donde hay al menos `min_cases` activos
    start_dates = (
        data[data["active"] > min_cases]
        .groupby("country")
        .agg(start_date=("date", "min"))
        .reset_index()
    )
    # mezclamos
    merged = pd.merge(merged, start_dates, on="country")
    # calculamos la cantidad de días que han pasado
    merged["total_days"] = (merged["max_date"] - merged["start_date"]).dt.days

    # ahora mezclamos con los datos demográficos
    if use_demographics:
        demographic = (
            demographic_data(as_dict=False)
            .reset_index()
            .rename(columns={"Country": "country"})
            .fillna(0)
        )
        merged = pd.merge(merged, demographic, on="country")

    # mezclamos con las medidas
    if use_responses:
        responses = get_responses()
        # agrupamos por país y categoría para obtener la cantidad de medidas tomadas por cada país en cada categoría
        responses = (
            responses[["Country", "Category", "Measure"]]
            .groupby(["Country", "Category"])
            .agg("count")
            .reset_index()
        )
        # al hacer un pivot las categorías se convierten en columnas, con el valor en cada celda
        responses = (
            responses.pivot_table(index="Country", columns="Category", values="Measure")
            .fillna(0)
            .reset_index()
        )
        # y mezclar
        merged = pd.merge(
            merged, responses.rename(columns={"Country": "country"}), on="country"
        )

    return merged


def run(tr):
    select_active = st.sidebar.number_input(
        "Preseleccionar países con # activos", 0, 10000000, 1000
    )

    raw = raw_information()
    countries = list(raw.keys())
    selected_countries = [
        c for c in countries if raw[c]["active"].max() > select_active
    ]
    st.info(
        f"Están pre-seleccionados los {len(selected_countries)} países con más de {select_active} casos."
    )
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

    st.write("## Estimación")

    st.sidebar.markdown("### Características del modelo")

    days_before = st.sidebar.slider("Días después de alcanzar el máximo", 0, 30, 7)
    min_cases = st.sidebar.number_input(
        "Número de casos donde empezar a contar", 0, 100000, 100
    )

    st.write(
        f"Se han selección los países que han alcanzado el máximo hace más de {days_before} días."
    )

    max_values = features_per_country(
        data,
        days_before=days_before,
        min_cases=min_cases,
        use_demographics=st.sidebar.checkbox("Usar características demográficas", True),
        use_responses=st.sidebar.checkbox("Usar características de las medidas", True),
    ).copy()

    X = max_values.drop(
        columns=["country", "max_date", "max_active", "start_date", "total_days"]
    ).to_dict(orient="records")

    predict_target = "total_days"  # st.selectbox("Objetivo a predecir", ["total_days", "max_active"])
    y = max_values[predict_target].values

    st.write(
        f"Prediciendo la columna `{predict_target}` a partir de los datos demográficos y de la cantidad de medidas tomadas."
    )

    st.write("### Datos")
    st.write(max_values)

    st.sidebar.markdown("### Parámetros de entrenamiento")

    predictor = ForecastModel(
        positive=st.sidebar.checkbox("Forzar factores positivos"),
        compute_cross_validation=st.sidebar.checkbox("Computar cross-validation", True),
        cross_validation_steps=st.sidebar.number_input(
            "Pasos de cross-validation", 1, 1000, 100
        ),
    )
    predictor.fit(X, y)
    max_values["predicted"] = predictor.predict(X).astype(int)
    max_values["predicted"] = max_values["predicted"].apply(lambda v: max(0, v))
    max_values["error"] = (
        max_values["predicted"] - max_values[predict_target]
    ).abs() / max_values[predict_target]

    st.write(
        f"#### Error relativo de ajuste (training error): `{max_values['error'].mean() * 100:.2f} %`"
    )
    st.write(
        f"#### Error relativo independiente (validation error): `{predictor.validation * 100:.2f} %`"
    )

    st.write("### Importancia de las características")
    st.write(predictor.feature_importance())

    st.write("## Predicción")

    country = st.selectbox("Seleccionar país", countries, countries.index("Cuba"))

    data_country = filter_data(raw, [country])
    data_country = features_per_country(
        data_country,
        days_before=0,
        min_cases=min_cases,
        use_demographics=True,
        use_responses=True,
    )

    X = data_country.drop(
        columns=["country", "max_date", "max_active", "start_date", "total_days"]
    ).to_dict(orient="records")

    st.write(X[0])

    prediction = predictor.predict(X)[0]
    predicted_values = pd.DataFrame(
        [
            dict(
                probabilidad=predictor.cdf(d),
                fecha=data_country["start_date"].values[0]
                + pd.Timedelta(days=int(prediction + d)),
            )
            for d in range(-5, 6)
        ]
    )

    st.write(predicted_values)

    st.altair_chart(
        alt.Chart(predicted_values)
        .mark_line()
        .encode(
            x=alt.X("fecha", title="Fecha esperada del pico máximo"),
            y=alt.Y("probabilidad", title="Probabilidad"),
        ),
        use_container_width=True,
    )

    st.write("## Estimación de la curva")

    features = []
    target = []

    st.sidebar.markdown("### Features")
    look_back = st.sidebar.slider("Look back days", 1, 15, 1)

    country = st.selectbox("Country", countries, countries.index("Cuba"))

    if st.checkbox("Preseleccionar países más similares (demográficamente)", True):
        selected_countries = most_similar_countries(
            country, st.slider("Cantidad", 10, 100, 20), demographic_data()
        )

    selected_countries = st.multiselect(
        "Países para estimar la curva", countries, selected_countries
    )

    if st.checkbox("Filtrar por similaridad de las curvas", True):
        similarity_by = st.selectbox("Similar por", ["recovered", "deaths"])
        most_similar = [
            c
            for c, _ in most_similar_curves(
                country,
                selected_countries,
                st.slider(
                    "Cantidad", 1, len(selected_countries), len(selected_countries) // 2
                ),
                similarity_by,
            )
        ]
        
        raw = raw_information()
        
        dfs = []
        for c in most_similar + [country]:
            df = raw[c].copy()
            df['country'] = c
            dfs.append(df)

        df = pd.concat(dfs)

        st.write(df)

        st.write(
            alt.Chart(df)
            .mark_line()
            .encode(
                x="date",
                y=alt.Y(similarity_by, scale=alt.Scale(type='linear')),
                color="country",
                tooltip="country",
            )
            + alt.Chart(df[df["country"] == country])
            .mark_circle(size=100, fill="red")
            .encode(x="date", y=alt.Y(similarity_by, scale=alt.Scale(type='linear')))
            .properties(width=800, height=500)
            .interactive()
        )

        selected_countries = most_similar

    features = extract_features(
        filter_data(raw, selected_countries, mode="down"),
        look_back=look_back,
        grouper="country",
        keep=("country", "date"),
        min_cases=min_cases,
    )

    if st.checkbox("Ver características"):
        st.info(
            f"Analizando puntos con más de {min_cases} activos y al menos {look_back} días de información."
        )
        st.write(features)

    Xtrain, ytrain = features_to_training(features, drop=("country", "date"))

    model = ForecastModel(
        max_iter=1000, compute_cross_validation=True, cross_validation_steps=100,
    )

    model.fit(Xtrain, ytrain)

    st.write(
        f"#### Error de ajuste (entrenamiento): `{100 - 100 * model.score(Xtrain, ytrain):.2f}%`"
    )
    st.write(
        f"#### Error de validación (independiente): `{model.validation * 100:.2}%`"
    )
    st.write(model.feature_importance())

    country_features_up = extract_features(
        filter_data(raw, [country], mode="up"), look_back=look_back, keep=("date",)
    )
    country_features_down = extract_features(
        filter_data(raw, [country], mode="down"), look_back=look_back, keep=("date",)
    )

    Xtest, ytest = features_to_training(country_features_down, drop=("date",))

    country_features = country_features_up.append(country_features_down)

    to_chart = country_features.melt(id_vars=["date"], value_vars=["target"])

    if len(Xtest) == 0:
        st.error(
            f"Este país no tiene curva de bajada con duración mayor de {look_back} días."
        )
        return

    if (
        st.selectbox(
            "Predecir a partir de", ["Último valor real", "Máximo de la curva"]
        )
        == "Máximo de la curva"
    ):
        x0 = Xtest[0]
        last_date = country_features_up["date"].max()
    else:
        x0 = Xtest[-1]
        last_date = country_features_down["date"].max()

    mappings = {f"active(n-{k})": f"active(n-{k-1})" for k in range(look_back, 1, -1)}
    mappings["active(n-1)"] = "yi"
    yfuture = model.estimate(
        x0, n=st.sidebar.number_input("Días a predecir", 1, 1000, 30), **mappings
    )

    predictions = []
    for i, yi in enumerate(yfuture):
        predictions.append(
            dict(date=last_date + pd.Timedelta(days=i), value=yi, variable="predicted",)
        )

    to_chart = to_chart.append(predictions)

    chart = st.altair_chart(
        alt.Chart(to_chart).mark_line().encode(x="date", y="value", color="variable"),
        use_container_width=True,
    )
