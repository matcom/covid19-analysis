import streamlit as st
import pandas as pd
import altair as alt

from sklearn.tree import export_graphviz

from ..data import *
from ..features import *


def run(tr):
    st.subheader(tr("Global epidemic evolution", "Evolución global de la epidemia"))

    st.write(
        tr(
            """
            The following graph shows a log/log plot of the average weekly number of new cases
            vs. the total number of confirmed cases.
            In this type of graph, most of the countries will follow a straight diagonal path
            during the pandemic stage, since the growth is exponential, hence the number of new cases
            is a factor of the total number of cases.
            It is very easy to see which countries are leaving the pandemic stage, since those
            will be shown as deviating from the diagonal and falling down pretty quickly.
            """,
            """
            La siguiente gráfica muestra una curva log/log de la cantidad promedio de nuevos casos semanales,
            contra la cantidad total de casos confirmados.
            En este tipo de gráfica, la mayoría de los países seguirán una línea diagonal durante todo el
            período de pandemia, ya que el crecimiento es exponencial, y por lo tanto el número de casos
            nuevos es siempre un factor multiplicado por el número total de casos.
            Es muy fácil ver qué países están saliendo del estado de pandemia, dado que esos países
            se verán desviados de la diagonal con una fuerte tendencia hacia abajo.
            """,
        )
    )

    window_size = st.slider("Window size (days)", 1, 15, 5)

    raw_dfs: pd.DataFrame = weekly_information(window_size)
    totals: pd.DataFrame = raw_dfs.groupby("country").agg(
        total=("confirmed", "max")
    )

    select_top = tr(
        "Countries with most cases", "Países con mayor cantidad de casos"
    )
    select_custom = tr("Custom selection", "Selección personalizada")
    selection_type = st.sidebar.selectbox(
        tr("Selection type", "Tipo de selección"), [select_top, select_custom]
    )
    all_countries = list(totals.index)

    if selection_type == select_top:
        total_countries = st.slider(
            tr("Number of countries to show", "Cantidad de países a mostrar"),
            1,
            len(all_countries),
            20,
        )
        selected_countries = list(
            totals.sort_values("total", ascending=False)[:total_countries].index
        )
    else:
        selected_countries = st.multiselect(
            tr("Select countries", "Selecciona los países"),
            all_countries,
            all_countries,
        )

    your_country = st.selectbox(
        "Select country", all_countries, all_countries.index("Cuba")
    )
    selected_countries.append(your_country)

    data = raw_dfs[raw_dfs["country"].isin(selected_countries)]

    if st.checkbox("Show data (all periods)"):
        st.write(data)

    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X(
                "confirmed",
                scale=alt.Scale(type="log"),
                title=tr("Total confirmed cases", "Casos totales confirmados"),
            ),
            y=alt.Y(
                "new",
                scale=alt.Scale(type="log"),
                title=tr(
                    "New cases (weekly average)", "Casos nuevos (promedio semanal)"
                ),
            ),
            color=alt.Color("country", title=tr("Country", "País")),
            tooltip="country",
        )
    )
    dots = (
        alt.Chart(data)
        .mark_point()
        .encode(
            x=alt.X("confirmed", scale=alt.Scale(type="log")),
            y=alt.Y("new", scale=alt.Scale(type="log")),
            color="country",
        )
    )

    text = chart.mark_text(align="left").encode(text="country")

    st.write((chart + text + dots).properties(width=800, height=600).interactive())

    st.write("### Prediciendo el efecto de cada medida")

    responses = get_responses()
    responses = responses[responses["Country"].isin(selected_countries)]
    confirmed = []
    new_cases = []

    for i, row in responses.iterrows():
        country = row["Country"]
        date = row["Date"]

        point_data = data[(data["country"] == country) & (data["date"] >= date)]

        if len(point_data) > 0:
            confirmed.append(point_data["confirmed"].values[0])
            new_cases.append(point_data["new"].values[0])
        else:
            confirmed.append(None)
            new_cases.append(None)

    responses["Confirmed"] = confirmed
    responses["New Cases"] = new_cases

    if st.checkbox("Show data (responses)"):
        st.write(responses)

    st.write("Estas son todas las medidas que se han tomado hasta la fecha.")

    chart = (
        alt.Chart(responses)
        .mark_circle(size=0.25)
        .encode(
            x=alt.X("Date"),
            y="Country",
            size=alt.Size("Confirmed", scale=alt.Scale(type="log", base=2)),
            color="Measure",
            tooltip="Measure",
        )
        .properties(width=900, height=550)
        .interactive()
    )

    st.write(chart)

    # chart = (
    #     alt.Chart(responses)
    #     .mark_line()
    #     .encode(
    #         x="Date",
    #         y=alt.Y("New Cases", scale=alt.Scale(type='log', base=10)),
    #         # size=alt.Size('Confirmed', scale=alt.Scale(type='log', base=2)),
    #         color="Country",
    #         tooltip="Country",
    #     )
    #     .properties(width=900, height=550).interactive()
    # )

    # st.write(chart)

    all_possible_responses = set(responses["Measure"])
    all_responses = responses.groupby("Country")

    explanation = st.empty()
    threshold = st.sidebar.slider("Safety threshold", 0.0, 1.0, 0.25)
    explanation.markdown(
        f"""
        Filtramos los países que se consideran `safe` como aquellos que tienen un descenso de más de `{threshold * 100} %` 
        de un período al siguiente."""
    )

    data["safe"] = data["growth"] < -threshold
    safe_countries = set(data[data["safe"] == True].index)

    st.write(
        data[data["safe"] == True]
        .groupby("country")
        .agg(crecimiento=("growth", "last"))
    )

    explanation = st.empty()

    days_before_effect = st.sidebar.slider(
        "Days before measure make effect", 0, 30, 15
    )

    explanation.markdown(
        f"""
        Considerando como efecto positivo, el producir un descenso mayor de `{threshold * 100} %` en un período,
        vamos a ver qué medidas tienen una mayor influencia en ese descenso, si son tomadas con 
        al`{days_before_effect}` días de adelanto al período deseado."""
    )

    model = st.sidebar.selectbox("Model", ["Logistic Regression", "Decision Tree"])

    classifier, vectorizer = predict_measures_importance(
        model, threshold, days_before_effect, data, all_responses
    )

    if model == "Decision Tree":
        graph = export_graphviz(
            classifier,
            feature_names=vectorizer.feature_names_,
            filled=True,
            rounded=True,
        )
        st.graphviz_chart(graph)
    elif model == "Logistic Regression":
        coeficients = vectorizer.inverse_transform(classifier.coef_)[0]
        coeficients = pd.DataFrame(
            [dict(Medida=m, Factor=f) for m, f in coeficients.items()]
        ).sort_values("Factor")

        st.write(coeficients)

        st.write(
            alt.Chart(coeficients)
            .mark_bar(size=10)
            .encode(
                x="Factor",
                y=alt.Y("Medida"),
                color=alt.condition(
                    alt.datum.Factor > 0,
                    alt.value("steelblue"),  # The positive color
                    alt.value("orange"),  # The negative color
                ),
            )
            .properties(height=600, width=600)
        )

    st.write("### Estimado para todos los días")

    model_parameters = predict_all_importances(
        model, threshold, data, all_responses
    )
    model_parameters["abs_factor"] = (
        model_parameters["factor"].abs() / model_parameters["factor"].max()
    )

    if st.checkbox("Show data (factors per day)"):
        st.write(model_parameters)

    chart = (
        alt.Chart(model_parameters)
        .transform_filter(alt.datum.factor > 0)
        .mark_circle()
        .encode(
            x=alt.X("days:N", title="Dias luego de la medida"),
            y=alt.Y("measure", title="Medidas a tomar"),
            size=alt.Size("abs_factor", title="Importancia relativa"),
            # color=alt.condition(
            #     alt.datum.factor > 0,
            #     alt.value("steelblue"),  # The positive color
            #     alt.value("orange"),  # The negative color
            # ),
        )
        .properties(width=900)
    )

    st.write(chart)

    st.write("### Estimado del tiempo de efecto de cada medida")

    st.write(
        f"""
        Veamos a cuántos días se nota por primera vez el efecto de una medida.
        Esta gráfica muestra la cantidad de países que han observado un decrecimiento
        de al menos `{threshold * 100}`% de los casos en un período de `{window_size}` días
        con respecto al período anterior, al cabo de **X** días de haber tomado una medida determinada.
        """
    )

    min_days, max_days = st.slider("Range of analysis", 0, 60, (7, 30))

    measures_effects = get_measures_effects(responses, data, threshold)
    measures_effects = measures_effects[
        (measures_effects["distance"] >= min_days)
        & (measures_effects["distance"] <= max_days)
    ]

    if st.checkbox("Show data (measures effects)"):
        st.write(measures_effects)

    chart = (
        alt.Chart(measures_effects)
        .mark_circle()
        .encode(
            x=alt.X(
                "distance:N",
                title=f"Días entre la medida y una disminución en un {threshold * 100}% de los casos",
            ),
            y=alt.Y("measure", title="Medidas tomadas"),
            color=alt.Color("category", title="Tipo de medida"),
            size=alt.Size("count(country)", title="Cantidad de países"),
            shape="category",
            tooltip="measure",
        )
    )

    st.write(chart)
