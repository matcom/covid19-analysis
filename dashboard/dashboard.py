import textwrap
import streamlit as st
import datetime
import json
import pandas as pd
import altair as alt
import numpy as np
import graphviz
import itertools
import random

from altair import datum
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from pathlib import Path
from i18n import translate


# taken from <https://gist.github.com/jpconsuegra/45b63b68673044bd6074cf918c9a83b1>
def tab(section, title=None):
    import collections

    if not hasattr(tab, "__tabs__"):
        tab.__tabs__ = collections.defaultdict(dict)

        def run(sec, *args, **kwargs):
            func = st.sidebar.selectbox(sec, list(tab.__tabs__[sec]))
            func = tab.__tabs__[sec][func]
            func(*args, **kwargs)

        tab.run = run

    def wrapper(func):
        name = " ".join(s.title() for s in func.__name__.split("_"))
        tab.__tabs__[section][title or name] = func
        return func

    return wrapper


@st.cache
def get_responses():
    responses = pd.read_csv(
        Path(__file__).parent.parent / "data/responses.tsv", sep="\t"
    ).fillna("")
    responses["Date"] = pd.to_datetime(
        responses["Date"], format="%d/%m/%Y", errors="coerce"
    )
    responses = responses[responses["Date"] > "2020-01-01"]
    return responses


@st.cache
def demographic_data():
    return (
        pd.read_csv(
            Path(__file__).parent.parent / "data/world_demographics.tsv", sep="\t"
        )
        .set_index("Country")
        .to_dict("index")
    )


def country_similarity(source, country, stats_dict):
    source_stats = stats_dict[source]
    country_stats = stats_dict[country]

    similarity = []

    for key in ["Population", "Density", "Fertility", "Med. age", "Urban"]:
        baseline = source_stats[key]
        value = abs(country_stats[key] - baseline) / baseline
        if value == 0:
            return 1e50
        similarity.append(value)

    return sum(similarity) / len(similarity)


def most_similar_countries(country, count, stats_dict):
    all_similarities = {
        c: country_similarity(country, c, stats_dict) for c in stats_dict
    }
    return sorted(all_similarities, key=all_similarities.get)[:count]


def most_similar_curves(source, countries_to_analize, total):
    raw = raw_information()

    countries_to_analize = [c for c in countries_to_analize if c in raw]

    def get_data(country):
        df = raw[country]
        return df[df["confirmed"] > 0]["confirmed"].values

    source_data = get_data(source)

    exponent = 1
    normalize = True
    window = 15
    k = 7

    similarities = {
        country: sliding_similarity(
            source_data, get_data(country), exponent, normalize, window
        )
        for country in countries_to_analize
    }

    similarities = {c: (k, v) for c, (k, v) in similarities.items() if v is not None}
    return sorted(similarities.items(), key=lambda t: t[1][0])[:total]


@st.cache
def raw_information():
    with open(Path(__file__).parent.parent / "data" / "timeseries.json") as fp:
        raw_data = json.load(fp)

    data = {}
    for k, v in raw_data.items():
        df = pd.DataFrame(v)
        df = df[df["confirmed"] > 0]
        df["date"] = pd.to_datetime(df["date"])
        df["active"] = df["confirmed"] - df["recovered"] - df["deaths"]
        data[k] = df

    return data


@st.cache
def weekly_information(window_size: int = 7):
    raw_dfs = raw_information()

    dfs = []
    for country, df in raw_dfs.items():
        df = df.copy()
        start_day = df["date"].values[0]
        df["period"] = df["date"].apply(lambda t: (t - start_day).days // window_size)
        df["period"] = df["period"] - df["period"].min()
        df["new"] = df["confirmed"].diff().fillna(0)
        df = (
            df.groupby("period")
            .agg(
                confirmed=("confirmed", "max"),
                new=("new", "sum"),
                date=("date", "first"),
            )
            .reset_index()
        )
        df["growth"] = df["new"].pct_change().fillna(1.0)
        df["country"] = country
        df = df[(df["confirmed"] > 100) & (df["new"] > 10)]
        df = df[:-1]
        dfs.append(df)

    return pd.concat(dfs).reset_index()


def similarity(source, country, exponent=1, normalize=True):
    if len(country) < len(source):
        return 1e50

    min_len = min(len(source), len(country))
    cuba = source[0:min_len]
    country = country[0:min_len]

    def metric(vi, vj):
        t = abs(vi - vj)
        b = abs(vi) if normalize else 1
        return (t / b) ** exponent

    residuals = [metric(vi, vj) for vi, vj in zip(cuba, country)]
    msqe = sum(residuals) / len(residuals)

    return msqe


def sliding_similarity(source, country, exponent=1, normalize=True, window_size=15):
    min_sim = 1e50
    min_sample = None

    for i in range(window_size + 1):
        sample = country[i:]

        if len(sample) >= len(source):
            new_sim = similarity(source, sample, exponent, normalize)

            if new_sim < min_sim:
                min_sim = new_sim
                min_sample = sample

    return min_sim, min_sample


@st.cache
def get_measures_effects(responses: pd.DataFrame, data: pd.DataFrame, threshold):
    measures_effects = []

    for _, row in responses.iterrows():
        country = row["Country"]
        measure = row["Measure"]
        date = row["Date"]

        selected = data[
            (data["country"] == country)
            & (data["date"] >= date)
            # & (data['growth'] <- -threshold)
        ]

        # for i in np.arange(0, 1, 0.05):
        growth = selected[selected["growth"] <= -threshold]

        if len(growth) == 0:
            continue

        min_date = growth["date"].min()

        measures_effects.append(
            dict(
                country=country,
                measure=measure,
                category=row["Category"],
                taken=date,
                effect=min_date,
                distance=(min_date - date).days,
                size=threshold,
            )
        )

    return pd.DataFrame(measures_effects)


def main():
    st.write(
        "## COVID-19 Dashboard [(🔗 Github)](https://github.com/matcom/covid19-analysis)"
    )

    tr = translate(
        st.sidebar.selectbox("Language / Idioma", ["🇪🇸 Español", "🇬🇧 English"])
    )
    st.info(
        tr(
            "🇪🇸 Puedes cambiar el idioma en el sidebar a tu izquierda.",
            "🇬🇧 You can change the language in the sidebar to your left.",
        )
    )
    st.write(
        tr(
            """
            Welcome. In this dashboard you will find up-to-date information on COVID-19 
            including a variety of visualizations to help you understand what is going on.
            """,
            """
            Bienvenido. En este dashboard encontrarás información actualizada sobre el COVID-19
            incluyendo varias visualizaciones que te ayudarán a enteder mejor qué está sucediendo.
            """,
        )
    )

    section = tr("Select section", "Seleccionar sección")

    @tab(section, tr("View country details", "Ver detalles del país"))
    def view_country_details():
        st.subheader(tr("Country details", "Detalles del país"))

        st.write(
            tr(
                """
                This section shows the raw information for a given country.
                """,
                """
                Esta sección muestra la información cruda para un país determinado.
                """,
            )
        )

        raw = raw_information()
        countries = list(raw.keys())
        country = st.selectbox(
            tr("Select a country", "Selecciona un país"),
            countries,
            countries.index("Cuba"),
        )
        data = raw[country]
        data = data.melt(["date"])
        data = data[data["value"] > 0]
        # data = data[data['variable'] == 'confirmed']

        if st.checkbox(tr("Show raw data", "Mostrar datos")):
            st.write(data)

        scale = st.sidebar.selectbox(
            tr("Chart scale", "Tipo de escala"), ["linear", "log"]
        )

        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.5)
            .encode(
                x=alt.X("date", title=tr("Date", "Fecha")),
                y=alt.Y(
                    "value",
                    scale=alt.Scale(type=scale),
                    title=tr("Cases", "Casos"),
                    stack=False,
                ),
                color=alt.Color("variable", title=tr("Type", "Tipo")),
            )
            .properties(
                width=700,
                height=400,
                title=tr("Evolution of cases", "Evolución de los casos"),
            )
        )

        st.write(chart)

        if scale == "linear":
            st.info(
                "💡 "
                + tr(
                    """
                    **Suggestion:** change the scale to `log` to better appreciate small values.
                    In a linear scale (right now) an exponential growth is very hard to appreciate.
                    """,
                    """
                    **Sugerencia:** cambia la escala a `log` para poder apreciar mejor los valores pequeños.
                    En una escala lineal (como ahora) es más difícil de apreciar un crecimiento exponencial.
                    """,
                )
            )
        else:
            st.success(
                "💡"
                + tr(
                    """
                    In a logarithmic scale, smaller values are easier to appreciate. 
                    An exponential growth is seen here as a straight line.
                    """,
                    """
                    En una escala logarítmica, los valores pequeños son más fáciles de apreciar.
                    Un crecimiento exponencial en esta escala se ve como una linea recta.
                    """,
                )
            )

        data = demographic_data()
        raw = raw_information()

        mode = st.sidebar.selectbox("Compare with", ["Most similar", "Custom"])

        variable_to_look = st.sidebar.selectbox(
            "Variable", ["confirmed", "deaths", "recovered", "active"]
        )
        evacuated = st.sidebar.number_input("Evacuated cases (total)", 0, 1000, 0)

        def get_data(country):
            df = raw[country]
            return df[df[variable_to_look] > 0][variable_to_look].values

        if mode == "Most similar":
            similar_count = st.slider("Most similar countries", 5, len(data), 10)
            similar_countries = most_similar_countries(country, 3 * similar_count, data)

            if st.checkbox("Show partial selection"):
                st.write(similar_countries)
                st.write(
                    pd.DataFrame(
                        [
                            dict(country=k, **data[k])
                            for k in [country] + similar_countries
                        ]
                    )
                )

            similar_countries = most_similar_curves(
                country, similar_countries, similar_count
            )
        else:
            countries_to_compare = st.multiselect(
                "Countries to compare", list(set(countries) - {country})
            )
            similar_countries = {
                c: sliding_similarity(get_data(country), get_data(c))
                for c in countries_to_compare
            }
            similar_countries = {
                c: (k, v) for c, (k, v) in similar_countries.items() if v is not None
            }
            similar_countries = list(similar_countries.items())

        df = []

        for c, (_, data) in similar_countries:
            for i, x in enumerate(data):
                df.append(dict(pais=c, dia=i, casos=x))

        raw_country = raw[country]
        raw_country = raw_country[raw_country[variable_to_look] > 0][variable_to_look]

        for i, x in enumerate(raw_country):
            df.append(dict(pais=country, dia=i, casos=x))

        df = pd.DataFrame(df)

        alt.Chart(df).mark_line().encode(
            x="dia", y="casos", color="pais", tooltip="pais",
        ) + alt.Chart(df[df["pais"] == country]).mark_circle(
            size=100, fill="red"
        ).encode(
            x="dia", y="casos",
        ).properties(
            width=800, height=500
        ).interactive()

        st.write("### Forecast")

        serie = get_data(country)

        st.sidebar.markdown("### Forecast parameters")
        steps_back = st.sidebar.slider("Steps back", 1, len(serie) - 2, 7)
        skip_fraction = st.sidebar.slider("Skip fraction", 0.0, 0.25, 0.1)
        min_reports = st.sidebar.slider("Minimun number of reports", 0, 100, 5)
        use_values = True
        use_diferences = False

        def _extract_features(serie, X=None, y=None):
            X = [] if X is None else X
            y = [] if y is None else y

            serie = serie[int(skip_fraction * len(serie)) :]

            for i in range(steps_back, len(serie)):
                features = []

                if serie[i] < min_reports:
                    continue

                if use_values:
                    features.extend(serie[i - steps_back : i])
                if use_diferences:
                    for j in range(i - steps_back + 1, i):
                        features.append(serie[j] - serie[j - 1])

                current = serie[i]

                X.append(features)
                y.append(current)

            return X, y

        def extract_features(series):
            X = []
            y = []

            for country, serie in series.items():
                _extract_features(serie, X, y)

            return np.asarray(X), np.asarray(y)

        X, y = extract_features({k: v[1] for k, v in similar_countries})

        def build_model():
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)

            lr = Lasso(fit_intercept=False, positive=True, max_iter=10000, tol=0.001)
            lr.fit(Xtrain, ytrain)

            return lr

        def predict(model, data, n, previous=1):
            data = list(data)
            result = []

            for i in range(n):
                data.append(data[-1])
                X, y = _extract_features(data)
                X = X[-previous:]

                ypred = model.predict(X)
                result.append(ypred[0])
                data[-1] = ypred[0]

            return result

        previous = 1
        simulations = st.sidebar.slider("Simulations", 3, 30, 7)

        Y = []

        for i in range(30):
            lr = build_model()
            yp = predict(lr, serie, n=simulations, previous=previous)
            yp.insert(0, serie[-previous])
            Y.append(yp)

        Y = np.asarray(Y)

        ymean = Y.mean(axis=0)
        ystdv = Y.std(axis=0)

        real = []

        for i, d in enumerate(serie):
            real.append(dict(day=1 + i, value=d,))

        real = pd.DataFrame(real)

        forecast = []

        for i, (x, s) in enumerate(zip(ymean, ystdv)):
            x -= evacuated

            forecast.append(
                dict(
                    day=i + len(serie),
                    mean=round(x),
                    mean_50_up=round(0.67 * s + x),
                    mean_50_down=round(-0.67 * s + x),
                    mean_95_up=round(3 * s + x),
                    mean_95_down=round(-3 * s + x),
                )
            )

        if st.checkbox("Show data"):
            st.table(
                pd.DataFrame(
                    [
                        {
                            "Día": "Día %i" % d["day"],
                            "Predicción": d["mean"],
                            "50%% error": d["mean_50_up"],
                            "95%% error": d["mean_95_up"],
                        }
                        for d in forecast
                    ]
                ).set_index("Día")
            )

            st.write("#### Model parameters")
            st.write(lr.coef_)

        forecast = pd.DataFrame(forecast)

        scale = st.sidebar.selectbox("Scale", ["linear", "log"])

        prediction = (
            alt.Chart(forecast)
            .mark_line(color="red")
            .encode(x="day", y=alt.Y("mean", scale=alt.Scale(type=scale)),)
        )
        texts = prediction.mark_text(align="left", dx=5).encode(text="mean",)

        chart = (
            (
                alt.Chart(forecast)
                .mark_area(color="red", opacity=0.1)
                .encode(
                    x=alt.X("day", title="Date"),
                    y=alt.Y("mean_95_up", title="", scale=alt.Scale(type=scale)),
                    y2=alt.Y2("mean_95_down", title=""),
                )
                + alt.Chart(forecast)
                .mark_area(color="red", opacity=0.3)
                .encode(
                    x="day",
                    y=alt.Y("mean_50_up", title="", scale=alt.Scale(type=scale)),
                    y2=alt.Y2("mean_50_down", title=""),
                )
                + alt.Chart(forecast)
                .mark_circle(color="red")
                .encode(
                    x="day",
                    y=alt.Y("mean", scale=alt.Scale(type=scale)),
                    tooltip="mean",
                )
                + prediction
                + texts
                + alt.Chart(real)
                .mark_line(color="blue")
                .encode(x="day", y=alt.Y("value", scale=alt.Scale(type=scale)),)
                + alt.Chart(real)
                .mark_rule(color="blue")
                .encode(
                    x="day",
                    y=alt.Y("value", scale=alt.Scale(type=scale)),
                    tooltip="value",
                )
            )
            .properties(width=800, height=500,)
            .interactive()
        )

        st.write(chart)

    @tab(section, tr("Global epidemic evolution", "Evolución global de la epidemia"))
    def all_countries_curve():
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

    @tab(section, "Simulación de la epidemia")
    def simulation():
        st.write("### Simulando la epidemia")

        st.write(
            # """
            # Debido a irregularidades en las pruebas realizadas,
            # la cantidad de casos totales confirmados en un país puede ser un estimado
            # muy por debajo de la cantidad de casos reales.
            # Idealmente, cada país debería hacer pruebas a una porción considerable de la
            # población, mediante un muestreo aleatorio. Como en la práctica esto es virtualmente
            # imposible, en muchos lugares solamente se están realizando pruebas a los casos
            # que muestran síntomas, y más aún, a los casos más graves.
            """
            En esta sección construiremos un modelo muy sencillo para estimar, en un país
            arbitrario, cómo se vería la situación en función de la cantidad y la forma en
            la que se realicen las pruebas.
            """
        )

        st.info(
            "En el panel de la derecha aparecerán todos los parámetros de este modelo."
        )

        def simulate(
            days,
            simulate_hospital=False,
            simulate_quarantine=False,
            simulate_quarantine_infected=False,
        ):
            history = []

            infected = 1
            pop = population * 1000000
            susceptible = pop - infected
            dead = 0
            recovered = 0
            quarantined = 0

            for day in range(days):
                history.append(
                    dict(
                        day=day,
                        infected=infected,
                        quarantined=quarantined,
                        susceptible=susceptible,
                        dead=dead,
                        recovered=recovered,
                        has_care=0,
                        needs_care=0,
                    )
                )

                if infected < 1 and quarantined < 1:
                    break

                # En función de las medidas de cuarentena general los rates de infección
                # pueden variar
                if simulate_quarantine and infected >= quarantine_start:
                    people_met = quarantine_people
                    infect_rate = quarantine_infect
                else:
                    people_met = n_meet
                    infect_rate = p_infect

                # cada persona infecciosa puede infectar como promedio a cualquiera
                # de los susceptibles
                new_infected = min(
                    susceptible, infected * infect_rate * people_met * susceptible / pop
                )

                if simulate_hospital:
                    # personas enfermas que necesitan cuidados
                    need_bed = infected * percent_needs_beds
                    need_uci = infected * percent_needs_uci
                    need_vents = infected * percent_needs_vents

                    # personas que tienen los cuidados adecuados
                    has_bed = min(total_beds, need_bed)
                    has_uci = min(total_uci, need_uci)
                    has_vents = min(total_vents, need_vents)

                    # ajustamos el rate de fallecidos en función de los cuidados disponibles
                    has_care = has_bed + has_uci + has_vents
                    needs_care = need_bed + need_uci + need_vents
                    has_critical = has_uci + has_vents
                    is_critical = need_uci + need_vents
                    uncared = is_critical - has_critical
                    new_dead = (uncared) * uncare_death + (infected - uncared) * p_dead

                    history[-1]["has_care"] = has_care
                    history[-1]["needs_care"] = needs_care
                else:
                    # simulación simple
                    new_dead = p_dead * infected

                new_recovered = p_recover * infected

                infected = infected - new_dead - new_recovered + new_infected
                dead += new_dead
                recovered += new_recovered
                susceptible = susceptible - new_infected

                # Si se aplica cuarentena selectiva, los infectados tienen una probabilidad
                # de ser detectados y puestos en cuarentena
                if simulate_quarantine_infected:
                    new_quarantined = infected * chance_discover_infected
                    infected -= new_quarantined

                    # las personas en cuarentena también pueden morir or recuperarse
                    recovered_quarantined = quarantined * p_recover
                    dead_quarantined = quarantined * p_dead

                    recovered += recovered_quarantined
                    dead += dead_quarantined
                    quarantined = (
                        quarantined
                        + new_quarantined
                        - recovered_quarantined
                        - dead_quarantined
                    )

            df = pd.DataFrame(history).fillna(0).astype(int)

            # parámetros a posteriori
            df["r0"] = df["infected"].pct_change().fillna(0)
            df["new_dead"] = df["dead"].diff().fillna(0)
            df["letality"] = (df["new_dead"] / df["infected"]).fillna(0)
            df["healthcare_overflow"] = df["needs_care"] > df["has_care"]
            df["virus"] = (df["infected"] > 0) | (df["quarantined"] > 0)

            return df

        st.sidebar.markdown("### Datos demográficos")
        population = st.sidebar.number_input(
            "Población (millones de habitantes)", 1, 1000, 1
        )

        st.sidebar.markdown("### Parámetros epidemiológicos")
        n_meet = st.sidebar.slider("Personas en contacto diario", 1, 100, 20)
        p_infect = st.sidebar.slider("Probabilidad de infectar", 0.0, 1.0, 0.1)
        p_dead = st.sidebar.slider("Probabilidad de morir (diaria)", 0.0, 0.1, 0.02)
        p_recover = st.sidebar.slider(
            "Probabilidad de curarse (diaria)", 0.0, 0.5, 0.2
        )

        st.write("### Simulando una epidemia sin control (modelo SIR simple)")

        st.write(
            f"""
            Asumamos un país que tiene una población de `{population}` millones de habitantes. 
            La epidemia comienza el día 1, con un 1 paciente infectado.
            
            Asumamos que cada persona infectada es capaz de infectar a otra persona
            susceptible con una probabilidad de `{100 * p_infect:.0f}%`
            durante toda la duración de la enfermedad, asumiendo que no hay ningún tipo de control.

            Además, cada persona como promedio conocerá diariamente `{n_meet}` personas.
            De estas personas, cierta porción estará sana, y susceptible de ser infectada,
            y cierta porción estará infectada o recuperada. Por lo tanto, el crecimiento efectivo será
            de mayor al inicio, pero luego se frenará naturalmente a medida que toda la población
            es infectada.

            Asumiremos además que una persona infectada tiene una posibilidad de `{100 * p_dead:.0f}%` de morir
            o una probabilidad de `{100 * p_recover:.0f}%` de recuperarse cada día.
            """
        )

        st.write(
            """
            Este es el resultado de nuestra simulación. Dependiendo de los parámetros, o bien toda la población
            eventualmente será infectada, o la enfermedad ni siquiera logrará despegar. Con estos parámetros, el 
            resultado es el siguiente.
            """
        )

        simulation = simulate(1000)

        if st.checkbox("Show simulation data"):
            st.write(simulation)

        def summary(simulation):
            facts = []

            all_infected = simulation[(simulation["susceptible"] == 0)]
            none_infected = simulation[(simulation["virus"] == False)]
            total_infected = population * 1000000 - simulation["susceptible"].min()

            def ft(day, message, value=""):
                facts.append(
                    dict(
                        day=day,
                        message=message,
                        value=value,
                        # percent=value / (population * 10000) if value else "",
                    )
                )

            if len(all_infected) > 0:
                ft(
                    day=all_infected["day"].min(),
                    message="Toda la población ha sido infectada",
                )
            else:
                ft(
                    day=simulation["day"].max(),
                    message="Total de personas infectadas finalmente",
                    value=total_infected,
                )
                ft(
                    day=simulation["day"].max(),
                    message="Total de personas nunca infectadas",
                    value=simulation["susceptible"].min(),
                )
            if len(none_infected) > 0:
                ft(
                    day=none_infected["day"].min(),
                    message="La enfermedad ha desaparecido por completo",
                )
            else:
                ft(
                    day=simulation["day"].max()+1,
                    message="La enfermedad no desaparece por completo",
                )
            
            ft(
                day=simulation["day"].max()+1,
                message="Mortalidad promedio final",
                value=f"{100 * simulation['dead'].max() / total_infected:.1f}%"
            )

            ft(
                day=simulation[simulation["infected"] == simulation["infected"].max()][
                    "day"
                ].min(),
                message="Pico máximo de infectados",
                value=simulation["infected"].max(),
            )

            ft(
                day=simulation[simulation["dead"] == simulation["dead"].max()][
                    "day"
                ].min(),
                message="Cantidad total de fallecidos",
                value=simulation["dead"].max(),
            )

            max_r0 = simulation[simulation["infected"] > 100]["r0"].max()

            ft(
                day=simulation[simulation["r0"] == max_r0]["day"].min(),
                message="Máximo rate de contagio (> 100 infectados)",
                value=round(max_r0 + 1, 2),
            )

            healthcare_overflow = simulation[simulation["healthcare_overflow"] == True]

            if len(healthcare_overflow) > 0:
                ft(
                    day=healthcare_overflow["day"].min(),
                    message="Sistema de salud colapsado",
                )
                ft(
                    day=healthcare_overflow["day"].max(),
                    message="Sistema de salud vuelve a la normalidad",
                )
            else:
                ft(
                    day=simulation["day"].max()+1,
                    message="El sistema de salud no colapsa",
                )

            facts = pd.DataFrame(facts).sort_values("day").set_index("day")
            return facts

        st.table(summary(simulation))

        st.write(
            alt.Chart(
                simulation.melt(
                    "day", value_vars=["infected", "susceptible", "dead", "recovered"]
                )
            )
            .mark_line()
            .encode(
                x="day",
                y=alt.Y("value", scale=alt.Scale(type="linear")),
                color="variable",
            )
            .properties(width=700, height=400)
        )

        st.info(
            """
            Prueba a mover el parámetro `Personas en contacto diario` para que veas el efecto
            que produce disminuir el contacto diario, 
            o el parámetro `Probabilidad de infectar` para que veas
            el efecto de reducir la probabilidad de contagio mediante el uso de máscaras y otras medidas
            higiénicas.
            """
        )

        st.write("### Simulando el estrés en el sistema de salud")

        st.sidebar.markdown("### Sistema de salud")

        total_beds = st.sidebar.number_input(
            "Camas disponibles", 0, population * 1000000, population * 10000
        )
        total_uci = st.sidebar.number_input(
            "Camas UCI disponibles", 0, population * 1000000, population * 1000
        )
        total_vents = st.sidebar.number_input(
            "Respiradores disponibles", 0, population * 1000000, population * 1000
        )

        st.sidebar.markdown("### Parámetros epidemiológicos")

        percent_needs_beds = st.sidebar.slider(
            "Necesitan hospitalización", 0.0, 1.0, 0.20
        )
        percent_needs_uci = st.sidebar.slider(
            "Necesitan cuidados intensivos", 0.0, 1.0, 0.05
        )
        percent_needs_vents = st.sidebar.slider("Necesitan respirador", 0.0, 1.0, 0.05)

        uncare_death = st.sidebar.slider("Fallecimiento sin cuidado", 0.0, 1.0, 0.25)

        st.write(
            f"""
            La simulación anterior es extremadamente simplificada, sin tener en cuenta las
            características del sistema de salud. Vamos a modelar una situación ligeramente
            más realista, donde el `{100 * percent_needs_beds:.0f}%` de las personas infectadas
            necesitan algún tipo de hospitalización, el `{100 * percent_needs_uci:.0f}%`
            necesitan cuidados intensivos, y el `{100 * percent_needs_vents:.0f}%` necesitan respiradores. 
            En total, el `{100 * (percent_needs_beds + percent_needs_uci + percent_needs_vents):.0f}%` de
            los infectados necesitará algún tipo de cuidado. 

            El sistema de salud cuenta con un total de `{total_beds}` camas en los hospitales,
            un total de `{total_uci}` camas de cuidado intensivo, y un total de `{total_vents}` respiradores.

            Para las personas infectadas que requieren de cuidados especiales (UCI o respiradores), si estos no están 
            disponibles, la probabilidad de fallecer será entonces de `{100 * uncare_death:.0f}%` en vez de `{100 * p_dead:.0f}%`
            por cada día que se esté sin cuidados especiales.
            """
        )

        simulation = simulate(1000, simulate_hospital=True)

        if st.checkbox("Show simulation data", key=2):
            st.write(simulation)

        st.table(summary(simulation))

        st.write(
            alt.Chart(
                simulation.melt(
                    "day", value_vars=["infected", "susceptible", "dead", "recovered"]
                )
            )
            .mark_line()
            .encode(
                x="day",
                y=alt.Y("value", scale=alt.Scale(type="linear")),
                color="variable",
            )
            .properties(width=700, height=400)
        )

        healthcare_overflow = simulation[simulation["healthcare_overflow"] == True]
        healthcare_normal = simulation[
            (simulation["healthcare_overflow"] == False)
            & (simulation["infected"] > 100)
        ]

        if len(healthcare_overflow) > 0:
            st.write(
                f"""
                Veamos como se comporta el porciento efectivo de muertes y el número de muertes diarias. 
                En este modelo los hospitales colapsan el día `{healthcare_overflow['day'].min()}`
                y vuelven a la normalidad el día `{healthcare_overflow['day'].max()}`.
                Durante este período, la probabilidad de morir máxima es `{100 * healthcare_overflow['letality'].max():.2f}%`,
                mientras que fuera de este período (> 100 infectados) la probabilidad promedio es `{100 * healthcare_normal['letality'].mean():.2f}%`.
                """
            )
        else:
            st.write(
                f"""
                Veamos como se comporta el porciento efectivo de muertes y el número de muertes diarias. 
                En este modelo los hospitales no colapsan.
                Durante este período, la probabilidad de morir promedio se mantiene en `{healthcare_normal['letality'].mean():.3f}`.
                """
            )

        st.write(
            alt.Chart(simulation[simulation["infected"] > 100])
            .mark_line()
            .encode(
                x=alt.X("day", title="Días"),
                y=alt.Y("letality", title="Letalidad (% muertes / inf.)"),
            )
            .properties(width=700, height=200)
        )

        st.write(
            alt.Chart(simulation[simulation["infected"] > 100])
            .mark_bar()
            .encode(
                x=alt.X("day", title="Días"),
                y=alt.Y("new_dead", title="Muertes nuevas por día"),
            )
            .properties(width=700, height=200)
        )

        st.sidebar.markdown("### Medidas")
        st.write("### Efectividad de las medidas")

        apply_quarantine = st.sidebar.checkbox("Cuarentena general")

        if apply_quarantine:
            quarantine_start = st.sidebar.number_input(
                "Comienzo de la cuarentena (cantidad de infectados)", 0, population * 1000000, 10000
            )
            quarantine_people = st.sidebar.slider(
                "Cantidad de personas en contacto diario", 0, 100, 3
            )
            quarantine_infect = st.sidebar.slider(
                "Probabilidad de infección en cuarentena", 0.0, 1.0, 0.05
            )

        apply_infected_quarantine = st.sidebar.checkbox("Cuarentena a los infectados")

        if apply_infected_quarantine:
            chance_discover_infected = st.sidebar.slider(
                "Probabilidad de detectar infectado (diario)", 0.0, 1.0, 0.2
            )

        simulation = simulate(
            1000,
            simulate_hospital=True,
            simulate_quarantine=apply_quarantine,
            simulate_quarantine_infected=apply_infected_quarantine,
        )

        if st.checkbox("Mostrar datos", key="show_data_measures"):
            st.write(simulation)

        st.write("#### Cuarentena general")

        if apply_quarantine:
            st.write(
                f"""
                Veamos ahora como cambia el panorama si todas las personas se quedan en casa.
                Asumiremos que la cuarentena comienza cuando se detectan 
                `{quarantine_start}` infectados.
                Esto ocurre el día `{simulation[simulation['infected'] >= quarantine_start]['day'].min()}`.
                Al activar la cuarentena, debido a que las personas se quedan en casa, 
                se reduce la cantidad de contactos diarios a `{quarantine_people}` personas.
                Además, se reduce la probabilidad de contagio a un `{100 * quarantine_infect:.0f}%` debido
                a una mayor higiene en el hogar.
                Cuando la cantidad de infectados en un día se reduzca a menos de `{quarantine_start}`
                la cuarentena deja de entrar en vigor, y así sucesivamente.
                """
            )
        else:
            st.warning("Activa `Cuarentena general` en el menú para ver este efecto.")

        st.write("#### Cuarentena a los infectados")

        if apply_infected_quarantine:
            st.write(
                f"""
                Veamos ahora qué sucede si las personas que se detectan como infectados se 
                ponen en estricta cuarenta (cero contacto).
                Asumiremos que cada persona infectada tiene una probabilidad 
                de un `{100 * chance_discover_infected:.0f}%` de ser detectado cada día (puede que sea asintomático
                o puede que se demore en presentar síntomas). Una vez detectada es puesta en cuarentena
                y ya no puede infectar a nadie más, hasta que muera o se recupere.
                """
            )
        else:
            st.warning(
                "Activa `Cuarentena a los infectados` en el menú para ver este efecto."
            )

        st.table(summary(simulation))

        st.write(
            alt.Chart(
                simulation.melt(
                    "day",
                    value_vars=[
                        "infected",
                        "susceptible",
                        "dead",
                        "recovered",
                        "quarantined",
                    ],
                )
            )
            .mark_line()
            .encode(
                x="day",
                y=alt.Y("value", scale=alt.Scale(type="linear")),
                color="variable",
            )
            .properties(width=700, height=400)
        )

        st.write("### Código fuente")
        st.write("Si quieres ver el código fuente de la simulación, haz click aquí.")

        if st.checkbox("Ver código"):
            import inspect

            lines = inspect.getsource(simulate)
            lines = textwrap.dedent(lines)

            st.code(lines)

    tab.run(section)


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


if __name__ == "__main__":
    main()
