import streamlit as st
import datetime
import json
import pandas as pd
import altair as alt
import numpy as np
import graphviz

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
        growth = selected[selected['growth'] <= -threshold]

        if len(growth) == 0:
            continue

        min_date = growth["date"].min()

        measures_effects.append(
            dict(country=country, measure=measure, category=row['Category'], taken=date, effect=min_date, distance=(min_date - date).days, size=threshold)
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

        if st.checkbox("Show data (responses)"):
            st.write(responses)

        st.write("Estas son todas las medidas que se han tomado hasta la fecha.")

        chart = (
            alt.Chart(responses)
            .mark_line(size=0.25)
            .encode(
                x="Date",
                y="Country",
                color="Country",
                shape="Category",
                tooltip="Measure",
            )
            .properties(width=800, height=500)
        )

        st.write(chart)

        all_possible_responses = set(responses["Measure"])
        all_responses = responses.groupby("Country")

        features = []

        explanation = st.empty()
        threshold = st.sidebar.slider("Safety threshold", 0.0, 1.0, 0.25)
        explanation.markdown(
            f"""
            Filtramos los países que se consideran `safe` como aquellos que tienen un descenso de más de `{threshold * 100} %` 
            de un período al siguiente."""
        )

        data["safe"] = data["growth"] < -threshold
        safe_countries = set(data[data["safe"] == True].index)

        st.write(data[data["safe"] == True])

        explanation = st.empty()

        days_before_effect = st.sidebar.slider("Days before measure make effect", 0, 30, 15)

        explanation.markdown(
            f"""
            Considerando como efecto positivo, el producir un descenso mayor de `{threshold * 100} %` en un período,
            vamos a ver qué medidas tienen una mayor influencia en ese descenso, si son tomadas con 
            al`{days_before_effect}` días de adelanto al período deseado."""
        )

        for i, row in data.iterrows():
            country = row.country
            date = row.date
            growth = row.growth

            try:
                country_responses = all_responses.get_group(country)
            except KeyError:
                continue

            country_responses = country_responses[
                country_responses["Date"]
                <= date - pd.Timedelta(days=days_before_effect)
            ]

            if len(country_responses) == 0:
                continue

            features.append(
                dict(
                    growth=growth,
                    **{measure: True for measure in country_responses["Measure"]},
                )
            )

        # for c, responses in all_responses.get_group(your_country):
        #     st.write(c)
        #     st.write(responses[['Measure', 'Date']].to_dict())
        #     break

        # st.write(all_responses)

        # features = (
        #     responses[["Country", "Measure"]]
        #     .groupby("Country")
        #     .agg(lambda s: list(set(s)))
        #     .to_dict("index")
        # )

        vectorizer = DictVectorizer(sparse=False)
        X = vectorizer.fit_transform(
            [{k: True for k in featureset if k != "growth"} for featureset in features]
        )
        y = [featureset["growth"] < -threshold for featureset in features]

        model = st.sidebar.selectbox("Model", ["Logistic Regression", "Decision Tree"])

        if model == "Decision Tree":
            classifier = DecisionTreeClassifier()
        elif model == "Logistic Regression":
            classifier = LogisticRegression()

        acc_scores = cross_val_score(classifier, X, y, cv=10, scoring="accuracy")
        f1_scores = cross_val_score(classifier, X, y, cv=10, scoring="f1_macro")

        st.info(
            "**Precisión:** %0.2f (+/- %0.2f) - **F1:** %0.2f (+/- %0.2f)"
            % (
                acc_scores.mean(),
                acc_scores.std() * 2,
                f1_scores.mean(),
                f1_scores.std() * 2,
            )
        )

        classifier.fit(X, y)

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

        st.write("### Estimado del tiempo de efecto de cada medida")

        st.write(
            f"""
            Veamos a cuántos días se nota por primera vez el efecto de una medida.
            Esta gráfica muestra la cantidad de países que han observado un decrecimiento
            de al menos `{threshold * 100}`% de los casos en un período de `{window_size}`` días
            con respecto al período anterior, al cabo de **X** días de haber tomado una medida determinada.
            """
        )

        min_days, max_days = st.slider("Range of analysis", 0, 60, (7, 30))

        measures_effects = get_measures_effects(responses, data, threshold)
        measures_effects = measures_effects[(measures_effects['distance'] >= min_days) & (measures_effects['distance'] <= max_days)]

        if st.checkbox("Show data (measures effects)"):
            st.write(measures_effects)

        chart = alt.Chart(measures_effects).mark_circle().encode(
            y='measure',
            x='distance:N',
            color='category',
            shape='category',
            size='count(country)',
            tooltip='measure',
        )

        st.write(chart)


    @tab(section, tr("Similarity analysis", "Análisis de similaridad"))
    def similarity():
        st.write(tr("### Similarity analisis", "### Análisis de similaridad"))

        data = demographic_data()
        raw = raw_information()
        countries = list(data.keys())

        country = st.selectbox("Select a country", countries, countries.index("Cuba"))

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

    tab.run(section)


if __name__ == "__main__":
    main()
