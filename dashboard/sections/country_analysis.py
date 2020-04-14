import streamlit as st
import altair as alt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

from ..data import *
from ..similarity import *


def run(tr):
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

    rolling_smooth = st.sidebar.slider("Smooth rolling window", 1, 15, 1)
    step_size = st.sidebar.slider("Step size", 1, 15, 3)

    raw = raw_information(rolling_smooth, step_size)
    countries = list(raw.keys())
    country = st.selectbox(
        tr("Select a country", "Selecciona un país"),
        countries,
        countries.index("Cuba"),
    )
    data = raw[country]
    data = data.melt(['date'])
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
    mode = st.sidebar.selectbox("Compare with", ["Most similar", "Custom"])

    variable_to_look = st.sidebar.selectbox(
        "Variable", ["confirmed", "deaths", "recovered", "active"]
    )
    evacuated = st.sidebar.number_input("Evacuated cases (total)", 0, 1000, 0)

    def get_data(country):
        df = raw[country]
        return df[df[variable_to_look] > 0][variable_to_look].values

    if mode == "Most similar":
        similar_count = st.slider("Most similar countries", 5, len(data), 20)
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
            country, similar_countries, similar_count, variable_to_look, rolling_smooth, step_size
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
            df.append(dict(pais=c, dia=(i*step_size) + step_size-1, casos=x))

    raw_country = raw[country]
    raw_country = raw_country[raw_country[variable_to_look] > 0][variable_to_look]

    for i, x in enumerate(raw_country):
        df.append(dict(pais=country, dia=(i * step_size) + step_size - 1, casos=x))

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

    model = st.sidebar.selectbox(
        "Forecast model", ["Linear Regression", "Sampling"]
    )
    simulations = st.sidebar.slider("Simulations", 3, 30, 7)

    if model == "Linear Regression":
        st.sidebar.markdown("#### Linear Regression")

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

            lr = Lasso(
                fit_intercept=False, positive=True, max_iter=10000, tol=0.001
            )
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

        Y = []

        for i in range(30):
            lr = build_model()
            yp = predict(lr, serie, n=simulations, previous=previous)
            yp.insert(0, serie[-previous])
            Y.append(yp)

        Y = np.asarray(Y)

        ymean = Y.mean(axis=0)
        ystdv = Y.std(axis=0)

        if st.checkbox("Show linear regression parameters"):
            st.write(lr.coef_)

    elif model == "Sampling":
        values = [[serie[-1]]] + [[] for _ in range(simulations)]

        for k, v in similar_countries:
            country_serie = v[1][len(serie) :]
            for i, x in enumerate(country_serie):
                if i >= simulations:
                    break

                values[i + 1].append(x)

        smooth_factor = st.sidebar.slider("Smoothing factor", 0.0, 1.0, 0.0)

        def smooth(x, alpha):
            sx = [x[0]]
            sy = x[0]

            for i in range(1, len(x)):
                sy = sy * alpha + (1 - alpha) * x[i]
                sx.append(sy)

            return sx

        ymean = smooth([np.mean(v) for v in values], smooth_factor)
        ystdv = smooth([np.std(v) for v in values], smooth_factor)

    real = []

    for i, d in enumerate(serie):
        real.append(dict(day=1 + i * step_size, value=d,))

    real = pd.DataFrame(real)
    last_day = real['day'].max()

    forecast = []

    for i, (x, s) in enumerate(zip(ymean, ystdv)):
        x -= evacuated

        forecast.append(
            dict(
                day=i * step_size + last_day,
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