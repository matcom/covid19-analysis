import streamlit as st
import json
import pandas as pd
import altair as alt

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


@st.cache(allow_output_mutation=True)
def raw_information():
    with open(Path(__file__).parent.parent / "data" / "timeseries.json") as fp:
        raw_data = json.load(fp)

    data = {}
    for k, v in raw_data.items():
        df = pd.DataFrame(v)
        df["date"] = pd.to_datetime(df["date"])
        data[k] = df

    return data


@st.cache
def weekly_information():
    raw_dfs = raw_information()

    dfs = []
    for country, df in raw_dfs.items():
        df["week"] = df["date"].apply(lambda t: t.week)
        df["week"] = df["week"] - df["week"].min()
        df["new"] = df["confirmed"].diff().fillna(0)
        df = (
            df.groupby("week")
            .agg(confirmed=("confirmed", "max"), new=("new", "mean"))
            .reset_index()
        )
        df["country"] = country
        df = df[(df["confirmed"] > 10) & (df["new"] > 10)]
        dfs.append(df)

    return pd.concat(dfs)


def main():
    st.write("# COVID-19 Dashboard")

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

        if st.checkbox(tr("Show raw data", "Mostrar datos")):
            st.write(data)

        scale = st.sidebar.selectbox(
            tr("Chart scale", "Tipo de escala"), ["linear", "log"]
        )

        chart = alt.Chart(data[data["confirmed"] > 0]).mark_bar(
            color="darkblue"
        ).encode(
            x=alt.X("date", title=tr("Date", "Fecha")),
            y=alt.Y(
                "confirmed",
                scale=alt.Scale(type=scale),
                title=tr("Confirmed cases", "Casos confirmados"),
            ),
            tooltip="confirmed",
        ) + alt.Chart(
            data[data["deaths"] > 0]
        ).mark_bar(
            color="darkred",
        ).encode(
            x=alt.X("date"),  # , title=tr("Date", "Fecha")),
            y=alt.Y(
                "deaths", scale=alt.Scale(type=scale)
            ),  # , title=tr("Confirmed deaths", "Muertes confirmadas")),
            tooltip="deaths",
        ).properties(
            width=600, title=tr("Evolution of cases", "Evolución de los casos")
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

        st.write(tr(
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
            """
        ))

        raw_dfs: pd.DataFrame = weekly_information()
        totals: pd.DataFrame = raw_dfs.groupby('country').agg(total=('confirmed', 'max'))

        select_top = tr("Countries with most cases", "Países con mayor cantidad de casos")
        select_custom = tr("Custom selection", "Selección personalizada")
        selection_type = st.sidebar.selectbox(tr("Selection type", "Tipo de selección"), [select_top, select_custom])
        all_countries = list(totals.index)

        if selection_type == select_top:
            total_countries = st.slider(tr("Number of countries to show", "Cantidad de países a mostrar"), 1, len(all_countries), 10)
            selected_countries = totals.sort_values('total', ascending=False)[:total_countries].index
        else:
            selected_countries = st.multiselect(tr("Select countries", "Selecciona los países"), all_countries, all_countries)

        # countries = ['US', 'Italy', 'China', 'Spain', 'France', 'United Kingdom', 'Germany', 'Cuba', 'Korea, South', 'Japan']
        # countries = raw_data.keys()
        data = raw_dfs[raw_dfs['country'].isin(selected_countries)]

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X('confirmed', scale=alt.Scale(type='log')),
            y=alt.Y('new', scale=alt.Scale(type='log')),
            color='country',
            tooltip='country',
        )
        dots = alt.Chart(data).mark_point().encode(
            x=alt.X('confirmed', scale=alt.Scale(type='log')),
            y=alt.Y('new', scale=alt.Scale(type='log')),
            color='country',
        )

        text = chart.mark_text(align='left').encode(
            text='country'    
        )

        st.write((chart + text + dots).properties(width=700, height=500).interactive())

    tab.run(section)


if __name__ == "__main__":
    main()
