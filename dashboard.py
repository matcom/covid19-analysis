import streamlit as st

from dashboard.i18n import translate
from dashboard.sections import country_analysis, global_analysis, simulation, new_simulation


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

sections = {
    tr("View country details", "Ver detalles del país"): country_analysis,
    tr("Global epidemic evolution", "Evolución global de la epidemia"): global_analysis,
    tr("Simulation", "Simulación de la epidemia"): simulation,
    tr("New simulación", "Simulación (nuevo)"): new_simulation,
}

section = st.sidebar.selectbox(tr("Select section", "Seleccionar sección"), list(sections))
sections[section].run(tr)
