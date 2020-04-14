import streamlit as st

from dashboard.i18n import translate
from dashboard.sections import (
    country_analysis,
    global_analysis,
    simulation,
    new_simulation,
    testing_analysis,
    intro,
)

st.write(
    "## COVID-19 Dashboard [(🔗 Github)](https://github.com/matcom/covid19-analysis)"
)

tr = translate(st.sidebar.selectbox("Language / Idioma", ["🇪🇸 Español", "🇬🇧 English"]))

sections = {
    "Intro": intro,
    tr("Single country analysis", "Análsis de un país"): country_analysis,
    tr("Global epidemic evolution", "Evolución global de la epidemia"): global_analysis,
    tr("Simulation", "Simulación"): simulation,
    tr("Testing analysis", "Análisis de las pruebas"): testing_analysis,
    tr("Simulation (new / incomplete)", "Simulación (nuevo / incompleto)"): new_simulation,
}

section = st.sidebar.selectbox(
    tr("Select section", "Seleccionar sección"), list(sections)
)

sections[section].run(tr)
