import streamlit as st

from dashboard.i18n import translate
from dashboard.sections import (
    country_analysis,
    global_analysis,
    # simulation,
    # new_simulation,
    # testing_analysis,
    # intro,
    curve_prediction,
    # cubadata,
    cuba_simulation,
)


def main():
    # st.write(
    #     "## COVID-19 Dashboard [(🔗 Github)](https://github.com/matcom/covid19-analysis)"
    # )

    # if st.text_input("Contraseña / Password:", type='password') != "oye el de la cornetica":
    #     st.error("Contraseña incorrecta / Wrong password")
    #     return

    tr = translate("🇪🇸 Español") #st.sidebar.selectbox("Language / Idioma", [, "🇬🇧 English"]))

    sections = {
        # "Entrada de datos": cubadata,
        # "Intro": intro,
        tr("Single country analysis", "Análsis de un país"): country_analysis,
        tr("Curve prediction", "Predicción de la bajada"): curve_prediction,
        tr("Global epidemic evolution", "Evolución global de la epidemia"): global_analysis,
        "Simulación": cuba_simulation,
        # tr("Simulation", "Simulación (I)"): simulation,
        # tr("Simulation (new / incomplete)", "Simulación (II)"): new_simulation,
        # tr("Testing analysis", "Análisis de las pruebas"): testing_analysis,
    }

    section = st.sidebar.selectbox(
        "Seleccionar sección", list(sections)
    )

    sections[section].run(tr)


if __name__ == "__main__":
    main()
