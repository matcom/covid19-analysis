import streamlit as st


def run(tr):
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

    # st.info(
    #     tr(
    #         "🇪🇸 Puedes cambiar el idioma en el sidebar a tu izquierda.",
    #         "🇬🇧 You can change the language in the sidebar to your left.",
    #     )
    # )
