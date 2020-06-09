import streamlit as st

from dashboard import (
    estimation,
    simulation,
)


def main():
    sections = {
        "Entrada de datos": estimation,
        "Simulación": simulation,
    }

    section = st.sidebar.selectbox(
        "Seleccionar sección", list(sections)
    )

    sections[section].run()


if __name__ == "__main__":
    main()
