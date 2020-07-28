import streamlit as st

from dashboard import (
    estimation,
    simulation,
)


def main():
    sections = {
        "Simulación": simulation,
        "Estimación": estimation,
    }

    section = st.sidebar.selectbox(
        "Seleccionar sección", list(sections)
    )

    sections[section].run()


if __name__ == "__main__":
    main()
