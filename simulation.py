import streamlit as st
import pandas as pd


@st.cache
def load_interaction_estimates():
    # no funciona, error "Could not find a version that satisfies the requirement xldr"
    df = pd.read_excel("/src/data/contact_matrices_152_countries/MUestimates_all_locations_1.xlsx", sheet_name=None)

    return df


def main():
    st.title("Simulación de la epidemia")

    interaction_estimates = load_interaction_estimates()

    st.write(interaction_estimates)


if __name__ == "__main__":
    main()
