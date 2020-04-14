import streamlit as st
import altair as alt

from ..data import testing_data


def run(tr):
    data = testing_data()

    if st.checkbox("Raw data"):
        st.write(data)

    countries = set(data['country'])
    countries_to_plot = st.multiselect("Plot only", sorted(countries))

    st.write(alt.Chart(data[data['country'].isin(countries_to_plot)]).mark_line().encode(
        x='date',
        color='country',
        y='total'
    ).properties(width=700).interactive())
