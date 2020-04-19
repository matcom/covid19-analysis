import streamlit as st
import altair as alt

from ..data import raw_information
from ..simulation import Simulation
from ..simulation import compute_similarity
from ..simulation import optimize_similarity


def run(tr):
    def post_process(df):
        df["active"] = df["asymptomatic"] + df["symptomatic"]  # + df["Positive"]
        df["confirmed"] = simulation["pop"] - df["susceptible"]
        df["new_deaths"] = df["deaths"].diff().fillna(0)
        df["letality"] = df["new_deaths"] / df["active"]
        df["letality_smooth"] = df["letality"].rolling(10).mean().fillna(0)
        return df

    simulation = Simulation(after_run=post_process)

    simulation.add_state("susceptible")
    simulation.add_state("asymptomatic")
    simulation.add_state("symptomatic")
    simulation.add_state("recovered")
    simulation.add_state("deaths")
    # simulation.add_state("Positive")

    simulation["pop"] = (
        st.sidebar.number_input("Potentially susceptible population", 0, 1 * 1000 ** 3, 10 * 1000 ** 2)
    )

    # people getting active
    st.sidebar.markdown("### Infection dynamics")

    simulation["n_meet"] = st.sidebar.slider("People meeting", 0, 100, 10)
    simulation["n_test"] = st.sidebar.slider("Tests per day", 0, 10000, 1000)
    simulation["p_n_test"] = st.sidebar.slider(
        "Tests applied to symptomatic", 0.0, 1.0, 0.8
    )
    simulation["p_infect_asymptomatic"] = st.sidebar.slider(
        "Probability of infection from asymptomatic", 0.0, 1.0, 0.1
    )
    simulation["p_infect_symptomatic"] = st.sidebar.slider(
        "Probability of infection from symptomatic", 0.0, 1.0, 0.2
    )

    simulation.add_transition(
        "susceptible",
        "asymptomatic",
        "asymptomatic * n_meet * p_infect_asymptomatic * susceptible / (pop - deaths)",
    ),

    simulation.add_transition(
        "susceptible",
        "asymptomatic",
        "symptomatic * n_meet * p_infect_symptomatic * susceptible / (pop - deaths)",
    ),

    # simulation.add_transition(
    #     "asymptomatic",
    #     "Positive",
    #     "min(asymptomatic * n_test * (1 - p_n_test) / (pop - deaths), asymptomatic * (1 - (p_asympt_recover + p_symptoms)))",
    # )

    # simulation.add_transition(
    #     "symptomatic",
    #     "Positive",
    #     "min(symptomatic * (1 - (p_sympt_recover + p_sympt_dead)), n_test * p_n_test)",
    # )

    # people developing symptoms and disease
    st.sidebar.markdown("### Disease evolution")

    simulation["p_symptoms"] = st.sidebar.slider(
        "Chance of developing symptoms (daily)", 0.0, 1.0, 0.1
    )
    simulation["p_asympt_recover"] = st.sidebar.slider(
        "Chance of recovering for asymptomatic patients (daily)", 0.0, 1.0, 0.2
    )
    simulation["p_sympt_dead"] = st.sidebar.slider(
        "Chance of dying for symptomatic patients (daily)", 0.0, 1.0, 0.05
    )
    simulation["p_sympt_recover"] = st.sidebar.slider(
        "Chance of recovering for symptomatic patients (daily)", 0.0, 1.0, 0.1
    )

    simulation.add_transition(
        "asymptomatic", "symptomatic", "asymptomatic * p_symptoms"
    )
    simulation.add_transition(
        "asymptomatic", "recovered", "asymptomatic * p_asympt_recover"
    )
    simulation.add_transition("symptomatic", "deaths", "symptomatic * p_sympt_dead")
    simulation.add_transition(
        "symptomatic", "recovered", "symptomatic * p_sympt_recover"
    )
    # simulation.add_transition("Positive", "deaths", "Positive * p_sympt_dead")
    # simulation.add_transition("Positive", "recovered", "Positive * p_sympt_recover")

    st.write("### Comparison with a real curve")

    raw = raw_information()
    countries = list(raw.keys())
    country = st.selectbox("Select a country to compare with", countries, countries.index("Cuba"))

    real = raw[country].set_index('date')

    if st.checkbox("Show real country data"):
        st.write(real)

    columns = list(real.columns)
    columns = st.multiselect("Columns to compare", columns, columns)

    if st.checkbox("Find best parameters (slow)"):

        parameters = {}

        for param, value in simulation.parameters.items():
            if isinstance(value, float):
                min_value, max_value = st.slider(param, min_value=0.0, max_value=1.0, value=(0.0, 1.0))
            elif isinstance(value, int) and value < 100:
                min_value, max_value = st.slider(param, min_value=0, max_value=100, value=(0, 100))
            else:
                continue

            parameters[param] = (min_value, max_value) #, (max_value - min_value) / 10)

        if st.button("Run optimization"):
            best_params = optimize_similarity(simulation, real, columns, parameters, susceptible=simulation['pop']-1, asymptomatic=1)
            st.write(best_params)

    error, series = compute_similarity(simulation, real, columns, susceptible=simulation['pop']-1, asymptomatic=1)

    real_data_to_chart = real[columns].reset_index()
    simulation_data_to_chart = series[columns]
    simulation_data_to_chart['date'] = real_data_to_chart['date']

    if st.checkbox("Show simulated data"):
        st.write(simulation_data_to_chart)

    st.write(f"#### Approximation error: {error:.2f}")

    chart1 = alt.Chart(real_data_to_chart.melt("date")).mark_line().encode(
        x='date',
        y="value",
        color='variable',
    )

    chart2 = alt.Chart(simulation_data_to_chart.melt("date")).mark_line(size=3).encode(
        x='date',
        y="value",
        color='variable',
    )

    st.altair_chart(chart1 + chart2, use_container_width=True)

    st.sidebar.markdown("### Visualization")
    st.write("### Simulated curve")

    data = simulation.run(
        200, susceptible=simulation["pop"] - 1, asymptomatic=1
    ).astype(int)

    if st.checkbox("Show simulation data"):
        st.write(data)

    visualize = list(data.columns)
    visualize = st.multiselect(
        "Columns to visualize", visualize, ["asymptomatic", "symptomatic", "deaths"]
    )

    scale = st.sidebar.selectbox("Scale", ["linear", "log"])

    st.altair_chart(
        alt.Chart(data[visualize].reset_index().melt("index"))
        .transform_filter(alt.datum.value > 0)
        .mark_line()
        .encode(
            x="index", y=alt.Y("value", scale=alt.Scale(type=scale)), color="variable"
        ),
        use_container_width=True,
    )

    st.write("### Visualizing the graphical model")

    graph = simulation.graph()
    st.write(graph)
