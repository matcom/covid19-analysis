import streamlit as st

from ..simulation import Simulation


def run(tr):
    simulation = Simulation()

    simulation.add_state("Susceptible")
    simulation.add_state("Asymptomatic")
    simulation.add_state("Symptomatic")
    simulation.add_state("Recovered")
    simulation.add_state("Dead")

    def alive(d):
        return starting_population - d["Dead"]

    starting_population = st.sidebar.number_input(
        "Population", 0, 100, 10
    ) * 1000000

    # people getting sick
    st.sidebar.markdown("### Infection dynamics")

    n_meet = st.sidebar.slider("People meeting", 0, 100, 10)
    p_infect_asymptomatic = st.sidebar.slider(
        "Probability of infection from asymptomatic", 0.0, 1.0, 0.1
    )
    p_infect_symptomatic = st.sidebar.slider(
        "Probability of infection from symptomatic", 0.0, 1.0, 0.2
    )

    simulation.add_transition(
        "Susceptible",
        "Asymptomatic",
        lambda d: (
            d["Asymptomatic"]
            * n_meet
            * p_infect_asymptomatic
            * d["Susceptible"]
            / alive(d)
        ),
    )
    simulation.add_transition(
        "Susceptible",
        "Asymptomatic",
        lambda d: (
            d["Symptomatic"]
            * n_meet
            * p_infect_symptomatic
            * d["Susceptible"]
            / alive(d)
        ),
    )

    # people developing symptoms and disease
    st.sidebar.markdown("### Disease evolution")

    p_symptoms = st.sidebar.slider(
        "Chance of developing symptoms (daily)", 0.0, 1.0, 0.1
    )
    p_asympt_recover = st.sidebar.slider(
        "Chance of recovering for asymptomatic patients (daily)", 0.0, 1.0, 0.2
    )
    p_sympt_dead = st.sidebar.slider(
        "Chance of dying for symptomatic patients (daily)", 0.0, 1.0, 0.05
    )
    p_sympt_recover = st.sidebar.slider(
        "Chance of recovering for symptomatic patients (daily)", 0.0, 1.0, 0.1
    )

    simulation.add_transition("Asymptomatic", "Symptomatic", p_symptoms)
    simulation.add_transition("Asymptomatic", "Recovered", p_asympt_recover)
    simulation.add_transition("Symptomatic", "Dead", p_sympt_dead)
    simulation.add_transition("Symptomatic", "Recovered", p_sympt_recover)

    data = simulation.run(
        1000, Susceptible=starting_population, Asymptomatic=1
    ).astype(int)

    data["Sick"] = data["Asymptomatic"] + data["Symptomatic"]
    data["New dead"] = data["Dead"].diff().fillna(0)
    data["Letality"] = data["New dead"] / data["Sick"]
    data["Letality (smooth)"] = data['Letality'].rolling(10).mean()
    data = data[data["Sick"] > 0]

    visualize = list(data.columns)
    visualize = st.multiselect("Columns to visualize", visualize, ["Asymptomatic", "Symptomatic", "Dead"])

    st.line_chart(data[visualize])

    st.write("### Visualizing the graphical model")

    graph = simulation.graph()
    st.write(graph)