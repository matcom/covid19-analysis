import streamlit as st
import altair as alt

from ..session import get as get_session
from ..data import raw_information
from ..simulation import Simulation
from ..simulation import compute_similarity
from ..simulation import optimize_similarity


def run(tr):
    st.write("### Simulación")

    st.write(
        """
        A continuación mostramos una simulación basada en un modelo epidemiológico simple.
        Puedes modificar los parámetros a la izquierda para ver diferentes escenarios.
        
        Al final de esta página podrás experimentar con datos reales y ajustar los 
        parámetros de forma automática.
        """
    )

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

    session = get_session(parameters=dict())

    simulation["pop"] = st.sidebar.number_input(
        "Población susceptible potencial (máximo)",
        0,
        1 * 1000 ** 3,
        session.parameters.get("pop", 1 * 1000 ** 2),
    )

    # people getting active
    st.sidebar.markdown("### Parámetros de infección")

    simulation["n_meet"] = st.sidebar.slider(
        "Cantidad promedio de contacto diario",
        0,
        100,
        int(session.parameters.get("n_meet", 10.0)),
    )
    # simulation["n_test"] = st.sidebar.slider(
    #     "Tests per day", 0, 10000, int(session.parameters.get("n_test", 1000))
    # )
    # simulation["p_n_test"] = st.sidebar.slider(
    #     "Tests applied to symptomatic",
    #     0.0,
    #     1.0,
    #     session.parameters.get("p_n_test", 0.8),
    #     step=0.001,
    #     format="%.3f",
    # )
    simulation["p_infect_asymp"] = st.sidebar.slider(
        "Probabilidad de infectarse por contacto asimptomático",
        0.0,
        1.0,
        session.parameters.get("p_infect_asymp", 0.1),
        step=0.001,
        format="%.3f",
    )
    simulation["p_infect_symp"] = st.sidebar.slider(
        "Probabilidad de infectarse por contacto simptomático",
        0.0,
        1.0,
        session.parameters.get("p_infect_symp", 0.2),
        step=0.001,
        format="%.3f",
    )

    simulation.add_transition(
        "susceptible",
        "asymptomatic",
        "asymptomatic * n_meet * p_infect_asymp * susceptible / (pop - deaths)",
    ),

    simulation.add_transition(
        "susceptible",
        "asymptomatic",
        "symptomatic * n_meet * p_infect_symp * susceptible / (pop - deaths)",
    ),

    # simulation.add_transition(
    #     "asymptomatic",
    #     "Positive",
    #     "min(asymptomatic * n_test * (1 - p_n_test) / (pop - deaths), asymptomatic * (1 - (p_recover_asymp + p_symptoms)))",
    # )

    # simulation.add_transition(
    #     "symptomatic",
    #     "Positive",
    #     "min(symptomatic * (1 - (p_recover_symp + p_dead_symp)), n_test * p_n_test)",
    # )

    # people developing symptoms and disease
    st.sidebar.markdown("### Parámetros de evolución")

    simulation["p_symptoms"] = st.sidebar.slider(
        "Probabilidad (diaria) de desarrollar síntomas",
        0.0,
        1.0,
        session.parameters.get("p_symptoms", 0.1),
        step=0.001,
        format="%.3f",
    )
    simulation["p_recover_asymp"] = st.sidebar.slider(
        "Probabilidad (diaria) de curarse para asimptomáticos",
        0.0,
        1.0,
        session.parameters.get("p_recover_asymp", 0.2),
        step=0.001,
        format="%.3f",
    )
    simulation["p_dead_symp"] = st.sidebar.slider(
        "Probabilidad (diaria) de morir para simptomáticos",
        0.0,
        1.0,
        session.parameters.get("p_dead_symp", 0.05),
        step=0.001,
        format="%.3f",
    )
    simulation["p_recover_symp"] = st.sidebar.slider(
        "Probabilidad (diaria) de curarse para simptomáticos",
        0.0,
        1.0,
        session.parameters.get("p_recover_symp", 0.1),
        step=0.001,
        format="%.3f",
    )

    simulation.add_transition(
        "asymptomatic", "symptomatic", "asymptomatic * p_symptoms"
    )
    simulation.add_transition(
        "asymptomatic", "recovered", "asymptomatic * p_recover_asymp"
    )
    simulation.add_transition("symptomatic", "deaths", "symptomatic * p_dead_symp")
    simulation.add_transition(
        "symptomatic", "recovered", "symptomatic * p_recover_symp"
    )
    # simulation.add_transition("Positive", "deaths", "Positive * p_dead_symp")
    # simulation.add_transition("Positive", "recovered", "Positive * p_recover_symp")

    days = st.slider("Días a simular", 0, 1000, 100)

    data = simulation.run(
        days, susceptible=simulation["pop"] - 1, asymptomatic=1
    ).astype(int)

    if st.checkbox("Ver datos de la simulación (tabular)"):
        st.write(data)

    visualize = list(data.columns)
    visualize = st.multiselect(
        "Columnas a visualizar", visualize, ["asymptomatic", "symptomatic", "deaths"]
    )

    st.sidebar.markdown("### Visualización")
    scale = st.sidebar.selectbox("Escala", ["linear", "log"])

    st.altair_chart(
        alt.Chart(data[visualize].reset_index().melt("index"))
        .transform_filter(alt.datum.value > 0)
        .mark_line()
        .encode(
            x="index", y=alt.Y("value", scale=alt.Scale(type=scale)), color="variable"
        ),
        use_container_width=True,
    )

    st.write("### Explicación del modelo epidemiológico")

    graph = simulation.graph(edges=st.checkbox("Ver fórmulas en cada transición"))
    st.write(graph)

    st.write(
        f"""
        El grafo anterior muestra el modelo de simulación que estamos utilizando.
        La simulación comienza en el día 1, con un paciente asintomático infectado.
        A continuación te explicamos la dinámica de la simulación, ten en cuenta que es solamente
        un modelo altamente simplificado que no tiene en cuenta todos los posibles fenómenos.

        Asumiremos un total máximo de `{simulation['pop']:,d}` personas potencialmente susceptibles.
        Esta puede ser la población total de un país, una localidad, etc.

        Cada día, cada persona como promedio entra en contacto con `{simulation['n_meet']}` personas,
        por lo que si uno de ellos está enfermo y el otro es susceptible, potencialmente pudiera producirse una nueva infección.
        Las personas asintomáticas transmiten la enfermedad con una probablidad del `{simulation['p_infect_asymp'] * 100:.1f}%`,
        mientras que las personas sintomáticas transmiten la enfermedad con una 
        probablidad del `{simulation['p_infect_symp'] * 100:.1f}%`.

        Cada día, una persona asimptomática puede desarrollar sintómas con una probabilidad
        del `{simulation['p_symptoms'] * 100:0.1f}%` o recuperarse con una probablidad 
        del `{simulation['p_recover_asymp']*100:.1f}%`.
        En cambio una persona simptomática puede morir con una probabilidad
        del `{simulation['p_dead_symp']*100:.1f}%` o recuperarse con una probabilidad
        del `{simulation['p_recover_symp']*100:.1f}%`.
        Asumimos que las personas recuperadas no pueden volver a infectarse.
        """
    )

    st.write("### Comparación con datos reales")

    st.write(
        """
        En esta sección podrás comparar los resultados de la simulación con los datos
        reales de un país.
        """
    )

    raw = raw_information()
    countries = list(raw.keys())
    country = st.selectbox(
        "Seleccione un país para comparar", countries, countries.index("Cuba")
    )

    real = raw[country].set_index("date")

    if st.checkbox("Ver datos reales (tabular)"):
        st.write(real)

    columns = list(real.columns)
    columns = st.multiselect("Columnas a comparar", columns, columns)

    error, series = compute_similarity(
        simulation, real, columns, susceptible=simulation["pop"] - 1, asymptomatic=1
    )

    real_data_to_chart = real[columns].reset_index()
    simulation_data_to_chart = series[columns].copy()

    simulation_data_to_chart["date"] = real_data_to_chart["date"]

    st.write(
        """
        Las líneas punteadas representan los valores de la simulación, mientras
        que las líneas gruesas representan los valores reales. Prueba a ajustar
        los parámetros de la simulación para acercar cada línea punteada a la correspondiente
        línea gruesa.
        """
    )

    error_label = st.empty()
    comparison_chart = st.empty()

    st.write("### Optimizar paramétros de la simulación")

    st.write(
        f"""
        Ahora intentaremos encontrar automáticamente el mejor conjunto de parámetros
        para aproximar la curva de `{country}`.

        A continuación verás todos los parámetros del modelo. Puedes ajustar los rangos
        máximos y mínimos permisibles de estos parámetros. Si quieres que algún parámetro
        tenga un valor fijo, pon el máximo y mínimo al mismo valor.
        """
    )

    parameters = {}

    for param, value in simulation.parameters.items():
        if isinstance(value, float):
            min_value, max_value = st.slider(
                param, min_value=0.0, max_value=1.0, value=(0.0, 1.0)
            )
        elif isinstance(value, int) and value < 100:
            min_value, max_value = st.slider(
                param, min_value=0, max_value=100, value=(0, 100)
            )
        else:
            continue

        parameters[param] = (
            min_value,
            max_value,
        )

    if st.button("Ejecutar"):
        with st.spinner("🔧 Ejecutando optimización... Esto puede tardar unos segundos (de Windows 😝)..."):
            best_params = optimize_similarity(
                simulation,
                real,
                columns,
                parameters,
                # callback,
                susceptible=simulation["pop"] - 1,
                asymptomatic=1,
            )


        st.write("#### Mejores parámetros encontrados")
        st.write(best_params)
        session.parameters.update(best_params)

        st.success(
            """
            Estos son los mejores parámetros encontrados. La gráfica comparativa ya ha sido
            actualizada. Haz click en el siguiente botón para actualizar esta página con 
            estos parámetros.
            """)

        st.button("Recargar con estos parámetros")
    else:
        st.info(
            """
            Haz click en `Ejecutar` para iniciar la optimización. Este proceso puede tardar
            unos segundos. El algoritmo de optimización buscará entre los rangos de los parámetros
            que has establecido un juego de valores que produzca la mejor aproximación.
            Ten en cuenta que la solución es aproximada, el mejor juego de parámetros puede no ser alcanzable
            tanto porque los rangos no lo permiten, como porque el algoritmo de optimización es inherentemente
            aproximado.
            """
        )

    error, series = compute_similarity(
        simulation, real, columns, susceptible=simulation["pop"] - 1, asymptomatic=1
    )

    real_data_to_chart = real[columns].reset_index()
    simulation_data_to_chart = series[columns].copy()

    simulation_data_to_chart["date"] = real_data_to_chart["date"]

    error_label.markdown(f"#### Error de aproximación: {error:.2f}")

    chart1 = (
        alt.Chart(real_data_to_chart.melt("date"))
        .mark_line()
        .encode(x="date", y="value", color="variable",)
    )

    chart2 = (
        alt.Chart(simulation_data_to_chart.melt("date"))
        .mark_line(point=True)
        .encode(x="date", y="value", color="variable",)
    )

    comparison_chart.altair_chart(chart1 + chart2, use_container_width=True)
