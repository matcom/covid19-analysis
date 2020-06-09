import streamlit as st
import altair as alt
import pandas as pd
import textwrap


def run(tr):
    st.write("### Simulando la epidemia")

    st.write(
        """
        En esta sección construiremos un modelo muy sencillo para estimar, en un país
        arbitrario, cómo se vería la situación en función de la cantidad y la forma en
        la que se realicen las pruebas.
        """
    )

    st.info(
        "En el panel de la derecha aparecerán todos los parámetros de este modelo."
    )

    def simulate(
        days,
        simulate_hospital=False,
        simulate_quarantine=False,
        simulate_quarantine_infected=False,
    ):
        history = []

        infected = 1
        pop = population * 1000000
        susceptible = pop - infected
        dead = 0
        recovered = 0
        quarantined = 0

        for day in range(days):
            history.append(
                dict(
                    day=day,
                    infected=infected,
                    quarantined=quarantined,
                    susceptible=susceptible,
                    dead=dead,
                    recovered=recovered,
                    has_care=0,
                    needs_care=0,
                )
            )

            if infected < 1 and quarantined < 1:
                break

            # En función de las medidas de cuarentena general los rates de infección
            # pueden variar
            if simulate_quarantine and infected >= quarantine_start:
                people_met = quarantine_people
                infect_rate = quarantine_infect
            else:
                people_met = n_meet
                infect_rate = p_infect

            # cada persona infecciosa puede infectar como promedio a cualquiera
            # de los susceptibles
            new_infected = min(
                susceptible, infected * infect_rate * people_met * susceptible / pop
            )

            if simulate_hospital:
                # personas enfermas que necesitan cuidados
                need_bed = infected * percent_needs_beds
                need_uci = infected * percent_needs_uci
                need_vents = infected * percent_needs_vents

                # personas que tienen los cuidados adecuados
                has_bed = min(total_beds, need_bed)
                has_uci = min(total_uci, need_uci)
                has_vents = min(total_vents, need_vents)

                # ajustamos el rate de fallecidos en función de los cuidados disponibles
                has_care = has_bed + has_uci + has_vents
                needs_care = need_bed + need_uci + need_vents
                has_critical = has_uci + has_vents
                is_critical = need_uci + need_vents
                uncared = is_critical - has_critical
                new_dead = (uncared) * uncare_death + (infected - uncared) * p_dead

                history[-1]["has_care"] = has_care
                history[-1]["needs_care"] = needs_care
            else:
                # simulación simple
                new_dead = p_dead * infected

            new_recovered = p_recover * infected

            infected = infected - new_dead - new_recovered + new_infected
            dead += new_dead
            recovered += new_recovered
            susceptible = susceptible - new_infected

            # Si se aplica cuarentena selectiva, los infectados tienen una probabilidad
            # de ser detectados y puestos en cuarentena
            if simulate_quarantine_infected:
                new_quarantined = infected * chance_discover_infected
                infected -= new_quarantined

                # las personas en cuarentena también pueden morir or recuperarse
                recovered_quarantined = quarantined * p_recover
                dead_quarantined = quarantined * p_dead

                recovered += recovered_quarantined
                dead += dead_quarantined
                quarantined = (
                    quarantined
                    + new_quarantined
                    - recovered_quarantined
                    - dead_quarantined
                )

        df = pd.DataFrame(history).fillna(0).astype(int)

        # parámetros a posteriori
        df["r0"] = df["infected"].pct_change().fillna(0)
        df["new_dead"] = df["dead"].diff().fillna(0)
        df["letality"] = (df["new_dead"] / df["infected"]).fillna(0)
        df["healthcare_overflow"] = df["needs_care"] > df["has_care"]
        df["virus"] = (df["infected"] > 0) | (df["quarantined"] > 0)

        return df

    st.sidebar.markdown("### Datos demográficos")
    population = st.sidebar.number_input(
        "Población (millones de habitantes)", 1, 1000, 1
    )

    st.sidebar.markdown("### Parámetros epidemiológicos")
    n_meet = st.sidebar.slider("Personas en contacto diario", 1, 100, 20)
    p_infect = st.sidebar.slider("Probabilidad de infectar", 0.0, 1.0, 0.1)
    p_dead = st.sidebar.slider("Probabilidad de morir (diaria)", 0.0, 0.1, 0.02)
    p_recover = st.sidebar.slider("Probabilidad de curarse (diaria)", 0.0, 0.5, 0.2)

    st.write("### Simulando una epidemia sin control (modelo SIR simple)")

    st.write(
        f"""
        Asumamos un país que tiene una población de `{population}` millones de habitantes. 
        La epidemia comienza el día 1, con un 1 paciente infectado.
        
        Asumamos que cada persona infectada es capaz de infectar a otra persona
        susceptible con una probabilidad de `{100 * p_infect:.0f}%`
        durante toda la duración de la enfermedad, asumiendo que no hay ningún tipo de control.

        Además, cada persona como promedio conocerá diariamente `{n_meet}` personas.
        De estas personas, cierta porción estará sana, y susceptible de ser infectada,
        y cierta porción estará infectada o recuperada. Por lo tanto, el crecimiento efectivo será
        de mayor al inicio, pero luego se frenará naturalmente a medida que toda la población
        es infectada.

        Asumiremos además que una persona infectada tiene una posibilidad de `{100 * p_dead:.0f}%` de morir
        o una probabilidad de `{100 * p_recover:.0f}%` de recuperarse cada día.
        """
    )

    st.write(
        """
        Este es el resultado de nuestra simulación. Dependiendo de los parámetros, o bien toda la población
        eventualmente será infectada, o la enfermedad ni siquiera logrará despegar. Con estos parámetros, el 
        resultado es el siguiente.
        """
    )

    simulation = simulate(1000)

    if st.checkbox("Show simulation data"):
        st.write(simulation)

    def summary(simulation):
        facts = []

        all_infected = simulation[(simulation["susceptible"] == 0)]
        none_infected = simulation[(simulation["virus"] == False)]
        total_infected = population * 1000000 - simulation["susceptible"].min()

        def ft(day, message, value=""):
            facts.append(
                dict(
                    day=day,
                    message=message,
                    value=value,
                    # percent=value / (population * 10000) if value else "",
                )
            )

        if len(all_infected) > 0:
            ft(
                day=all_infected["day"].min(),
                message="Toda la población ha sido infectada",
            )
        else:
            ft(
                day=simulation["day"].max(),
                message="Total de personas infectadas finalmente",
                value=total_infected,
            )
            ft(
                day=simulation["day"].max(),
                message="Total de personas nunca infectadas",
                value=simulation["susceptible"].min(),
            )
        if len(none_infected) > 0:
            ft(
                day=none_infected["day"].min(),
                message="La enfermedad ha desaparecido por completo",
            )
        else:
            ft(
                day=simulation["day"].max() + 1,
                message="La enfermedad no desaparece por completo",
            )

        ft(
            day=simulation["day"].max() + 1,
            message="Mortalidad promedio final",
            value=f"{100 * simulation['dead'].max() / total_infected:.1f}%",
        )

        ft(
            day=simulation[simulation["infected"] == simulation["infected"].max()][
                "day"
            ].min(),
            message="Pico máximo de infectados",
            value=simulation["infected"].max(),
        )

        ft(
            day=simulation[simulation["dead"] == simulation["dead"].max()][
                "day"
            ].min(),
            message="Cantidad total de fallecidos",
            value=simulation["dead"].max(),
        )

        max_r0 = simulation[simulation["infected"] > 100]["r0"].max()

        ft(
            day=simulation[simulation["r0"] == max_r0]["day"].min(),
            message="Máximo rate de contagio (> 100 infectados)",
            value=round(max_r0 + 1, 2),
        )

        healthcare_overflow = simulation[simulation["healthcare_overflow"] == True]

        if len(healthcare_overflow) > 0:
            ft(
                day=healthcare_overflow["day"].min(),
                message="Sistema de salud colapsado",
            )
            ft(
                day=healthcare_overflow["day"].max(),
                message="Sistema de salud vuelve a la normalidad",
            )
        else:
            ft(
                day=simulation["day"].max() + 1,
                message="El sistema de salud no colapsa",
            )

        facts = pd.DataFrame(facts).sort_values("day").set_index("day")
        return facts

    st.table(summary(simulation))

    st.write(
        alt.Chart(
            simulation.melt("day", value_vars=["infected", "dead", "recovered"])
        )
        .mark_line()
        .encode(
            x="day",
            y=alt.Y("value", scale=alt.Scale(type="linear")),
            color="variable",
        )
        .properties(width=700, height=400)
    )

    st.info(
        """
        Prueba a mover el parámetro `Personas en contacto diario` para que veas el efecto
        que produce disminuir el contacto diario, 
        o el parámetro `Probabilidad de infectar` para que veas
        el efecto de reducir la probabilidad de contagio mediante el uso de máscaras y otras medidas
        higiénicas.
        """
    )

    st.write("### Simulando el estrés en el sistema de salud")

    st.sidebar.markdown("### Sistema de salud")

    total_beds = st.sidebar.number_input(
        "Camas disponibles", 0, population * 1000000, population * 10000
    )
    total_uci = st.sidebar.number_input(
        "Camas UCI disponibles", 0, population * 1000000, population * 1000
    )
    total_vents = st.sidebar.number_input(
        "Respiradores disponibles", 0, population * 1000000, population * 1000
    )

    st.sidebar.markdown("### Parámetros epidemiológicos")

    percent_needs_beds = st.sidebar.slider(
        "Necesitan hospitalización", 0.0, 1.0, 0.20
    )
    percent_needs_uci = st.sidebar.slider(
        "Necesitan cuidados intensivos", 0.0, 1.0, 0.05
    )
    percent_needs_vents = st.sidebar.slider("Necesitan respirador", 0.0, 1.0, 0.05)

    uncare_death = st.sidebar.slider("Fallecimiento sin cuidado", 0.0, 1.0, 0.25)

    st.write(
        f"""
        La simulación anterior es extremadamente simplificada, sin tener en cuenta las
        características del sistema de salud. Vamos a modelar una situación ligeramente
        más realista, donde el `{100 * percent_needs_beds:.0f}%` de las personas infectadas
        necesitan algún tipo de hospitalización, el `{100 * percent_needs_uci:.0f}%`
        necesitan cuidados intensivos, y el `{100 * percent_needs_vents:.0f}%` necesitan respiradores. 
        En total, el `{100 * (percent_needs_beds + percent_needs_uci + percent_needs_vents):.0f}%` de
        los infectados necesitará algún tipo de cuidado. 

        El sistema de salud cuenta con un total de `{total_beds}` camas en los hospitales,
        un total de `{total_uci}` camas de cuidado intensivo, y un total de `{total_vents}` respiradores.

        Para las personas infectadas que requieren de cuidados especiales (UCI o respiradores), si estos no están 
        disponibles, la probabilidad de fallecer será entonces de `{100 * uncare_death:.0f}%` en vez de `{100 * p_dead:.0f}%`
        por cada día que se esté sin cuidados especiales.
        """
    )

    simulation = simulate(1000, simulate_hospital=True)

    if st.checkbox("Show simulation data", key=2):
        st.write(simulation)

    st.table(summary(simulation))

    st.write(
        alt.Chart(
            simulation.melt("day", value_vars=["infected", "dead", "recovered"])
        )
        .mark_line()
        .encode(
            x="day",
            y=alt.Y("value", scale=alt.Scale(type="linear")),
            color="variable",
        )
        .properties(width=700, height=400)
    )

    healthcare_overflow = simulation[simulation["healthcare_overflow"] == True]
    healthcare_normal = simulation[
        (simulation["healthcare_overflow"] == False)
        & (simulation["infected"] > 100)
    ]

    if len(healthcare_overflow) > 0:
        st.write(
            f"""
            Veamos como se comporta el porciento efectivo de muertes y el número de muertes diarias. 
            En este modelo los hospitales colapsan el día `{healthcare_overflow['day'].min()}`
            y vuelven a la normalidad el día `{healthcare_overflow['day'].max()}`.
            Durante este período, la probabilidad de morir máxima es `{100 * healthcare_overflow['letality'].max():.2f}%`,
            mientras que fuera de este período (> 100 infectados) la probabilidad promedio es `{100 * healthcare_normal['letality'].mean():.2f}%`.
            """
        )
    else:
        st.write(
            f"""
            Veamos como se comporta el porciento efectivo de muertes y el número de muertes diarias. 
            En este modelo los hospitales no colapsan.
            Durante este período, la probabilidad de morir promedio se mantiene en `{healthcare_normal['letality'].mean():.3f}`.
            """
        )

    st.write(
        alt.Chart(simulation[simulation["infected"] > 100])
        .mark_line()
        .encode(
            x=alt.X("day", title="Días"),
            y=alt.Y("letality", title="Letalidad (% muertes / inf.)"),
        )
        .properties(width=700, height=200)
    )

    st.write(
        alt.Chart(simulation[simulation["infected"] > 100])
        .mark_bar()
        .encode(
            x=alt.X("day", title="Días"),
            y=alt.Y("new_dead", title="Muertes nuevas por día"),
        )
        .properties(width=700, height=200)
    )

    st.sidebar.markdown("### Medidas")
    st.write("### Efectividad de las medidas")

    apply_quarantine = st.sidebar.checkbox("Cuarentena general")

    if apply_quarantine:
        quarantine_start = st.sidebar.number_input(
            "Comienzo de la cuarentena (cantidad de infectados)",
            0,
            population * 1000000,
            10000,
        )
        quarantine_people = st.sidebar.slider(
            "Cantidad de personas en contacto diario", 0, 100, 3
        )
        quarantine_infect = st.sidebar.slider(
            "Probabilidad de infección en cuarentena", 0.0, 1.0, 0.05
        )

    apply_infected_quarantine = st.sidebar.checkbox("Cuarentena a los infectados")

    if apply_infected_quarantine:
        chance_discover_infected = st.sidebar.slider(
            "Probabilidad de detectar infectado (diario)", 0.0, 1.0, 0.2
        )

    simulation = simulate(
        1000,
        simulate_hospital=True,
        simulate_quarantine=apply_quarantine,
        simulate_quarantine_infected=apply_infected_quarantine,
    )

    if st.checkbox("Mostrar datos", key="show_data_measures"):
        st.write(simulation)

    st.write("#### Cuarentena general")

    if apply_quarantine:
        st.write(
            f"""
            Veamos ahora como cambia el panorama si todas las personas se quedan en casa.
            Asumiremos que la cuarentena comienza cuando se detectan 
            `{quarantine_start}` infectados.
            Esto ocurre el día `{simulation[simulation['infected'] >= quarantine_start]['day'].min()}`.
            Al activar la cuarentena, debido a que las personas se quedan en casa, 
            se reduce la cantidad de contactos diarios a `{quarantine_people}` personas.
            Además, se reduce la probabilidad de contagio a un `{100 * quarantine_infect:.0f}%` debido
            a una mayor higiene en el hogar.
            Cuando la cantidad de infectados en un día se reduzca a menos de `{quarantine_start}`
            la cuarentena deja de entrar en vigor, y así sucesivamente.
            """
        )
    else:
        st.warning("Activa `Cuarentena general` en el menú para ver este efecto.")

    st.write("#### Cuarentena a los infectados")

    if apply_infected_quarantine:
        st.write(
            f"""
            Veamos ahora qué sucede si las personas que se detectan como infectados se 
            ponen en estricta cuarenta (cero contacto).
            Asumiremos que cada persona infectada tiene una probabilidad 
            de un `{100 * chance_discover_infected:.0f}%` de ser detectado cada día (puede que sea asintomático
            o puede que se demore en presentar síntomas). Una vez detectada es puesta en cuarentena
            y ya no puede infectar a nadie más, hasta que muera o se recupere.
            """
        )
    else:
        st.warning(
            "Activa `Cuarentena a los infectados` en el menú para ver este efecto."
        )

    st.table(summary(simulation))

    st.write(
        alt.Chart(
            simulation.melt(
                "day", value_vars=["infected", "dead", "recovered", "quarantined",],
            )
        )
        .mark_line()
        .encode(
            x="day",
            y=alt.Y("value", scale=alt.Scale(type="linear")),
            color="variable",
        )
        .properties(width=700, height=400)
    )

    st.write("### Código fuente")
    st.write("Si quieres ver el código fuente de la simulación, haz click aquí.")

    if st.checkbox("Ver código"):
        import inspect

        lines = inspect.getsource(simulate)
        lines = textwrap.dedent(lines)

        st.code(lines)
