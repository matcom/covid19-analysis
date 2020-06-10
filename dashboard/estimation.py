import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import random
import networkx as nx
import collections

from .data import load_cuba_data


@st.cache
def get_events(data):
    events = []

    for i, d in data.iterrows():
        person_id = d["Cons"]

        try:
            age = int(d["Edad"])
            sex = "FEMALE" if d["Sexo"] == "F" else "MALE"
        except ValueError:
            continue

        farr = d['Fecha Arribo']

        state = "L" if pd.isna(farr) else "F"

        if d["Asintomatico"]:
            events.append(
                dict(
                    from_state=state,
                    to_state="A",
                    duration=0,
                    age=age,
                    sex=sex,
                    id=person_id,
                )
            )

            if d["Evolución"] == "Alta":
                events.append(
                    dict(
                        from_state="A",
                        to_state="R",
                        duration=(
                            pd.to_datetime(d["Fecha Alta"])
                            - pd.to_datetime(d["F. Conf"])
                        ).days,
                        age=age,
                        sex=sex,
                        id=person_id,
                    )
                )

            continue

        try:
            symptoms_start = pd.to_datetime(d["FIS"], format="%m/%d/%Y", errors="raise")
            admission_start = pd.to_datetime(d["FI"], format="%m/%d/%Y", errors="raise")
        except:
            continue

        events.append(
            dict(
                from_state=state,
                to_state="I",
                duration=0 if state == "L" else (symptoms_start - farr).days,
                age=age,
                sex=sex,
                id=person_id,
            )
        )

        events.append(
            dict(
                from_state="I",
                to_state="H",
                duration=(admission_start - symptoms_start).days,
                age=age,
                sex=sex,
                id=person_id,
            )
        )

        try:
            alta = pd.to_datetime(d["Fecha Alta"], format="%m/%d/%Y", errors="raise")
        except:
            continue

        if d["Evolución"] == "Fallecido":
            events.append(
                dict(
                    from_state="H",
                    to_state="D",
                    duration=(alta - admission_start).days,
                    age=age,
                    sex=sex,
                    id=person_id,
                )
            )
        elif d["Evolución"] == "Alta":
            events.append(
                dict(
                    from_state="H",
                    to_state="R",
                    duration=(alta - admission_start).days,
                    age=age,
                    sex=sex,
                    id=person_id,
                )
            )

    return pd.DataFrame(events)


@st.cache
def get_daily_values(data, asympt_length):
    day_states = []
    fend = data["Fecha Alta"].max() + pd.Timedelta(days=1)

    for i, row in data.iterrows():
        fe: pd.Timestamp = row["Fecha Arribo"]
        fs: pd.Timestamp = row["FIS"]
        fi: pd.Timestamp = row["FI"]
        fc: pd.Timestamp = row["F. Conf"]
        fa: pd.Timestamp = row["Fecha Alta"]

        contacts = row['# de contactos']

        if pd.isna(contacts):
            contacts = 0

        if pd.isna(fa):
            fa = fend

        if pd.isna(fc):
            continue

        if not pd.isna(fe):
            day_states.append(dict(day=fe, id=row["Cons"], status="nuevo-extranjero"))

        day_states.append(dict(day=fc, id=row["Cons"], status="nuevo-confirmado"))

        if fa < fend:
            day_states.append(dict(day=fa, id=row["Cons"], status="nuevo-alta"))

        for day in range((fa - fc).days):
            day_states.append(
                dict(
                    day=fc + pd.Timedelta(days=day), id=row["Cons"], status="activo"
                )
            )
        
        if not pd.isna(fi):
            day_states.append(dict(day=fi, id=row["Cons"], status="nuevo-ingreso"))
        
            for day in range((fa - fi).days):
                day_states.append(
                    dict(
                        day=fi + pd.Timedelta(days=day), id=row["Cons"], status="ingresado"
                    )
                )
                
        if not pd.isna(fs):
            day_states.append(dict(day=fs, id=row["Cons"], status="nuevo-síntoma"))
            day_states.append(dict(day=fs - pd.Timedelta(days=random.randint(0, asympt_length)), id=row["Cons"], status="infectado"))

            for day in range((fc - fs).days):
                day_states.append(
                    dict(
                        day=fs + pd.Timedelta(days=day),
                        id=row["Cons"],
                        status="infeccioso",
                    )
                )
        else:
            day_states.append(dict(day=fc - pd.Timedelta(days=random.randint(0, asympt_length)), id=row["Cons"], status="infectado"))

        if contacts == 0:
            continue

        if row['Asintomatico']:
            for day in range(asympt_length):
                day_states.append(
                    dict(
                        day=fc - pd.Timedelta(days=day),
                        id=row["Cons"],
                        status="contacto",
                        value=contacts / asympt_length
                    )
                )
        else:
            if pd.isna(fi) or pd.isna(fs):
                continue
            
            total_days = asympt_length + (fi - fs).days

            for day in range(total_days):
                day_states.append(
                    dict(
                        day=fi - pd.Timedelta(days=day),
                        id=row["Cons"],
                        status="contacto",
                        value=contacts / total_days
                    )
                )

    return pd.DataFrame(day_states).fillna(1)


def run():
    data = load_cuba_data()

    asympt_length = st.sidebar.number_input("Días promedio de desarrollar síntomas", 0, 100, 5)

    if st.checkbox("Ver datos raw"):
        st.write(data)  

    transitions(data)

    day_states = get_daily_values(data, asympt_length)
    foreigners = day_states[day_states['status'] == 'nuevo-extranjero']

    st.write("### Rate de llegada de extranjeros diaria")

    st.altair_chart(
        alt.Chart(foreigners).mark_bar().encode(
            x='day',
            y='sum(value)'
        ),
        use_container_width=True,
    )

    return


    st.altair_chart(
        alt.Chart(
            day_states[
                day_states["status"].isin(["nuevo-síntoma", "nuevo-confirmado", "nuevo-ingreso", "nuevo-alta"])
            ]
        )
        .mark_bar(opacity=0.75, size=3)
        .encode(
            x=alt.X("monthdate(day)"),
            y=alt.Y("count(id)", stack=True),
            color="status",
        ),
        use_container_width=True,
    )

    st.altair_chart(
        alt.Chart(day_states[day_states["status"].isin(["ingresado", "infeccioso", "activo"])])
        .mark_line()
        .encode(x="monthdate(day)", y="count(id)", color="status"),
        use_container_width=True,
    )

    st.altair_chart(
        alt.Chart(day_states[day_states["status"].isin(["contacto", "infectado"])])
        .mark_bar()
        .encode(x="monthdate(day)", y="sum(value)", color="status"),
        use_container_width=True,
    )

    df = day_states[day_states['status'].isin(['contacto', 'infectado'])].groupby(['day', 'status']).agg(value=('value', 'sum')).reset_index()
    df = df.pivot(index='day', columns='status', values='value').fillna(0).reset_index()
    df['rate'] = df['infectado'] / df['contacto']

    st.altair_chart(
        alt.Chart(df[df['contacto'] > 30])
        .mark_bar()
        .encode(x="monthdate(day)", y="rate"),
        use_container_width=True,
    )



def transitions(data: pd.DataFrame):
    st.write("## Estimación de transiciones para simulación")

    df = get_events(data)

    if st.checkbox("Ver todos los eventos"):
        st.write(df)

    st.write("### Transiciones generales")

    def compute_transitions(df):
        df = (
            df.groupby(["from_state", "to_state"])
            .agg(
                duration_mean=("duration", "mean"),
                duration_std=("duration", "std"),
                count=("id", "count"),
            )
            .fillna(0)
            .reset_index()
        )

        df["freq"] = df.apply(
            lambda r: r["count"]
            / df[df["from_state"] == r["from_state"]]["count"].sum(),
            axis="columns",
        )

        return df

    states = compute_transitions(df)
    st.write(states)

    graph = nx.DiGraph()

    for _, row in states.iterrows():
        graph.add_edge(row['from_state'], row['to_state'], frequency=row['freq'])

    st.graphviz_chart(str(nx.nx_pydot.to_pydot(graph)))

    if st.checkbox("Ver transiciones en CSV"):
        csv = []

        for age in range(0, 85, 5):
            for sex in ["FEMALE", "MALE"]:
                df_filter = df[(df["age"] >= age) & (df["age"] <= age * 5) & (df["sex"] == sex)]

                if len(df_filter) > 0:
                    states = compute_transitions(df_filter)
                    states["age"] = age
                    states["sex"] = sex
                    csv.append(states.set_index(["age", "sex", "from_state"]).round(3))

        df = pd.concat(csv)
        st.code(df.to_csv())


class TransitionEstimator:
    def __init__(self, data):
        self.data = data
        self.states = list(set(self.data["from_state"]))
        self._state_data = {}

        for s in self.states:
            self._state_data[s] = self.data[self.data["from_state"] == s]

    def transition(self, from_state, age, sex):
        """Computa las probabilidades de transición del estado `from_state` para una 
        persona con edad `age` y sexo `sex.
        """

        # De todos los datos que tenemos, vamos a ordenarlos por diferencia absoluta
        # con la edad deseada, y vamos cogiendo filas hasta acumular al menos 50
        # mediciones.
        evidence = self.data[
            (self.data["from_state"] == from_state) & (self.data["sex"] == sex)
        ].copy()
        evidence["AgeDiff"] = (evidence["age"] - age).abs()
        evidence = evidence.sort_values(["AgeDiff", "count"]).copy()
        evidence["CountCumul"] = evidence["count"].cumsum()

        if evidence["CountCumul"].max() > 50:
            min_required_evidence = evidence[evidence["CountCumul"] >= 50][
                "CountCumul"
            ].min()
            evidence = evidence[evidence["CountCumul"] <= min_required_evidence]

        # Ahora volvemos a agrupar por estado y recalculamos la media y varianza
        state_to = collections.defaultdict(lambda: dict(count=0, mean=0, std=0))

        if len(evidence) == 0:
            raise ValueError(f"No transitions for {from_state}, age={age}, sex={sex}.")

        for i, row in evidence.iterrows():
            d = state_to[row["to_state"]]
            d["count"] += row["count"]
            d["mean"] += row["count"] * row["duration_mean"]
            d["std"] += row["count"] * row["duration_std"]
            d["state"] = row["to_state"]

        for v in state_to.values():
            v["mean"] /= v["count"]
            v["std"] /= v["count"]

        # Finalmente tenemos un dataframe con forma
        # `count | mean | std | state`
        # con una entrada por cada estado
        # if not state_to:
        #     raise ValueError(f"No transitions for {from_state}, age={age}, sex={sex}.")

        return state_to