import streamlit as st
import pandas as pd
import numpy as np

from ..data import load_cuba_data


def run(tr):
    data = load_cuba_data().copy()

    def compute_days(d1, d2):
        try:
            return (pd.to_datetime(d2) - pd.to_datetime(d1)).days
        except:
            return np.nan

    # data['T(FI-FIS)'] = data.apply(lambda row: compute_days(row['FIS'], row['FI']), axis=1)
    # data['T(FConf-FI)'] = data.apply(lambda row: compute_days(row['FI'], row['F. Conf']), axis=1)
    # data['T(Alta-FI)'] = data.apply(lambda row: compute_days(row['FI'], row['Fecha Alta']), axis=1)
    # data['T(Alta-FConf)'] = data.apply(lambda row: compute_days(row['F. Conf'], row['Fecha Alta']), axis=1)

    st.write(data)

    events = []

    for i, d in data.iterrows():
        person_id = d["Cons"]

        try:
            age = int(d["Edad"])
            sex = "FEMALE" if d["Sexo"] == "F" else "MALE"
        except ValueError:
            continue

        state = "L"

        if d["Asintomatico"]:
            events.append(
                dict(
                    from_state="L",
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

        events.append(
            dict(
                from_state="L",
                to_state="I",
                duration=0,
                age=age,
                sex=sex,
                id=person_id,
            )
        )

        events.append(
            dict(
                from_state="I",
                to_state="Is",
                duration=0,
                age=age,
                sex=sex,
                id=person_id,
            )
        )

        try:
            symptoms_start = pd.to_datetime(d["FIS"], format="%m/%d/%Y", errors="raise")
            admission_start = pd.to_datetime(d["FI"], format="%m/%d/%Y", errors="raise")
        except:
            continue

        events.append(
            dict(
                from_state="Is",
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

    df = pd.DataFrame(events)

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

    st.write("### Transiciones por edad y género")

    for age1, age2 in [(0, 18), (18, 30), (30, 50), (50, 80), (80, 120)]:
        for sex in ["FEMALE", "MALE"]:
            df_filter = df[
                (df["age"] >= age1) & (df["age"] < age2) & (df["sex"] == sex)
            ]
            states = compute_transitions(df_filter)

            st.write(f"#### {sex} {age1}-{age2} años")
            st.write(states)

    if st.checkbox("Ver en CSV"):
        csv = []

        for age in set(df["age"]):
            for sex in ["FEMALE", "MALE"]:
                df_filter = df[(df["age"] == age) & (df["sex"] == sex)]

                if len(df_filter) > 0:
                    states = compute_transitions(df_filter)
                    states["age"] = age
                    states["sex"] = sex
                    csv.append(states.set_index(["age", "sex"]).round(3))

        st.code(pd.concat(csv).to_csv())
