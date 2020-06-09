import streamlit as st
import pandas as pd
import json
import datetime

from pathlib import Path


@st.cache
def load_cuba_data():
    data = pd.read_csv(
        Path(__file__).parent.parent / "data/datos_privados_Cuba.csv"
    )#.fillna(None)

    data['# de contactos'] = data['# de contactos'].astype(int, errors='ignore')
    data['Asintomatico'] = data['FIS'].str.contains('sint')
    data.loc[data['Asintomatico'] == 1,'FIS'] = ""

    for col in ["Fecha Arribo", "FIS", "FI", "F. Conf", "Fecha Alta"]:
        data[col] = pd.to_datetime(
            data[col], format="%m/%d/%Y", errors="coerce", exact=True
        )
        data.loc[data[col] < datetime.datetime(2020,1,1), col] = None

    return data


@st.cache
def get_responses():
    responses = pd.read_csv(
        Path(__file__).parent.parent / "data/responses.tsv", sep="\t"
    ).fillna("")
    responses["Date"] = pd.to_datetime(
        responses["Date"], format="%d/%m/%Y", errors="coerce"
    )
    responses = responses[responses["Date"] > "2020-01-01"]
    return responses


@st.cache
def demographic_data(as_dict=True):
    df = pd.read_csv(
        Path(__file__).parent.parent / "data/world_demographics.tsv", sep="\t"
    ).set_index("Country")

    if as_dict:
        return df.to_dict("index")

    return df


@st.cache
def raw_information(rolling_window_size=1, step=1):
    with open(Path(__file__).parent.parent / "data" / "timeseries.json") as fp:
        raw_data = json.load(fp)

    data = {}
    for k, v in raw_data.items():
        df = pd.DataFrame(v)
        df = df[df["confirmed"] > 0]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["active"] = df["confirmed"] - df["recovered"] - df["deaths"]
        df = df.rolling(rolling_window_size).mean()
        df = df.reset_index()

        if step > 1:
            df = df.iloc[(len(df) - 1) % step :: step, :]

        data[k] = df

    return data


@st.cache
def weekly_information(window_size: int = 7):
    raw_dfs = raw_information()

    dfs = []
    for country, df in raw_dfs.items():
        df = df.copy()
        start_day = df["date"].values[0]
        df["period"] = df["date"].apply(lambda t: (t - start_day).days // window_size)
        df["period"] = df["period"] - df["period"].min()
        df["new"] = df["confirmed"].diff().fillna(0)
        df = (
            df.groupby("period")
            .agg(
                confirmed=("confirmed", "max"),
                new=("new", "sum"),
                date=("date", "first"),
            )
            .reset_index()
        )
        df["growth"] = df["new"].pct_change().fillna(1.0)
        df["country"] = country
        df = df[(df["confirmed"] > 100) & (df["new"] > 10)]
        df = df[:-1]
        dfs.append(df)

    return pd.concat(dfs).reset_index()


@st.cache
def get_measures_effects(responses: pd.DataFrame, data: pd.DataFrame, threshold):
    measures_effects = []

    for _, row in responses.iterrows():
        country = row["Country"]
        measure = row["Measure"]
        date = row["Date"]

        selected = data[
            (data["country"] == country)
            & (data["date"] >= date)
            # & (data['growth'] <- -threshold)
        ]

        # for i in np.arange(0, 1, 0.05):
        growth = selected[selected["growth"] <= -threshold]

        if len(growth) == 0:
            continue

        min_date = growth["date"].min()

        measures_effects.append(
            dict(
                country=country,
                measure=measure,
                category=row["Category"],
                taken=date,
                effect=min_date,
                distance=(min_date - date).days,
                size=threshold,
            )
        )

    return pd.DataFrame(measures_effects)


@st.cache
def testing_data():
    df = pd.read_csv(Path(__file__).parent.parent / "data/testing.csv")

    df["country"] = df["Entity"]  # .str.replace("-.*", "").str.strip()
    df["total"] = df["Cumulative total"].fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["Date"])

    df = df[df["total"] > 0].sort_values(["country", "date"])

    return df[["country", "total", "date"]]
