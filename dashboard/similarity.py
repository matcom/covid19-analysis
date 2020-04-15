import streamlit as st

from .data import raw_information


def country_similarity(source, country, stats_dict):
    source_stats = stats_dict[source]
    country_stats = stats_dict[country]

    similarity = []

    for key in ["Population", "Density", "Fertility", "Med. age", "Urban"]:
        baseline = source_stats[key]
        value = abs(country_stats[key] - baseline) / baseline
        if value == 0:
            return 1e50
        similarity.append(value)

    return sum(similarity) / len(similarity)


def most_similar_countries(country, count, stats_dict):
    all_similarities = {
        c: country_similarity(country, c, stats_dict) for c in stats_dict
    }
    return sorted(all_similarities, key=all_similarities.get)[:count]


def most_similar_curves(source, countries_to_analize, total, variable, rolling_smooth=1, step_size=1):
    raw = raw_information(rolling_smooth, step_size)

    countries_to_analize = [c for c in countries_to_analize if c in raw]

    def get_data(country):
        df = raw[country]
        return df[df[variable] > 0][variable].values

    source_data = get_data(source)

    st.sidebar.markdown("### Similaridad de curvas")

    exponent = st.sidebar.slider("Exponente", 1.0, 2.0, 2.0)
    normalize = st.sidebar.checkbox("Normalizar similaridad", False)
    window = st.sidebar.slider("Ventana de comparación", 0, 30, 7)
    tail = st.sidebar.slider("Tamaño de cola a analizar", 0, 30, 7)

    similarities = {
        country: sliding_similarity(
            source_data, get_data(country), exponent, normalize, window, tail
        )
        for country in countries_to_analize
    }

    similarities = {c: (k, v) for c, (k, v) in similarities.items() if v is not None}
    return sorted(similarities.items(), key=lambda t: t[1][0])[:total]


def similarity(source, country, exponent=1, normalize=True, tail=7):
    if len(country) < len(source):
        return 1e50

    min_len = min(len(source), len(country))
    cuba = source[min_len - tail : min_len]
    country = country[min_len - tail : min_len]

    def metric(vi, vj):
        t = abs(vi - vj)
        b = abs(vi) if normalize else 1
        return (t / b) ** exponent

    residuals = [metric(vi, vj) for vi, vj in zip(cuba, country)]
    msqe = sum(residuals) / len(residuals)

    return msqe


def sliding_similarity(
    source, country, exponent=1, normalize=True, window_size=15, tail=7
):
    min_sim = 1e50
    min_sample = None

    for i in range(window_size + 1):
        sample = country[i:]

        if len(sample) >= len(source):
            new_sim = similarity(source, sample, exponent, normalize, tail)

            if new_sim < min_sim:
                min_sim = new_sim
                min_sample = sample

    return min_sim, min_sample
