import altair as alt
import pandas as pd
import phik
import streamlit as st


def bar_chart(
    source: pd.DataFrame,
    feature: str,
    color: str,
    bin: alt.Bin = None,
    x_title: str = None,
    y_title: str = None,
) -> None:
    """
    Функция для отображения столбчатых диаграмм

    :param source: датафрэйм
    :param feature: признаки для визуализации
    :param color: цвет столбцов
    :param bin: бинаризация кол-й переменной
    :param x_title: обозначение оси x
    :param y_title: обозначение оси y
    """
    chart = (
        alt.Chart(source)
        .transform_joinaggregate(
            total="sum(count)",
        )
        .transform_calculate(percent="datum.count / datum.total")
        .mark_bar(color=color)
        .encode(
            alt.X(feature, bin=bin, axis=alt.Axis(labelAngle=0, title=x_title)),
            alt.Y("sum(percent):Q", axis=alt.Axis(format=".0%", title=y_title)),
        )
        .properties(height=250)
    )

    st.altair_chart(chart, use_container_width=True)


@st.cache_data
def phik_data(df: pd.DataFrame):
    """
    Функция возвращает Фик корреляционную матрицу
    :param df: датафрэйм
    """
    data_types = {
        "TARGET": "categorical",
        "AGE": "interval",
        "POSTAL_ADDRESS_PROVINCE": "categorical",
        "WORK_TIME": "interval",
        "EDUCATION": "categorical",
        "GEN_INDUSTRY": "categorical",
        "CHILD_TOTAL": "ordinal",
        "PERSONAL_INCOME": "interval",
        "CLOSED_FL": "categorical",
    }

    interval_cols = [col for col, v in data_types.items() if v == "interval"]
    corr_matrix = (
        df[list(data_types.keys())]
        .phik_matrix(interval_cols=interval_cols, bins=20)
        .stack()
        .reset_index()
        .rename(
            columns={0: "correlation", "level_0": "variable", "level_1": "variable2"}
        )
    )
    corr_matrix["correlation_label"] = corr_matrix["correlation"].map("{:.1f}".format)

    return corr_matrix
