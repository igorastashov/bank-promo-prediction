import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from models.model import (load_model_and_predict, open_data, preprocess_data,
                          split_data)
from utils.utils import bar_chart, phik_data

image = Image.open("data/image.png")

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Promo Bank",
    page_icon=image,
)

st.title(
    """
    Предсказание отклика клиента на промо банка
    Определяем, кто из клиентов откликнется на предложение банка, а кто – нет.
    """
)

st.image(image)


def plot_age(df: pd.DataFrame):
    st.subheader("Возраст")
    source = df.AGE.value_counts().reset_index()
    bar_chart(source, "AGE:Q", color="#83c9ff", bin=alt.Bin(maxbins=10), x_title="лет")


def plot_postal_address(df: pd.DataFrame):
    st.subheader("Почтовый адрес")
    source = df.POSTAL_ADDRESS_PROVINCE.value_counts().reset_index()
    bar_chart(source, "POSTAL_ADDRESS_PROVINCE:N", color="#83c9ff", x_title="адрес")


def plot_education(df: pd.DataFrame):
    st.subheader("Образование")
    source = df.EDUCATION.value_counts().reset_index()
    bar_chart(source, "EDUCATION:N", color="#83c9ff", x_title="образование")


def top_gen_industry(df: pd.DataFrame):
    st.subheader("Отрасль работы")
    source = df.GEN_INDUSTRY.value_counts().reset_index()
    bar_chart(source, "GEN_INDUSTRY:N", color="#83c9ff", x_title="отрасль")


def personal_income(df: pd.DataFrame):
    st.subheader("Персональный доход")
    source = df.PERSONAL_INCOME.value_counts().reset_index()
    bar_chart(
        source,
        "PERSONAL_INCOME:Q",
        color="#83c9ff",
        bin=alt.Bin(maxbins=25),
        x_title="руб.",
    )


def plot_phik_matrix():
    st.subheader("Корреляционная матрица признаков")
    source = phik_data(df)
    plot = (
        alt.Chart(source)
        .mark_rect(strokeOpacity=0)
        .encode(
            x=alt.X(
                "variable:O", axis=alt.Axis(grid=False, title=None, labelLimit=360)
            ),
            y=alt.Y(
                "variable2:O", axis=alt.Axis(grid=False, title=None, labelLimit=360)
            ),
            color=alt.Color("correlation:Q", scale=alt.Scale(scheme="blues")),
        )
        .properties(height=760)
    )
    text = plot.mark_text(fontSize=16).encode(
        text="correlation_label",
        color=alt.condition(
            ((alt.datum.correlation > 0.75) | (alt.datum.correlation < 0.25)),
            alt.value("white"),
            alt.value("black"),
        ),
    )
    st.altair_chart(plot + text, use_container_width=True)


def input_features():
    age = st.slider("Возраст", min_value=1, max_value=80, value=30, step=1)

    postal_address = st.selectbox(
        "Почтовый адрес",
        (
            "Оренбургская область",
            "Кабардино-Балкария",
            "Иркутская область",
            "Ростовская область",
            "Белгородская область",
            "Вологодская область",
            "Волгоградская область",
            "Ярославская область",
            "Краснодарский край",
            "Карелия",
            "Хабаровский край",
            "Калужская область",
            "Еврейская АО",
            "Красноярский край",
            "Свердловская область",
            "Камчатская область",
            "Алтайский край",
            "Липецкая область",
            "Адыгея",
            "Челябинская область",
            "Мурманская область",
            "Амурская область",
            "Ленинградская область",
            "Тверская область",
            "Тульская область",
            "Воронежская область",
            "Кемеровская область",
            "Башкирия",
            "Пермская область",
            "Омская область",
            "Саратовская область",
            "Мордовская республика",
            "Новосибирская область",
            "Санкт-Петербург",
            "Магаданская область",
            "Ханты-Мансийский АО",
            "Псковская область",
            "Самарская область",
            "Орловская область",
            "Ивановская область",
            "Читинская область",
            "Татарстан",
            "Якутия",
            "Кировская область",
            "Астраханская область",
            "Коми",
            "Приморский край",
            "Чувашия",
            "Ставропольский край",
            "Костромская область",
            "Курская область",
            "Смоленская область",
            "Тюменская область",
            "Хакасия",
            "Брянская область",
            "Архангельская область",
            "Московская область",
            "Пензенская область",
            "Тамбовская область",
            "Курганская область",
            "Владимирская область",
            "Ульяновская область",
            "Бурятия",
            "Томская область",
            "Нижегородская область",
            "Калининградская область",
            "Удмуртия",
            "Рязанская область",
            "Карачаево-Черкесия",
            "Сахалинская область",
            "Горный Алтай",
            "Москва",
            "Марийская республика",
            "Новгородская область",
            "Ямало-Ненецкий АО",
            "Агинский Бурятский АО",
            "Усть-Ордынский Бурятский АО",
            "Эвенкийский АО",
            "Северная Осетия",
            "Калмыкия",
        ),
    )

    education = st.selectbox(
        "Образование",
        (
            "Среднее специальное",
            "Среднее",
            "Неполное среднее",
            "Высшее",
            "Неоконченное высшее",
            "Два и более высших образования",
            "Ученая степень",
        ),
    )

    child_total = st.slider(
        "Количество детей", min_value=0, max_value=10, value=0, step=1
    )

    gen_industry = st.selectbox(
        "Сфера деятельности (если клиент пенсионер то: 'not defined')",
        (
            "Торговля",
            "Информационные технологии",
            "Образование",
            "Государственная служба",
            "Другие сферы",
            "Сельское хозяйство",
            "Здравоохранение",
            "Металлургия/Промышленность/Машиностроение",
            "Коммунальное хоз-во/Дорожные службы",
            "Строительство",
            "Транспорт",
            "Банк/Финансы",
            "Ресторанный бизнес/Общественное питание",
            "Страхование",
            "Нефтегазовая промышленность",
            "СМИ/Реклама/PR-агенства",
            "Энергетика",
            "Салоны красоты и здоровья",
            "ЧОП/Детективная д-ть",
            "Развлечения/Искусство",
            "Наука",
            "Химия/Парфюмерия/Фармацевтика",
            "Сборочные производства",
            "Туризм",
            "Юридические услуги/нотариальные услуги",
            "Маркетинг",
            "Подбор персонала",
            "Информационные услуги",
            "Недвижимость",
            "Управляющая компания",
            "Логистика",
            "not defined",
        ),
    )

    work_time = st.number_input("Время работы на текущем месте (в месяцах)", 12)

    personal_income = st.number_input("Персональный доход", 40000)

    closed_fl = st.selectbox("Текущий статус кредита", ("Закрыт", "Не закрыт"))

    translatetion = {"Закрыт": 1, "Не закрыт": 0}
    data = {
        "AGE": age,
        "POSTAL_ADDRESS_PROVINCE": postal_address,
        "WORK_TIME": work_time,
        "EDUCATION": education,
        "GEN_INDUSTRY": gen_industry,
        "CHILD_TOTAL": child_total,
        "PERSONAL_INCOME": personal_income,
        "CLOSED_FL": translatetion[closed_fl],
    }

    df = pd.DataFrame(data, index=[0])

    return df


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)
    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def compute_tab1():
    st.write(
        """
            ### Введите данные клиента:
            """
    )


def write_user_data(df: pd.DataFrame):
    st.write("## Введенные данные")
    st.write(df)


def write_pred(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_pred(prediction, prediction_probas)


tab1, tab2 = st.tabs(["Исследовать", "Предсказать"])


if __name__ == "__main__":
    with tab1:
        row1, row2 = st.columns([1, 1])
        with row1:
            df = open_data()
            plot_age(df)
            st.divider()
            plot_postal_address(df)
            st.divider()
            plot_education(df)
            st.divider()
            top_gen_industry(df)
            st.divider()
            personal_income(df)
        with row2:
            plot_phik_matrix()

    with tab2:
        row1, row2 = st.columns([1, 3])
        with row1:
            compute_tab1()
            user_input_df = input_features()
        with row2:
            process_side_bar_inputs()
