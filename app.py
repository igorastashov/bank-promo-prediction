import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt
from model import open_data, preprocess_data, split_data, load_model_and_predict


image = Image.open('data/image.png')

st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="Promo Bank",
    page_icon=image
)

image = Image.open('data/image.png')

st.title(
    """
    Предсказание отклика клиента на промо банка
    Определяем, кто из клиентов откликнется на предложение банка, а кто – нет.
    """
)

st.image(image)

"""
Таблица 1: Исследование
"""



def bar_chart(source: pd.DataFrame,
              feature: str,
              color: str,
              bin: alt.Bin = None,
              x_title: str = None,
              y_title: str = None) -> None:
    """
    Function draws and displays bar chart.

    :param source: data to display
    :param feature: feature to visualize
    :param color: bars color
    :param bin: binarize data params
    :param x_title: x label title to display
    :param y_title: y label title to display
    """
    chart = alt.Chart(source).transform_joinaggregate(
        total='sum(count)',
    ).transform_calculate(
        percent="datum.count / datum.total"
    ).mark_bar(color=color).encode(
        alt.X(feature, bin=bin, axis=alt.Axis(labelAngle=0, title=x_title)),
        alt.Y('sum(percent):Q', axis=alt.Axis(format='.0%', title=y_title))
    ).properties(height=250)

    st.altair_chart(chart, use_container_width=True)

def plot_age(df):
    st.subheader('Возраст')
    source = df.AGE.value_counts().reset_index()
    bar_chart(source, 'AGE:Q', color='#83c9ff', bin=alt.Bin(maxbins=10))

def plot_postal_adress(df):
    st.subheader('Почтовый адрес')
    source = df.POSTAL_ADDRESS_PROVINCE.value_counts().reset_index()
    bar_chart(source, 'POSTAL_ADDRESS_PROVINCE:N', color='#83c9ff', y_title='percent')

def plot_education(df):
    st.subheader('Образование')
    source = df.EDUCATION.value_counts().reset_index()
    bar_chart(source, 'EDUCATION:N', color='#83c9ff', y_title='percent')






"""
Таблица 2: Предсказание
"""

def input_features():
    age = st.slider("Возраст", min_value=1, max_value=80, value=30,
                            step=1)

    postal_address = st.selectbox("Почтовый адрес", (
        'Оренбургская область', 'Кабардино-Балкария', 'Иркутская область',
        'Ростовская область', 'Белгородская область',
        'Вологодская область', 'Волгоградская область',
        'Ярославская область', 'Краснодарский край', 'Карелия',
        'Хабаровский край', 'Калужская область', 'Еврейская АО',
        'Красноярский край', 'Свердловская область', 'Камчатская область',
        'Алтайский край', 'Липецкая область', 'Адыгея',
        'Челябинская область', 'Мурманская область', 'Амурская область',
        'Ленинградская область', 'Тверская область', 'Тульская область',
        'Воронежская область', 'Кемеровская область', 'Башкирия',
        'Пермская область', 'Омская область', 'Саратовская область',
        'Мордовская республика', 'Новосибирская область',
        'Санкт-Петербург', 'Магаданская область', 'Ханты-Мансийский АО',
        'Псковская область', 'Самарская область', 'Орловская область',
        'Ивановская область', 'Читинская область', 'Татарстан', 'Якутия',
        'Кировская область', 'Астраханская область', 'Коми',
        'Приморский край', 'Чувашия', 'Ставропольский край',
        'Костромская область', 'Курская область', 'Смоленская область',
        'Тюменская область', 'Хакасия', 'Брянская область',
        'Архангельская область', 'Московская область',
        'Пензенская область', 'Тамбовская область', 'Курганская область',
        'Владимирская область', 'Ульяновская область', 'Бурятия',
        'Томская область', 'Нижегородская область',
        'Калининградская область', 'Удмуртия', 'Рязанская область',
        'Карачаево-Черкесия', 'Сахалинская область', 'Горный Алтай',
        'Москва', 'Марийская республика', 'Новгородская область',
        'Ямало-Ненецкий АО', 'Агинский Бурятский АО',
        'Усть-Ордынский Бурятский АО', 'Эвенкийский АО', 'Северная Осетия',
        'Калмыкия'))

    education = st.selectbox("Образование", (
        'Среднее специальное', 'Среднее', 'Неполное среднее', 'Высшее',
        'Неоконченное высшее', 'Два и более высших образования', 'Ученая степень'))

    child_total = st.slider(
        "Количество детей",
        min_value=0, max_value=10, value=0, step=1)

    gen_industry = st.selectbox("Сфера деятельности (если клиент пенсионер то: 'not defined')", (
        'Торговля', 'Информационные технологии', 'Образование',
        'Государственная служба', 'Другие сферы', 'Сельское хозяйство',
        'Здравоохранение', 'Металлургия/Промышленность/Машиностроение',
        'Коммунальное хоз-во/Дорожные службы',
        'Строительство', 'Транспорт', 'Банк/Финансы',
        'Ресторанный бизнес/Общественное питание', 'Страхование',
        'Нефтегазовая промышленность', 'СМИ/Реклама/PR-агенства',
        'Энергетика', 'Салоны красоты и здоровья', 'ЧОП/Детективная д-ть',
        'Развлечения/Искусство', 'Наука', 'Химия/Парфюмерия/Фармацевтика',
        'Сборочные производства', 'Туризм',
        'Юридические услуги/нотариальные услуги', 'Маркетинг',
        'Подбор персонала', 'Информационные услуги', 'Недвижимость',
        'Управляющая компания', 'Логистика', 'not defined'))

    work_time = st.number_input("Время работы на текущем месте (в месяцах)", 12)

# st.divider()

    personal_income = st.number_input("Персональный доход", 40000)

    closed_fl = st.selectbox("Текущий статус кредита", ("Закрыт", "Не закрыт"))

    translatetion = {
        "Закрыт": 1,
        "Не закрыт": 0
    }
    data = {
        "AGE": age,
        "POSTAL_ADDRESS_PROVINCE": postal_address,
        "WORK_TIME": work_time,
        "EDUCATION": education,
        "GEN_INDUSTRY": gen_industry,
        "CHILD_TOTAL": child_total,
        "PERSONAL_INCOME": personal_income,
        "CLOSED_FL": translatetion[closed_fl]
    }

    df = pd.DataFrame(data, index=[0])

    return df


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)
    st.write("## Вероятность предсказания")
    st.write(prediction_probas)

def compute_tab1():
    st.write("""
            ### Введите данные клиента:
            """
             )


def write_user_data(df):
    st.write("## Введенные данные")
    st.write(df)

def write_prediction(prediction, prediction_probas):
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
    write_prediction(prediction, prediction_probas)


tab1, tab2, tab3 = st.tabs([
    "Исследовать", "Предсказать", "Оценить"
])


if __name__ == "__main__":
    with tab2:
        row1, row2 = st.columns([1, 3])
        with row1:
            compute_tab1()
            user_input_df = input_features()
        with row2:
            process_side_bar_inputs()

    with tab1:
        df = open_data()
        plot_age(df)
        plot_postal_adress(df)
        plot_education(df)