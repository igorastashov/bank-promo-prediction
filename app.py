import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/image.png')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Promo Bank",
        page_icon=image,

    )

    st.write(
        """
        # Предсказание отклика клиента на промо банка
        Определяем, кто из клиентов откликнется на предложение банка, а кто – нет.
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    age = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=30,
                            step=1)
    postal_address = st.sidebar.selectbox("Почтовый адрес", (
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
    work_time = st.sidebar.number_input("Время работы на текущем месте (в месяцах)", 12)
    education = st.sidebar.selectbox("Образование", (
        'Среднее специальное', 'Среднее', 'Неполное среднее', 'Высшее',
        'Неоконченное высшее', 'Два и более высших образования', 'Ученая степень'))
    gen_industry = st.sidebar.selectbox("Класс (если вы пенсионер то: 'not defined')", (
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
    child_total = st.sidebar.slider(
        "Количество детей",
        min_value=0, max_value=10, value=0, step=1)
    personal_income = st.sidebar.number_input("Персональный доход", 40000)
    closed_fl = st.sidebar.selectbox("Текущий статус кредита", ("Закрыт", "Не закрыт"))

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


if __name__ == "__main__":
    process_main_page()
