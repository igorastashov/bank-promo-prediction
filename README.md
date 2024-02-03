##  Предсказание отклика клиента на промо банка


Astashov I.V., 2024.

Репозиторий содержит проект с первичной предобработкой полученных данных, их разведочным анализом
и обученной моделью, предсказывающую отклик клиента на промо банка. 

**Презентация ML - решения представлена в виде веб-приложения на платформе [Streamlit](https://bank-promo-prediction-777.streamlit.app/).**

Проект выполнен в рамках курса «Прикладной Python» магистерской программы НИУ ВШЭ 
[«Машинное обучение и высоконагруженные системы»](https://www.hse.ru/ma/mlds/).

## (1) Файлы

- app.py: файл приложения streamlit
- models/model.py: скрипт для обучения модели классификатора CatBoost 
- data/df_full.csv и trained_model.cbm: предобработанные данные и предобученная модель  
- requirements.txt: файл зависимостей

## (2) Запуск локально

### Shell

Для прямого запуска streamlit локально:

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ streamlit run app.py
Открыть http://localhost:8501 для просмотра приложения.
```

## (3) Данные

Предварительно собранные данные о клиентах `data/*.csv`:

- `clients.csv`: демография, образование, социальный статус, семья и активы;
- `close_loan.csv`: статус полученных кредитов;
- `job.csv`: информация о работе;
- `last_credit.csv`: информация о последнем кредите; 
- `loan.csv`: кредитная история;  
- `pens.csv`: статусы относительно пенсии;
- `salary.csv`: информация о доходах;
- `target.csv`: статус отклика на промо банка (целевой признак);
- `work.csv`: статусы относительно работы.

## (4) Разведочный анализ данных

В [файле](https://github.com/igorastashov/bank-promo-prediction/blob/master/notebooks/bank_promo_eda.ipynb), данные были отчищены от дубликатов, пропусков и аномальных 
значений. Агрегированны по полученным и погашенным кредитам и объединены в одну таблицу.
Проведен анализ, в следствии которого, был получен ответ на вопрос о том,
какие факторы влияют на отклик клиентов на промо.

## (A) Благодарности

Дизайн репозитория частично заимстован: [ссылка-1](https://rateyourflight.streamlit.app/), 
[ссылка-2](https://github.com/evgpat/streamlit_demo).
Используемые материалы: [Stepik](https://stepik.org/lesson/1009061/step/6?unit=1016868),
[Лекции HSE](https://www.youtube.com/watch?v=J0EnFcdGW1I&list=PLmA-1xX7IuzADGz3hSgPPm6ib11Z0HSML&index=4).
