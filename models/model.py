from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load
import pandas as pd

import _pickle as cPickle


def split_data(df: pd.DataFrame):
    y = df['TARGET']
    X = df[["AGE", "POSTAL_ADDRESS_PROVINCE", "WORK_TIME", "EDUCATION",
            "GEN_INDUSTRY", "CHILD_TOTAL", "PERSONAL_INCOME", "CLOSED_FL"]]

    return X, y


def open_data(path="data/df_full.csv"):
    df = pd.read_csv(path)
    df = df[["AGE", "POSTAL_ADDRESS_PROVINCE", "WORK_TIME", "EDUCATION",
             "GEN_INDUSTRY", "CHILD_TOTAL", "PERSONAL_INCOME", "CLOSED_FL",
             "TARGET"]]

    return df


def preprocess_data(df: pd.DataFrame, test=True):
    df.dropna(inplace=True)

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    if test:
        return X_df, y_df
    else:
        return X_df


def fit_and_save_model(X_df, y_df, path="data/trained_model.cbm"):
    categorical = [var for var in X_df.columns if X_df[var].dtype == 'O']

    model = CatBoostClassifier(cat_features=categorical, n_estimators = 150)
    model.fit(X_df, y_df,
              verbose=150, plot=False
              )

    test_prediction = model.predict(X_df).round()
    accuracy = accuracy_score(y_df, test_prediction)
    print(f"Model accuracy is {accuracy}")

    # Функция для сохранения модели

    with open(path, "wb") as file:
        cPickle.dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/trained_model.cbm"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0].round()

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Отсутствие отклика с вероятностью",
        1: "Отклик на предложение с вероятностью"
    }

    encode_prediction = {
        0: "К сожалению, от клиента не ожидается положительный отклик",
        1: "Ура! От клиента ожидается положительный отклик"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    # fit_and_save_model(X_df, y_df)
    # load_model_and_predict(df)
