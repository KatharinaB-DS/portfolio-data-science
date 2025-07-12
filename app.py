#Ten plik zawiera główną implementację backendu API przy użyciu frameworka Flask.
#Funkcjonalność:
#- Ładuje wytrenowany model `xgb_best_model.pkl` oraz listę wymaganych cech `feature_columns.pkl`
#- Definiuje dwa endpointy:
  
  #1. `/predict` (POST)
     #- Odbiera dane wejściowe w formacie JSON
     #- Przekształca dane na DataFrame, stosuje One-Hot Encoding i uzupełnia brakujące kolumny
     #- Zwraca przewidywaną wartość `log_price` w formacie JSON

 # 2. `/predict_form` (GET + POST)
     #- GET: Wyświetla formularz HTML do ręcznego wprowadzenia danych
     #- POST: Pobiera dane z formularza, przekształca je i wyświetla wynik w formie HTML wraz z przeliczoną prognozowaną ceną

#Cel to udostępnienie interfejsu do przewidywania cen mieszkań na podstawie danych wejściowych – zarówno przez API, jak i prosty formularz webowy.

from flask import render_template
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

model = joblib.load("xgb_best_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    except Exception as e:
        return jsonify({"error": str(e)})

    prediction = model.predict(input_df)[0]
    return jsonify({
        "predicted_LOG_PRICE": float(prediction)
    })


@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    if request.method == 'GET':
        return render_template('form.html')

    data = request.form.to_dict()

    for key in data:
        try:
            data[key] = float(data[key])
        except:
            pass

    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    cena = round(np.exp(prediction), 2)

    return f"""
    <h3>Logarytmiczna prognoza ceny (LOG_PRICE): {prediction:.4f}</h3>
    <h2>Prognozowana cena nieruchomości: ${cena:,.2f}</h2>
    <a href="/predict_form">← Wróć do formularza</a>
    """

if __name__ == '__main__':
    app.run(debug=True)