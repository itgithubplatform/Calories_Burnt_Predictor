from flask import Flask, request, render_template
from flask_cors import cross_origin
from sklearn.preprocessing import PowerTransformer
import pickle

app = Flask(__name__)

with open('D:\\EDA\\CaloriesBurnt_Predictor v1.0\\CaloriesBurnt_Predictor.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    transformer = data['transformer']


@app.route("/")
@cross_origin()
def home():
    return render_template("WebApp.html")


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":            
        gender = request.form['gender']
        if gender == 'Male':
            gender = 0
        else:
            gender = 1
            
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])
        
        transformed_data = transformer.transform([[age, height, weight, duration, heart_rate, body_temp]])
        
        prediction = model.predict([[gender] + list(transformed_data[0])])
        output = prediction[0]

        return render_template('WebApp.html', prediction_text=f'{output:.3f} Kilocalories (kcal)')

    return render_template("WebApp.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)