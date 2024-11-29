import pickle
import numpy as np

# Load the trained model and PowerTransformer
with open('D:\\EDA\\CaloriesBurnt_Predictor v1.0\\CaloriesBurnt_Predictor.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    transformer = data['transformer']

# Test data
test_data = [[0, 66, 171, 79, 11, 90, 40]]

transformed_test_data = transformer.transform([test_data[0][1:]])
print(transformed_test_data)
print(list(transformed_test_data[0]))
predicted_calories = model.predict([[test_data[0][0]] + list(transformed_test_data[0])])

# Display the result
print(f"Predicted Calories Burnt: {predicted_calories[0]:.2f}")
