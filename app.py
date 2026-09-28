from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load and preprocess the data
df = pd.read_csv('insurance.csv')

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Separate features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Print training features (for debug)
print("Training features:", X.columns.tolist())
print("Shape of training X:", X.shape)

# Train the model
model = RandomForestRegressor()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        age = int(request.form['age'])
        sex = int(request.form['Gender'])        # 0=male, 1=female
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])     # 0=no, 1=yes
        region = int(request.form['region'])     # 0=NW, 1=NE, 2=SE, 3=SW

        # Convert to one-hot encoded format
        sex_male = 1 if sex == 0 else 0
        smoker_yes = 1 if smoker == 1 else 0
        region_northeast = 1 if region == 1 else 0
        region_northwest = 1 if region == 0 else 0
        region_southeast = 1 if region == 2 else 0
        region_southwest = 1 if region == 3 else 0

        # Construct the input dictionary
        input_dict = {
            'age': age,
            'bmi': bmi,
            'children': children,
            'sex_male': sex_male,
            'smoker_yes': smoker_yes,
            'region_northeast': region_northeast,
            'region_northwest': region_northwest,
            'region_southeast': region_southeast,
            'region_southwest': region_southwest
        }

        # Convert to DataFrame and align columns with training data
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)[0]

        return render_template('op.html', pred=f"Expected amount is ₹{prediction:.2f}")
    except Exception as e:
        return render_template('op.html', pred=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
