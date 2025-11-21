from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# -----------------------------
# Load model, scaler, encoder
# -----------------------------
with open("augmented_more_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("augmented_more_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("augmented_more_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -----------------------------
# HTML Template
# -----------------------------
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Plant Health Predictor</title>
    <style>
        body { font-family: Arial; background-color: #e8f5e8; padding: 20px; }
        .container { max-width: 700px; margin: auto; background: #fff; padding: 20px; border-radius: 10px; }
        h1 { text-align: center; color: #2e7d32; }
        form { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        label { font-weight: bold; }
        input, select, button { padding: 8px; font-size: 14px; border-radius: 5px; }
        button { grid-column: span 2; background: #4caf50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        .result { margin-top: 20px; padding: 15px; background: #f1f8e9; border-radius: 5px; text-align: center; }
        .healthy { color: #2e7d32; }
        .moderate { color: #ff9800; }
        .unhealthy { color: #f44336; }
    </style>
</head>
<body>
<div class="container">
    <h1>Indoor Plant Health Predictor</h1>
    <form method="POST">
        <!-- Inputs (same as before) -->
        <label>Plant Name:</label>
        <select name="plant_name" required>
            <option value="Aloe Vera">Aloe Vera</option>
            <option value="Spider Plant">Spider Plant</option>
            <option value="Peace Lily">Peace Lily</option>
            <option value="Snake Plant">Snake Plant</option>
        </select>
        <label>Height (cm):</label><input type="number" step="0.01" name="height" required>
        <label>Leaf Count:</label><input type="number" name="leaf_count" required>
        <label>New Growth Count:</label><input type="number" name="new_growth" required>
        <label>Watering Amount (ml):</label><input type="number" step="0.01" name="watering_amount" required>
        <label>Watering Frequency (days):</label><input type="number" name="watering_freq" required>
        <label>Room Temperature (°C):</label><input type="number" step="0.01" name="temp" required>
        <label>Humidity (%):</label><input type="number" step="0.01" name="humidity" required>
        <label>Fertilizer Amount (ml):</label><input type="number" step="0.01" name="fert_amount" required>
        <label>Soil Moisture (%):</label><input type="number" step="0.01" name="soil_moisture" required>
        <label>Sunlight Exposure:</label>
        <select name="sunlight" required>
            <option value="6h full sun">6h full sun</option>
            <option value="Filtered sunlight through curtain">Filtered sunlight through curtain</option>
            <option value="Indirect light all day">Indirect light all day</option>
            <option value="Low light corner">Low light corner</option>
        </select>
        <label>Fertilizer Type:</label>
        <select name="fertilizer" required>
            <option value="Compost">Compost</option>
            <option value="Liquid feed">Liquid feed</option>
            <option value="Organic">Organic</option>
        </select>
        <label>Pest Presence:</label>
        <select name="pest_presence" required>
            <option value="None">None</option>
            <option value="Fungus gnats">Fungus gnats</option>
            <option value="Spider mites">Spider mites</option>
            <option value="Whiteflies">Whiteflies</option>
        </select>
        <label>Pest Severity:</label>
        <select name="pest_severity" required>
            <option value="None">None</option>
            <option value="Low">Low</option>
            <option value="Moderate">Moderate</option>
        </select>
        <label>Soil Type:</label>
        <select name="soil_type" required>
            <option value="Clay">Clay</option>
            <option value="Loamy">Loamy</option>
            <option value="Peaty">Peaty</option>
            <option value="Sandy">Sandy</option>
            <option value="Silty">Silty</option>
        </select>
        <button type="submit">Predict Health Score</button>
    </form>

    {% if result %}
    <div class="result">
        <h2>Prediction Result</h2>
        <p>Plant: {{ result.plant_name }}</p>
        <p>Predicted Health Score: <strong>{{ result.score }}</strong></p>
        <p class="{{ result.class }}">Classification: {{ result.classification }}</p>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

# -----------------------------
# Flask Route
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        # Collect numeric data
        numeric_data = {
            "Height_cm": float(request.form['height']),
            "Leaf_Count": float(request.form['leaf_count']),
            "New_Growth_Count": float(request.form['new_growth']),
            "Watering_Amount_ml": float(request.form['watering_amount']),
            "Watering_Frequency_days": float(request.form['watering_freq']),
            "Room_Temperature_C": float(request.form['temp']),
            "Humidity_%": float(request.form['humidity']),
            "Fertilizer_Amount_ml": float(request.form['fert_amount']),
            "Soil_Moisture_%": float(request.form['soil_moisture'])
        }

        # Scale numeric
        numeric_df = pd.DataFrame([numeric_data])
        numeric_df = pd.DataFrame(scaler.transform(numeric_df), columns=numeric_df.columns)

        # Collect categorical data
        categorical_data = {
            "Sunlight_Exposure": request.form['sunlight'],
            "Fertilizer_Type": request.form['fertilizer'],
            "Pest_Presence": request.form['pest_presence'],
            "Pest_Severity": request.form['pest_severity'],
            "Soil_Type": request.form['soil_type']
        }
        cat_df = pd.DataFrame([categorical_data])

        # Replace unknown 'None' categories with placeholder that exists in encoder
        for col in cat_df.columns:
            known_categories = encoder.categories_[list(cat_df.columns).index(col)]
            if cat_df[col][0] not in known_categories:
                cat_df[col][0] = known_categories[0]  # fallback to first category

        # Encode categorical
        cat_encoded = encoder.transform(cat_df)
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(), index=[0])

        # Combine numeric + categorical
        new_df = pd.concat([numeric_df, cat_encoded_df], axis=1)

        # Predict
        pred_score = model.predict(new_df)[0]
        score = round(pred_score, 3)

        # Classification
        if pred_score >= 7:
            classification = "Healthy"
            css_class = "healthy"
        elif pred_score >= 4:
            classification = "Moderate"
            css_class = "moderate"
        else:
            classification = "Unhealthy"
            css_class = "unhealthy"

        result = {
            'plant_name': request.form['plant_name'],
            'score': score,
            'classification': classification,
            'class': css_class
        }

    return render_template_string(html_template, result=result)


if __name__ == '__main__':
    app.run(debug=True)
