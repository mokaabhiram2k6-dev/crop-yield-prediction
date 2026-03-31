from flask import Flask, render_template, request
import pandas as pd
from difflib import get_close_matches
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load dataset
df = pd.read_excel("data.xlsx")
df.columns = df.columns.str.strip().str.lower()

def clean_text(text):
    return str(text).lower().replace(" ", "").replace("_", "")

# Soil mapping
original_soils = df["soil_type"].unique()
cleaned_soils = [clean_text(s) for s in original_soils]

df["soil_type"] = df["soil_type"].apply(clean_text)

# Features
features = [
    "soil_type",
    "soil_moisture_%",
    "temperature_c",
    "rainfall_mm",
    "humidity_%",
    "sunlight_hours"
]

target = "yield_kg_per_hectare"

X = df[features].copy()
y = df[target]

# Encoding
le = LabelEncoder()
X["soil_type"] = le.fit_transform(X["soil_type"])

# Scaling
scaler = MinMaxScaler()
num_cols = [
    "soil_moisture_%",
    "temperature_c",
    "rainfall_mm",
    "humidity_%",
    "sunlight_hours"
]

X[num_cols] = scaler.fit_transform(X[num_cols])

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Price & Cost data
crop_prices = {
    "Rice": 20,
    "Wheat": 22,
    "Maize": 17,
    "Groundnut": 60,
    "Millets": 24,
    "Sugarcane": 20
}

crop_costs = {
    "Rice": 1900,
    "Wheat": 2500,
    "Maize": 1450,
    "Groundnut": 2500,
    "Millets": 2250,
    "Sugarcane": 50000
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    crops = []
    selected_crop = None
    price = None
    cost = None
    revenue = None
    profit = None
    comparison = None
    risk = None

    past_data = df.head(5)
    avg_yield = round(df["yield_kg_per_hectare"].mean(), 2)

    if request.method == "POST":
        try:
            soil_input = request.form["soil"]
            moisture = float(request.form["moisture"])
            temp = float(request.form["temp"])
            rainfall = float(request.form["rainfall"])
            humidity = float(request.form["humidity"])
            sunlight = float(request.form["sunlight"])

            cleaned_input = clean_text(soil_input)

            # Matching (safe)
            match = get_close_matches(cleaned_input, cleaned_soils, n=1, cutoff=0.5)
            if match:
                soil_type = match[0]
            else:
                soil_type = cleaned_input

            # Prepare input
            user_data = pd.DataFrame([{
                "soil_type": soil_type,
                "soil_moisture_%": moisture,
                "temperature_c": temp,
                "rainfall_mm": rainfall,
                "humidity_%": humidity,
                "sunlight_hours": sunlight
            }])

            user_data["soil_type"] = le.transform(user_data["soil_type"])
            user_data[num_cols] = scaler.transform(user_data[num_cols])

            prediction = round(model.predict(user_data)[0])

            # Crop suggestion
            if "red" in soil_type:
                crops = ["Groundnut", "Millets"]
            elif "clay" in soil_type:
                crops = ["Rice", "Sugarcane"]
            else:
                crops = ["Wheat", "Maize"]

            selected_crop = crops[0]

            # Price & cost
            price = crop_prices.get(selected_crop, 20)
            cost = crop_costs.get(selected_crop, 25000)

            # Calculations
            revenue = prediction * price
            profit = revenue - cost

            # Comparison
            if prediction > avg_yield:
                comparison = "Increase "
            else:
                comparison = "Decrease "

            # Risk
            if rainfall < 40 or temp > 35:
                risk = "High Risk"
            else:
                risk = "Safe"

        except:
            prediction = "Error"

    return render_template(
        "index.html",
        prediction=prediction,
        crops=crops,
        selected_crop=selected_crop,
        price=price,
        cost=cost,
        revenue=revenue,
        profit=profit,
        avg_yield=avg_yield,
        comparison=comparison,
        risk=risk,
        past_data=past_data.to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(debug=True)