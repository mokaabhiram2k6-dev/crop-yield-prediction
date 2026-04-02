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

# Soil cleaning
df["soil_type"] = df["soil_type"].apply(clean_text)

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

# Crop data
crop_prices = {
    "Rice": 20,
    "Wheat": 22,
    "Maize": 17,
    "Groundnut": 60,
    "Millets": 24,
    "Sugarcane": 20
}

crop_costs = {
    "Rice": 30000,
    "Wheat": 25000,
    "Maize": 22000,
    "Groundnut": 28000,
    "Millets": 20000,
    "Sugarcane": 35000
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_crop = None
    price = None
    cost = None
    revenue = None
    profit = None
    comparison = None
    risk = None

    soil = ""
    moisture = ""
    temp = ""
    rainfall = ""
    humidity = ""
    sunlight = ""

    avg_yield = round(df[target].mean(), 2)

    if request.method == "POST":
        try:
            # CSV upload
            if "file" in request.files and request.files["file"].filename != "":
                file = request.files["file"]
                df_csv = pd.read_csv(file)
                latest = df_csv.iloc[-1]

                soil = clean_text(latest["soil_type"])
                moisture = float(latest["soil_moisture_%"])
                temp = float(latest["temperature_c"])
                rainfall = float(latest["rainfall_mm"])
                humidity = float(latest["humidity_%"])
                sunlight = float(latest["sunlight_hours"])

            else:
                soil = clean_text(request.form["soil"])
                moisture = float(request.form["moisture"])
                temp = float(request.form["temp"])
                rainfall = float(request.form["rainfall"])
                humidity = float(request.form["humidity"])
                sunlight = float(request.form["sunlight"])

            # Model input
            user_data = pd.DataFrame([{
                "soil_type": soil,
                "soil_moisture_%": moisture,
                "temperature_c": temp,
                "rainfall_mm": rainfall,
                "humidity_%": humidity,
                "sunlight_hours": sunlight
            }])

            user_data["soil_type"] = le.transform(user_data["soil_type"])
            user_data[num_cols] = scaler.transform(user_data[num_cols])

            prediction = round(model.predict(user_data)[0], 2)

            # Crop suggestion
            if "red" in soil:
                selected_crop = "Groundnut"
            elif "clay" in soil:
                selected_crop = "Rice"
            else:
                selected_crop = "Wheat"

            price = crop_prices[selected_crop]
            cost = crop_costs[selected_crop]

            revenue = prediction * price
            profit = revenue - cost

            # Comparison
            comparison = "Increase 📈" if prediction > avg_yield else "Decrease 📉"

            # Risk
            risk = "High Risk ⚠️" if rainfall < 40 or temp > 35 else "Safe ✅"

        except Exception as e:
            prediction = "Error"

    return render_template(
        "index.html",
        prediction=prediction,
        selected_crop=selected_crop,
        price=price,
        cost=cost,
        revenue=revenue,
        profit=profit,
        avg_yield=avg_yield,
        comparison=comparison,
        risk=risk,
        soil=soil,
        moisture=moisture,
        temp=temp,
        rainfall=rainfall,
        humidity=humidity,
        sunlight=sunlight
    )

if __name__ == "__main__":
    app.run(debug=True)
