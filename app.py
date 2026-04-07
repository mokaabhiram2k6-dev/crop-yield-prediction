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

# Clean soil
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

# Encode
le = LabelEncoder()
X["soil_type"] = le.fit_transform(X["soil_type"])

# Scale
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

# Prices & Costs
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
    avg_yield = round(df[target].mean(), 2)

    if request.method == "POST":
        try:
            file = request.files.get("file")

            if file and file.filename.endswith(".csv"):
                df_csv = pd.read_csv(file)
                latest = df_csv.iloc[-1]

                soil_input = latest["soil_type"]
                moisture = float(latest["soil_moisture_%"])
                temp = float(latest["temperature_c"])
                rainfall = float(latest["rainfall_mm"])
                humidity = float(latest["humidity_%"])
                sunlight = float(latest["sunlight_hours"])

            else:
                soil_input = request.form.get("soil", "")
                moisture = float(request.form.get("moisture", 0))
                temp = float(request.form.get("temp", 0))
                rainfall = float(request.form.get("rainfall", 0))
                humidity = float(request.form.get("humidity", 0))
                sunlight = float(request.form.get("sunlight", 0))

            cleaned_input = clean_text(soil_input)
            match = get_close_matches(cleaned_input, cleaned_soils, n=1, cutoff=0.5)
            soil_type = match[0] if match else cleaned_input

            user_data = pd.DataFrame([{
                "soil_type": soil_type,
                "soil_moisture_%": moisture,
                "temperature_c": temp,
                "rainfall_mm": rainfall,
                "humidity_%": humidity,
                "sunlight_hours": sunlight
            }])

            # Safe encoding
            try:
                user_data["soil_type"] = le.transform(user_data["soil_type"])
            except:
                user_data["soil_type"] = 0

            # Safe scaling
            user_data[num_cols] = scaler.transform(user_data[num_cols])

            prediction = round(model.predict(user_data)[0], 2)

            # Crop logic
            if "red" in soil_type:
                selected_crop = "Groundnut"
            elif "clay" in soil_type:
                selected_crop = "Rice"
            else:
                selected_crop = "Wheat"

            price = crop_prices.get(selected_crop, 20)
            cost = crop_costs.get(selected_crop, 25000)

            revenue = prediction * price
            profit = revenue - cost

        except Exception as e:
            print("ERROR:", e)
            prediction = "Error"

    return render_template(
        "index.html",
        prediction=prediction,
        selected_crop=selected_crop,
        price=price,
        cost=cost,
        revenue=revenue,
        profit=profit,
        avg_yield=avg_yield
    )

if __name__ == "__main__":
    app.run(debug=True)
