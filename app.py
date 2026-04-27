from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# =========================
# LOAD BOTH FILES
# =========================
df1 = pd.read_excel("data1.xlsx", engine="openpyxl")
df2 = pd.read_excel("data2.xlsx", engine="openpyxl")

# COMBINE
df = pd.concat([df1, df2], ignore_index=True)

# CLEAN COLUMN NAMES
df.columns = df.columns.str.strip().str.lower()

# =========================
# FEATURES
# =========================
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

# =========================
# ENCODING
# =========================
le = LabelEncoder()
X["soil_type"] = le.fit_transform(X["soil_type"])

# =========================
# SCALING
# =========================
scaler = MinMaxScaler()

num_cols = [
    "soil_moisture_%",
    "temperature_c",
    "rainfall_mm",
    "humidity_%",
    "sunlight_hours"
]

X[num_cols] = scaler.fit_transform(X[num_cols])

# =========================
# MODEL
# =========================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# PRICE & COST
# =========================
crop_prices = {
    "Rice": 20,
    "Wheat": 22,
    "Maize": 17
}

crop_costs = {
    "Rice": 30000,
    "Wheat": 25000,
    "Maize": 22000
}

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_crop = None
    price = None
    cost = None
    revenue = None
    profit = None

    if request.method == "POST":
        try:
            soil = request.form["soil"].lower()
            moisture = float(request.form["moisture"])
            temp = float(request.form["temp"])
            rainfall = float(request.form["rainfall"])
            humidity = float(request.form["humidity"])
            sunlight = float(request.form["sunlight"])

            # SAFE ENCODE
            if soil in le.classes_:
                soil_encoded = le.transform([soil])[0]
            else:
                soil_encoded = 0

            user_data = pd.DataFrame([{
                "soil_type": soil_encoded,
                "soil_moisture_%": moisture,
                "temperature_c": temp,
                "rainfall_mm": rainfall,
                "humidity_%": humidity,
                "sunlight_hours": sunlight
            }])

            user_data[num_cols] = scaler.transform(user_data[num_cols])

            prediction = round(model.predict(user_data)[0])

            # CROP LOGIC
            if moisture > 50:
                selected_crop = "Rice"
            elif temp > 30:
                selected_crop = "Maize"
            else:
                selected_crop = "Wheat"

            price = crop_prices[selected_crop]
            cost = crop_costs[selected_crop]

            revenue = prediction * price
            profit = revenue - cost

        except Exception as e:
            print(e)
            prediction = "Error"

    return render_template(
        "index.html",
        prediction=prediction,
        selected_crop=selected_crop,
        price=price,
        cost=cost,
        revenue=revenue,
        profit=profit
    )

if __name__ == "__main__":
    app.run(debug=True)
