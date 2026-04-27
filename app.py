from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# =========================
# LOAD DATA
# =========================
df1 = pd.read_excel("data1.xlsx", engine="openpyxl")
df2 = pd.read_excel("data2.xlsx", engine="openpyxl")

df = pd.concat([df1, df2], ignore_index=True)
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

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    suggested_crops = []
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

            # SAFE ENCODING
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

            # =========================
            # MULTI CROP LOGIC
            # =========================
            if moisture > 50:
                suggested_crops = ["Rice", "Sugarcane", "Millets"]
            elif temp > 30:
                suggested_crops = ["Maize", "Groundnut", "Millets"]
            else:
                suggested_crops = ["Wheat", "Maize", "Rice"]

            # FIRST crop for calculation
            selected_crop = suggested_crops[0]

            price = crop_prices.get(selected_crop, 20)
            cost = crop_costs.get(selected_crop, 25000)

            revenue = prediction * price
            profit = revenue - cost

        except Exception as e:
            print(e)
            prediction = "Error"

    return render_template(
        "index.html",
        prediction=prediction,
        suggested_crops=suggested_crops,
        selected_crop=selected_crop,
        price=price,
        cost=cost,
        revenue=revenue,
        profit=profit
    )

if __name__ == "__main__":
    app.run(debug=True)
