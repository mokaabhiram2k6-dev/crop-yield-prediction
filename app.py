from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# =========================
# LOAD DATA
# =========================
df1 = pd.read_excel("data1.xlsx", engine="openpyxl")
df2 = pd.read_excel("data2.xlsx", engine="openpyxl")

# normalize column names
df1.columns = df1.columns.str.strip().str.lower()
df2.columns = df2.columns.str.strip().str.lower()

# =========================
# FIX YIELD
# =========================
# data1: m2 → hectare
if "yield_kg_per_m2" in df1.columns:
    df1["yield"] = df1["yield_kg_per_m2"] * 10000

# data2: already hectare
if "yield_kg_per_hectare" in df2.columns:
    df2["yield"] = df2["yield_kg_per_hectare"]

# =========================
# RENAME COLUMNS
# =========================
df2 = df2.rename(columns={
    "soil moisture_%": "soil_moisture",
    "temperature_c": "temperature",
    "rainfall_mm": "rainfall",
    "humidity_%": "humidity",
    "sunlight_hour": "sunlight"
})

# =========================
# MERGE
# =========================
df = pd.concat([df1, df2], ignore_index=True)

# =========================
# CLEAN DATA
# =========================
df = df.fillna(0)

# ensure all features exist
features = ["soil_moisture", "temperature", "rainfall", "humidity", "sunlight"]

for col in features:
    if col not in df.columns:
        df[col] = 0

# =========================
# MODEL
# =========================
X = df[features]
y = df["yield"]

model = LinearRegression()
model.fit(X, y)

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    crops = []

    if request.method == "POST":
        try:
            soil_type = request.form["soil_type"]

            moisture = float(request.form["moisture"])
            temp = float(request.form["temperature"])
            rain = float(request.form["rainfall"])
            humidity = float(request.form["humidity"])
            sunlight = float(request.form["sunlight"])

            pred = model.predict([[moisture, temp, rain, humidity, sunlight]])[0]

            # =========================
            # TOP 3 CROPS
            # =========================
            similar = df.sort_values(by="yield", ascending=False).head(3)

            crops = similar[["crop_type"]].to_dict(orient="records")

            result = round(pred, 2)

        except Exception as e:
            print("ERROR:", e)
            result = "Error"

    return render_template("index.html", result=result, crops=crops)


if __name__ == "__main__":
    app.run(debug=True)
