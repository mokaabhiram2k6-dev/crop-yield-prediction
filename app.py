from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# =========================
# SOIL TYPE NORMALISATION
# =========================
SOIL_NORMALISE = {
    "loamy soil":"loamy","sandy loam":"sandy","well-drained loam":"loamy",
    "rich silty soil":"silty","moist loamy soil":"loamy","loose sandy loam":"sandy",
    "well-drained loamy soil":"loamy","rich well-drained soil":"loamy",
    "red soils":"red","arid and desert soils":"sandy","alluvial soils":"alluvial",
    "laterite and lateritic soils":"laterite","black soils":"black",
    "saline and alkaline soils":"sandy","peaty and marshy soils":"peaty",
    "forest and mountain soils":"loamy",
}

# =========================
# IQR OUTLIER REMOVAL
# =========================
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# =========================
# LOAD DATA
# =========================
def load_data():
    frames = []

    if os.path.exists("data1.xlsx"):
        df1 = pd.read_excel("data1.xlsx")
        df1.columns = df1.columns.str.strip().str.lower()
        df1 = df1.rename(columns={
            "avg_temperature_c":"temperature",
            "humidity_percent":"humidity",
            "yield_kg_per_m2":"yield_raw"
        })
        df1["yield"] = pd.to_numeric(df1["yield_raw"], errors="coerce") * 10000
        frames.append(df1[["crop_type","soil_type","temperature","humidity","yield"]])

    if os.path.exists("data2.xlsx"):
        df2 = pd.read_excel("data2.xlsx")
        df2.columns = df2.columns.str.strip().str.lower()
        df2 = df2.rename(columns={
            "temperature_c":"temperature",
            "humidity_%":"humidity",
            "yield_kg_per_hectare":"yield"
        })
        frames.append(df2[["crop_type","soil_type","temperature","humidity","yield"]])

    df = pd.concat(frames, ignore_index=True)

    # Clean
    df = df.dropna()
    df["soil_type"] = df["soil_type"].astype(str).str.lower().map(SOIL_NORMALISE).fillna("loamy")

    # 👉 IQR OUTLIER REMOVAL
    for col in ["temperature","humidity","yield"]:
        df = remove_outliers_iqr(df, col)

    return df

df = load_data()

# =========================
# ENCODING
# =========================
le_crop = LabelEncoder()
le_soil = LabelEncoder()

df["crop_encoded"] = le_crop.fit_transform(df["crop_type"])
df["soil_encoded"] = le_soil.fit_transform(df["soil_type"])

# =========================
# TRAIN MODEL (Random Forest)
# =========================
X = df[["soil_encoded","temperature","humidity"]]
y = df["yield"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# PRICES & COSTS
# =========================
PRICES = {"Rice":22,"Wheat":20,"Maize":18,"Cotton":55,"Sugarcane":3,
"Soybean":35,"Groundnut":48,"Turmeric":80,"Tomato":25,"Onion":15,
"Banana":20,"Jowar":18,"Bajra":17,"Ragi":22,"Chili":90,"Garlic":60,
"Mustard":45,"Cashew":120}

COSTS = {"Rice":25000,"Wheat":22000,"Maize":20000,"Cotton":35000,
"Sugarcane":30000,"Soybean":20000,"Groundnut":24000,"Turmeric":40000,
"Tomato":30000,"Onion":22000,"Banana":28000,"Jowar":15000,"Bajra":14000,
"Ragi":16000,"Chili":38000,"Garlic":32000,"Mustard":18000,"Cashew":22000}

DEFAULT_PRICE = 50
DEFAULT_COST = 25000

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET","POST"])
def index():
    crops = []
    result = None
    error = None

    if request.method == "POST":
        try:
            soil = request.form.get("soil").lower()
            temp = float(request.form.get("temperature"))
            humidity = float(request.form.get("humidity"))

            soil_enc = le_soil.transform([soil])[0]

            predictions = []

            # Predict for ALL crops
            for crop in le_crop.classes_:
                crop_enc = le_crop.transform([crop])[0]

                pred_yield = model.predict([[soil_enc, temp, humidity]])[0]

                price = PRICES.get(crop, DEFAULT_PRICE)
                cost = COSTS.get(crop, DEFAULT_COST)

                revenue = pred_yield * price
                profit = revenue - cost

                predictions.append({
                    "name": crop,
                    "yield": round(pred_yield,2),
                    "profit": round(profit,2)
                })

            # Sort by profit
            top = sorted(predictions, key=lambda x: x["profit"], reverse=True)[:3]

            crops = top

            result = "ML Prediction Done"

        except Exception as e:
            error = str(e)

    return render_template("index.html", crops=crops, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
