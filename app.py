from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

# =========================
# LOAD & MERGE DATA
# =========================
def load_data():
    frames = []

    if os.path.exists("data1.xlsx"):
        df1 = pd.read_excel("data1.xlsx")
        df1.columns = df1.columns.str.strip().str.lower()
        df1 = df1.rename(columns={
            "avg_temp":          "temperature",
            "humidity_r":        "humidity",
            "yield_kg_per_m2":   "yield"
        })
        frames.append(df1)

    if os.path.exists("data2.xlsx"):
        df2 = pd.read_excel("data2.xlsx")
        df2.columns = df2.columns.str.strip().str.lower()
        df2 = df2.rename(columns={
            "temperature_c":       "temperature",
            "humidity_%":          "humidity",
            "yield_kg_per_hectare":"yield"
        })
        frames.append(df2)

    if not frames:
        # Fallback sample data when Excel files are missing
        data = {
            "soil_type":  ["loamy","loamy","loamy","sandy","sandy","clay","clay","silty","peaty","loamy",
                           "sandy","clay","loamy","sandy","loamy","clay","loamy","silty","loamy","sandy"],
            "crop_type":  ["Rice","Wheat","Maize","Groundnut","Bajra","Cotton","Ragi","Soybean","Banana","Sugarcane",
                           "Jowar","Onion","Tomato","Chili","Turmeric","Mustard","Banana","Maize","Chili","Wheat"],
            "temperature":[28,22,30,35,38,32,27,26,25,30,
                           36,24,28,30,27,20,25,28,31,23],
            "humidity":   [75,60,65,40,35,50,70,68,80,72,
                           38,55,70,60,75,50,80,65,62,58],
            "yield":      [4200,3600,5100,2700,2600,2500,2400,2900,8100,7200,
                           2600,3800,6800,3500,3300,2500,7400,4700,3000,2800]
        }
        return pd.DataFrame(data)

    df = pd.concat(frames, ignore_index=True)
    needed = ["soil_type", "temperature", "humidity", "yield", "crop_type"]
    existing = [c for c in needed if c in df.columns]
    df = df[existing].dropna()
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df = df.dropna()
    return df

df = load_data()

# Crop prices and costs (₹)
PRICES = {
    "Rice":22, "Wheat":20, "Maize":18, "Cotton":55, "Sugarcane":3,
    "Soybean":35, "Groundnut":48, "Turmeric":80, "Tomato":25,
    "Onion":15, "Banana":20, "Jowar":18, "Bajra":17, "Ragi":22,
    "Chili":90, "Garlic":60, "Mustard":45
}
COSTS = {
    "Rice":25000, "Wheat":22000, "Maize":20000, "Cotton":35000, "Sugarcane":30000,
    "Soybean":20000, "Groundnut":24000, "Turmeric":40000, "Tomato":30000,
    "Onion":22000, "Banana":28000, "Jowar":15000, "Bajra":14000, "Ragi":16000,
    "Chili":38000, "Garlic":32000, "Mustard":18000
}
DEFAULT_PRICE = 50
DEFAULT_COST  = 25000

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    crops  = []
    form   = {}

    if request.method == "POST":
        try:
            soil     = request.form.get("soil", "").strip()
            moisture = float(request.form.get("moisture", 0))
            temp     = float(request.form.get("temperature", 0))
            rainfall = float(request.form.get("rainfall", 0))
            humidity = float(request.form.get("humidity", 0))
            sunlight = float(request.form.get("sunlight", 0))

            form = {
                "soil": soil, "moisture": moisture, "temperature": temp,
                "rainfall": rainfall, "humidity": humidity, "sunlight": sunlight
            }

            # Yield score formula
            result = round((moisture * 10) + (temp * 20) + (rainfall * 5) +
                           (humidity * 8) + (sunlight * 15), 2)

            # Filter by soil type; fallback to full dataset if < 3 results
            filtered = df.copy()
            if soil and "soil_type" in df.columns:
                soil_match = df[df["soil_type"].astype(str).str.lower() == soil.lower()]
                if len(soil_match) >= 3:
                    filtered = soil_match

            # Best yield per unique crop
            best = (filtered
                    .sort_values("yield", ascending=False)
                    .drop_duplicates(subset=["crop_type"])
                    .head(3))

            for _, row in best.iterrows():
                name      = str(row.get("crop_type", "Unknown")).strip()
                yield_val = float(row.get("yield", 0))
                price     = PRICES.get(name, DEFAULT_PRICE)
                cost      = COSTS.get(name, DEFAULT_COST)
                revenue   = round(yield_val * price, 2)
                profit    = round(revenue - cost, 2)
                crops.append({
                    "name":    name,
                    "yield":   round(yield_val, 2),
                    "price":   price,
                    "cost":    cost,
                    "revenue": revenue,
                    "profit":  profit
                })

        except Exception as e:
            print("ERROR:", e)

    return render_template("index.html", result=result, crops=crops, form=form)


if __name__ == "__main__":
    app.run(debug=True)
