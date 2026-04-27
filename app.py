from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

# =========================
# SOIL TYPE NORMALISATION
# Maps raw Excel soil names → pill values shown in the UI
# =========================
SOIL_NORMALISE = {
    # data1.xlsx variants (greenhouse — loamy-family)
    "loamy soil":              "loamy",
    "sandy loam":              "sandy",
    "well-drained loam":       "loamy",
    "rich silty soil":         "silty",
    "moist loamy soil":        "loamy",
    "loose sandy loam":        "sandy",
    "well-drained loamy soil": "loamy",
    "rich well-drained soil":  "loamy",
    # data2.xlsx variants
    "red soils":                      "red",
    "arid and desert soils":          "sandy",
    "alluvial soils":                 "alluvial",
    "laterite and lateritic soils":   "laterite",
    "black soils":                    "black",
    "saline and alkaline soils":      "sandy",
    "peaty and marshy soils":         "peaty",
    "forest and mountain soils":      "loamy",
}

# =========================
# LOAD & MERGE DATA
# =========================
def load_data():
    frames = []

    # --- data1.xlsx  (greenhouse crops, yield in kg/m2 → convert to kg/ha) ---
    p1 = "data1.xlsx"
    if os.path.exists(p1):
        df1 = pd.read_excel(p1)
        df1.columns = df1.columns.str.strip().str.lower()
        df1 = df1.rename(columns={
            "avg_temperature_c": "temperature",
            "humidity_percent":  "humidity",
            "yield_kg_per_m2":   "yield_raw",
        })
        df1["yield"] = pd.to_numeric(df1["yield_raw"], errors="coerce") * 10000
        frames.append(df1[["crop_type", "soil_type", "temperature", "humidity", "yield"]])

    # --- data2.xlsx  (field crops, yield already in kg/ha) ---
    p2 = "data2.xlsx"
    if os.path.exists(p2):
        df2 = pd.read_excel(p2)
        df2.columns = df2.columns.str.strip().str.lower()
        df2 = df2.rename(columns={
            "temperature_c":       "temperature",
            "humidity_%":          "humidity",
            "yield_kg_per_hectare":"yield",
        })
        frames.append(df2[["crop_type", "soil_type", "temperature", "humidity", "yield"]])

    if not frames:
        # Fallback sample data when Excel files are missing
        data = {
            "soil_type":  ["loamy","loamy","loamy","sandy","sandy","clay","clay","silty","peaty","loamy",
                           "sandy","clay","loamy","sandy","loamy","clay","loamy","silty","loamy","sandy",
                           "red","red","black","black","alluvial","alluvial","laterite","laterite"],
            "crop_type":  ["Rice","Wheat","Maize","Groundnut","Bajra","Cotton","Ragi","Soybean","Banana","Sugarcane",
                           "Jowar","Onion","Tomato","Chili","Turmeric","Mustard","Banana","Maize","Chili","Wheat",
                           "Groundnut","Cotton","Soybean","Cotton","Rice","Wheat","Ragi","Cashew"],
            "temperature":[28,22,30,35,38,32,27,26,25,30,
                           36,24,28,30,27,20,25,28,31,23,
                           33,31,29,30,27,22,28,30],
            "humidity":   [75,60,65,40,35,50,70,68,80,72,
                           38,55,70,60,75,50,80,65,62,58,
                           45,48,60,55,75,65,70,65],
            "yield":      [4200,3600,5100,2700,2600,2500,2400,2900,8100,7200,
                           2600,3800,6800,3500,3300,2500,7400,4700,3000,2800,
                           2800,2600,3100,2900,4500,3700,2300,2100]
        }
        return pd.DataFrame(data)

    df = pd.concat(frames, ignore_index=True)
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df = df.dropna(subset=["crop_type", "soil_type", "yield"])

    # Normalise soil_type to match UI pill values
    df["soil_type"] = (df["soil_type"]
                       .astype(str)
                       .str.strip()
                       .str.lower()
                       .map(SOIL_NORMALISE)
                       .fillna("loamy"))
    return df

df = load_data()

# Crop prices and costs (Rs. per kg / per hectare)
PRICES = {
    "Rice":22, "Wheat":20, "Maize":18, "Cotton":55, "Sugarcane":3,
    "Soybean":35, "Groundnut":48, "Turmeric":80, "Tomato":25,
    "Onion":15, "Banana":20, "Jowar":18, "Bajra":17, "Ragi":22,
    "Chili":90, "Garlic":60, "Mustard":45, "Cashew":120,
    "Cucumber":18, "Pepper":60, "Lettuce":30, "Spinach":25,
    "Radish":12, "Beans":40, "Basil":100,
}
COSTS = {
    "Rice":25000, "Wheat":22000, "Maize":20000, "Cotton":35000, "Sugarcane":30000,
    "Soybean":20000, "Groundnut":24000, "Turmeric":40000, "Tomato":30000,
    "Onion":22000, "Banana":28000, "Jowar":15000, "Bajra":14000, "Ragi":16000,
    "Chili":38000, "Garlic":32000, "Mustard":18000, "Cashew":22000,
    "Cucumber":25000, "Pepper":35000, "Lettuce":20000, "Spinach":18000,
    "Radish":12000, "Beans":22000, "Basil":15000,
}
DEFAULT_PRICE = 50
DEFAULT_COST  = 25000

# Validation bounds
BOUNDS = {
    "moisture":    (0, 100),
    "temperature": (-10, 60),
    "rainfall":    (0, 5000),
    "humidity":    (0, 100),
    "sunlight":    (0, 24),
}

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    crops  = []
    form   = {}
    error  = None

    if request.method == "POST":
        try:
            soil     = request.form.get("soil", "").strip()
            moisture = float(request.form.get("moisture", 0))
            temp     = float(request.form.get("temperature", 0))
            rainfall = float(request.form.get("rainfall", 0))
            humidity = float(request.form.get("humidity", 0))
            sunlight = float(request.form.get("sunlight", 0))

            # Server-side validation
            fields = {
                "moisture": moisture, "temperature": temp,
                "rainfall": rainfall, "humidity": humidity, "sunlight": sunlight
            }
            for field, val in fields.items():
                lo, hi = BOUNDS[field]
                if not (lo <= val <= hi):
                    raise ValueError(f"{field.capitalize()} must be between {lo} and {hi}.")

            form = {
                "soil": soil, "moisture": moisture, "temperature": temp,
                "rainfall": rainfall, "humidity": humidity, "sunlight": sunlight
            }

            # Suitability score
            result = round((moisture * 10) + (temp * 20) + (rainfall * 5) +
                           (humidity * 8) + (sunlight * 15), 2)

            # Filter by normalised soil type; fallback to full dataset if < 3 rows
            filtered = df.copy()
            if soil:
                soil_match = df[df["soil_type"] == soil.lower()]
                if len(soil_match) >= 3:
                    filtered = soil_match

            # Best average yield per unique crop
            best = (filtered
                    .groupby("crop_type", as_index=False)["yield"]
                    .mean()
                    .sort_values("yield", ascending=False)
                    .head(3))

            for _, row in best.iterrows():
                name      = str(row["crop_type"]).strip()
                yield_val = float(row["yield"])
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
                    "profit":  profit,
                })

        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = "Something went wrong. Please check your inputs and try again."
            print("ERROR:", e)

    return render_template("index.html", result=result, crops=crops, form=form, error=error)


if __name__ == "__main__":
    app.run(debug=True)
