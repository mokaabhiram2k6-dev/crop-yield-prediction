from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# =========================
# LOAD & CLEAN DATA
# =========================
df1 = pd.read_excel("data1.xlsx")
df2 = pd.read_excel("data2.xlsx")

# Standardize column names
df1.columns = df1.columns.str.strip().str.lower()
df2.columns = df2.columns.str.strip().str.lower()

# Rename columns to common format
df1 = df1.rename(columns={
    "avg_temp": "temperature",
    "humidity_r": "humidity",
    "yield_kg_per_m2": "yield",
    "crop_type": "crop_type",
    "soil_type": "soil_type"
})

df2 = df2.rename(columns={
    "temperature_c": "temperature",
    "humidity_%": "humidity",
    "yield_kg_per_hectare": "yield",
    "crop_type": "crop_type",
    "soil_type": "soil_type"
})

# Keep only required columns
df1 = df1[["soil_type", "temperature", "humidity", "yield", "crop_type"]]
df2 = df2[["soil_type", "temperature", "humidity", "yield", "crop_type"]]

# Combine
df = pd.concat([df1, df2], ignore_index=True)

# Clean data
df = df.dropna()
df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
df = df.dropna()

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    crops = []

    if request.method == "POST":
        try:
            soil = request.form.get("soil")
            moisture = float(request.form.get("moisture"))
            temp = float(request.form.get("temperature"))
            rainfall = float(request.form.get("rainfall"))
            humidity = float(request.form.get("humidity"))
            sunlight = float(request.form.get("sunlight"))

            # Simple prediction formula
            result = (moisture * 10) + (temp * 20) + (rainfall * 5) + (humidity * 8) + (sunlight * 15)

            # =========================
            # FILTER + TOP CROPS
            # =========================
            filtered = df.copy()

            if soil:
                filtered = df[df["soil_type"].astype(str).str.lower() == soil.lower()]
                if filtered.empty:
                    filtered = df

            top = filtered.sort_values(by="yield", ascending=False)

            # remove duplicates
            top_unique = top.drop_duplicates(subset=["crop_type"])

            top3 = top_unique.head(3)

            # =========================
            # ADD COST + PROFIT
            # =========================
            for _, row in top3.iterrows():
                crop_name = row["crop_type"]
                yield_val = row["yield"]

                price = 50      # default ₹/kg
                cost = 20000    # default cost

                revenue = yield_val * price
                profit = revenue - cost

                crops.append({
                    "name": crop_name,
                    "price": round(price, 2),
                    "cost": round(cost, 2),
                    "revenue": round(revenue, 2),
                    "profit": round(profit, 2)
                })

        except Exception as e:
            print("ERROR:", e)

    return render_template("index.html", result=result, crops=crops)


if __name__ == "__main__":
    app.run(debug=True)
