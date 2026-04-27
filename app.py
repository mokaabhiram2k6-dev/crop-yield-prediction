from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# =========================
# LOAD DATA
# =========================
df1 = pd.read_excel("data1.xlsx")
df2 = pd.read_excel("data2.xlsx")

# clean column names
df1.columns = df1.columns.str.strip().str.lower()
df2.columns = df2.columns.str.strip().str.lower()

# =========================
# RENAME COLUMNS SAFELY
# =========================
df1 = df1.rename(columns={
    "avg_temp": "temperature",
    "humidity_r": "humidity",
    "yield_kg_per_m2": "yield"
})

df2 = df2.rename(columns={
    "temperature_c": "temperature",
    "humidity_%": "humidity",
    "yield_kg_per_hectare": "yield"
})

# =========================
# MERGE (NO COLUMN FORCE)
# =========================
df = pd.concat([df1, df2], ignore_index=True)

# =========================
# KEEP ONLY EXISTING COLUMNS
# =========================
needed = ["soil_type", "temperature", "humidity", "yield", "crop_type"]

existing_cols = [col for col in needed if col in df.columns]
df = df[existing_cols]

# clean
df = df.dropna()
df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
df = df.dropna()

# =========================
# ROUTE
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

            result = (moisture * 10) + (temp * 20) + (rainfall * 5) + (humidity * 8) + (sunlight * 15)

            filtered = df.copy()

            if "soil_type" in df.columns and soil:
                temp_df = df[df["soil_type"].astype(str).str.lower() == soil.lower()]
                if not temp_df.empty:
                    filtered = temp_df

            top = filtered.sort_values(by="yield", ascending=False)

            top_unique = top.drop_duplicates(subset=["crop_type"])

            top3 = top_unique.head(3)

            for _, row in top3.iterrows():
                name = row.get("crop_type", "Unknown")
                yield_val = row.get("yield", 0)

                price = 50
                cost = 20000

                revenue = yield_val * price
                profit = revenue - cost

                crops.append({
                    "name": name,
                    "price": price,
                    "cost": cost,
                    "revenue": round(revenue, 2),
                    "profit": round(profit, 2)
                })

        except Exception as e:
            print("ERROR:", e)

    return render_template("index.html", result=result, crops=crops)


if __name__ == "__main__":
    app.run(debug=True)
