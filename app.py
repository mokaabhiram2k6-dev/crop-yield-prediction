from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# =========================
# LOAD & CLEAN DATA
# =========================
df1 = pd.read_excel("data1.xlsx", engine="openpyxl")
df2 = pd.read_excel("data2.xlsx", engine="openpyxl")

df = pd.concat([df1, df2], ignore_index=True)

# clean column names
df.columns = df.columns.str.strip().str.lower()

# IMPORTANT FIX (removes NaN error)
df = df.dropna()

# =========================
# FEATURES
# =========================
features = ["soil_moisture", "temperature", "rainfall", "humidity", "sunlight"]

X = df[features]
y = df["yield"]

# TRAIN MODEL
model = LinearRegression()
model.fit(X, y)

# =========================
# HOME
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

            # prediction
            pred = model.predict([[moisture, temp, rain, humidity, sunlight]])[0]

            # =========================
            # MULTIPLE CROP SUGGESTION
            # =========================
            similar = df[
                (df["soil_type"] == soil_type)
            ].sort_values(by="yield", ascending=False).head(3)

            crops = similar[["crop", "price", "cost"]].to_dict(orient="records")

            result = round(pred, 2)

        except:
            result = "Error"

    return render_template("index.html", result=result, crops=crops)


if __name__ == "__main__":
    app.run(debug=True)
