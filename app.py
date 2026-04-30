from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# =========================
# SOIL TYPE NORMALISATION
# =========================
SOIL_NORMALISE = {
    "loamy soil":              "loamy",
    "sandy loam":              "sandy",
    "well-drained loam":       "loamy",
    "rich silty soil":         "silty",
    "moist loamy soil":        "loamy",
    "loose sandy loam":        "sandy",
    "well-drained loamy soil": "loamy",
    "rich well-drained soil":  "loamy",
    "red soils":                      "red",
    "arid and desert soils":          "sandy",
    "alluvial soils":                 "alluvial",
    "laterite and lateritic soils":   "laterite",
    "black soils":                    "black",
    "saline and alkaline soils":      "sandy",
    "peaty and marshy soils":         "peaty",
    "forest and mountain soils":      "loamy",
}

SOIL_NUMERIC = {
    "loamy": 1, "sandy": 2, "silty": 3, "clay": 4,
    "red": 5, "black": 6, "alluvial": 7, "laterite": 8, "peaty": 9,
}

# =========================
# LOAD & MERGE DATA
# =========================
def load_data():
    frames = []

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
        if "ph" not in df1.columns:
            df1["ph"] = 6.5
        frames.append(df1[["crop_type", "soil_type", "temperature", "humidity", "ph", "yield"]])

    p2 = "data2.xlsx"
    if os.path.exists(p2):
        df2 = pd.read_excel(p2)
        df2.columns = df2.columns.str.strip().str.lower()
        df2 = df2.rename(columns={
            "temperature_c":        "temperature",
            "humidity_%":           "humidity",
            "yield_kg_per_hectare": "yield",
        })
        if "ph" not in df2.columns:
            df2["ph"] = 6.5
        frames.append(df2[["crop_type", "soil_type", "temperature", "humidity", "ph", "yield"]])

    if not frames:
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
            "ph":         [6.0,6.5,6.2,6.0,7.0,7.5,5.5,6.5,6.0,6.5,
                           7.0,6.0,6.5,6.0,5.5,6.5,6.0,6.2,6.0,6.5,
                           6.0,7.5,7.0,7.5,6.5,6.5,5.5,5.5],
            "yield":      [4200,3600,5100,2700,2600,2500,2400,2900,8100,7200,
                           2600,3800,6800,3500,3300,2500,7400,4700,3000,2800,
                           2800,2600,3100,2900,4500,3700,2300,2100]
        }
        return pd.DataFrame(data)

    df = pd.concat(frames, ignore_index=True)
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df["ph"]    = pd.to_numeric(df["ph"],    errors="coerce").fillna(6.5)
    df = df.dropna(subset=["crop_type", "soil_type", "yield"])
    df["soil_type"] = (df["soil_type"]
                       .astype(str)
                       .str.strip()
                       .str.lower()
                       .map(SOIL_NORMALISE)
                       .fillna("loamy"))
    return df


# =========================
# IQR OUTLIER REMOVAL
# ph is now included in outlier check
# =========================
def remove_outliers_iqr(df, columns=("temperature", "humidity", "ph", "yield")):
    original_len = len(df)
    clean_frames = []

    for crop, group in df.groupby("crop_type"):
        mask = pd.Series([True] * len(group), index=group.index)
        for col in columns:
            if col not in group.columns:
                continue
            q1 = group[col].quantile(0.25)
            q3 = group[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask &= group[col].between(lower, upper)
        clean_frames.append(group[mask])

    cleaned_df = pd.concat(clean_frames, ignore_index=True)
    removed = original_len - len(cleaned_df)
    print(f"[IQR] Removed {removed} outlier rows out of {original_len} "
          f"({removed / max(original_len, 1) * 100:.1f}%) → {len(cleaned_df)} rows remain.")
    return cleaned_df


# =========================
# TRAIN RANDOM FOREST MODEL
# Features: soil_num, temperature, humidity, ph  ← ph added
# =========================
def train_model(df):
    try:
        df = df.copy()
        df["soil_num"] = df["soil_type"].map(SOIL_NUMERIC).fillna(0).astype(int)

        crop_counts = df["crop_type"].value_counts()
        valid_crops = crop_counts[crop_counts >= 2].index
        df = df[df["crop_type"].isin(valid_crops)]

        if len(df) < 10:
            print("[RF] Not enough data to train model.")
            return None, None

        X = df[["soil_num", "temperature", "humidity", "ph"]].values
        y = df["crop_type"].values

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        unique_classes = len(np.unique(y_enc))
        stratify = y_enc if unique_classes <= len(y_enc) // 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=stratify
        )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"[RF] Model trained — Test accuracy: {accuracy * 100:.1f}% "
              f"| Classes: {list(le.classes_)}")
        return model, le

    except Exception as e:
        print(f"[RF] Training failed: {e}")
        return None, None


# =========================
# STARTUP: load → clean → train
# =========================
raw_df              = load_data()
clean_df            = remove_outliers_iqr(raw_df)
rf_model, label_enc = train_model(clean_df)


# =========================
# PREDICT WITH RANDOM FOREST
# ph is now passed as a feature
# =========================
def rf_predict(soil_type, temperature, humidity, ph, top_n=3):
    if rf_model is None or label_enc is None:
        return []

    soil_num = SOIL_NUMERIC.get(soil_type.lower(), 0)
    X_input  = np.array([[soil_num, temperature, humidity, ph]])
    proba    = rf_model.predict_proba(X_input)[0]

    pairs = sorted(
        zip(label_enc.classes_, proba),
        key=lambda x: x[1],
        reverse=True
    )
    return pairs[:top_n]


# =========================
# FINANCIAL LOOKUP TABLES
# =========================
PRICES = {
    "Rice": 22, "Wheat": 20, "Maize": 18, "Cotton": 55, "Sugarcane": 3,
    "Soybean": 35, "Groundnut": 48, "Turmeric": 80, "Tomato": 25,
    "Onion": 15, "Banana": 20, "Jowar": 18, "Bajra": 17, "Ragi": 22,
    "Chili": 90, "Garlic": 60, "Mustard": 45, "Cashew": 120,
    "Cucumber": 18, "Pepper": 60, "Lettuce": 30, "Spinach": 25,
    "Radish": 12, "Beans": 40, "Basil": 100,
}
COSTS = {
    "Rice": 25000, "Wheat": 22000, "Maize": 20000, "Cotton": 35000, "Sugarcane": 30000,
    "Soybean": 20000, "Groundnut": 24000, "Turmeric": 40000, "Tomato": 30000,
    "Onion": 22000, "Banana": 28000, "Jowar": 15000, "Bajra": 14000, "Ragi": 16000,
    "Chili": 38000, "Garlic": 32000, "Mustard": 18000, "Cashew": 22000,
    "Cucumber": 25000, "Pepper": 35000, "Lettuce": 20000, "Spinach": 18000,
    "Radish": 12000, "Beans": 22000, "Basil": 15000,
}
DEFAULT_PRICE = 50
DEFAULT_COST  = 25000

BOUNDS = {
    "moisture":    (0, 100),
    "temperature": (-10, 60),
    "rainfall":    (0, 5000),
    "humidity":    (0, 100),
    "sunlight":    (0, 24),
    "ph":          (0, 14),
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
            ph       = float(request.form.get("ph", 7.0))

            # Server-side validation
            fields = {
                "moisture": moisture, "temperature": temp,
                "rainfall": rainfall, "humidity": humidity,
                "sunlight": sunlight, "ph": ph,
            }
            for field, val in fields.items():
                lo, hi = BOUNDS[field]
                if not (lo <= val <= hi):
                    raise ValueError(f"{field.upper()} must be between {lo} and {hi}.")

            form = {
                "soil": soil, "moisture": moisture, "temperature": temp,
                "rainfall": rainfall, "humidity": humidity,
                "sunlight": sunlight, "ph": ph,
            }

            # Suitability score
            # pH contributes up to 10 pts; ideal is 6.5, penalty grows as pH drifts away
            ph_score = max(0, 10 - abs(ph - 6.5) * 20)
            result = round(
                (moisture * 10) + (temp * 20) + (rainfall * 5) +
                (humidity * 8) + (sunlight * 15) + ph_score,
                2
            )

            # STEP 1: Random Forest prediction
            rf_results = rf_predict(soil, temp, humidity, ph, top_n=3)

            if rf_results:
                for crop_name, confidence in rf_results:
                    crop_rows = clean_df[clean_df["crop_type"] == crop_name]
                    yield_val = float(crop_rows["yield"].mean()) if len(crop_rows) > 0 else 3000.0
                    price   = PRICES.get(crop_name, DEFAULT_PRICE)
                    cost    = COSTS.get(crop_name, DEFAULT_COST)
                    revenue = round(yield_val * price, 2)
                    profit  = round(revenue - cost, 2)
                    crops.append({
                        "name":       crop_name,
                        "yield":      round(yield_val, 2),
                        "price":      price,
                        "cost":       cost,
                        "revenue":    revenue,
                        "profit":     profit,
                        "confidence": round(confidence * 100, 1),
                    })

            else:
                # STEP 2: Fallback — yield-based ranking
                filtered = clean_df.copy()
                if soil:
                    soil_match = clean_df[clean_df["soil_type"] == soil.lower()]
                    if len(soil_match) >= 3:
                        filtered = soil_match

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
                        "name":       name,
                        "yield":      round(yield_val, 2),
                        "price":      price,
                        "cost":       cost,
                        "revenue":    revenue,
                        "profit":     profit,
                        "confidence": None,
                    })

        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = "Something went wrong. Please check your inputs and try again."
            print("ERROR:", e)

    return render_template("index.html", result=result, crops=crops, form=form, error=error)


if __name__ == "__main__":
    app.run(debug=True)
