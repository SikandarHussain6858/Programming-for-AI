from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

titanic_model = None
housing_model = None

api_key = "lab6-secret-key"

titanic_required = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
titanic_numeric = ["pclass", "age", "sibsp", "parch", "fare"]

housing_required = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude"]


def fix_ohe_nans(pipeline):
    """fix nan values in ohe categories for sklearn compat"""
    from sklearn.preprocessing import OneHotEncoder
    ct = pipeline.named_steps.get("preprocess")
    if ct is None:
        return
    for name, trans, cols in ct.transformers_:
        if isinstance(trans, OneHotEncoder):
            for i, cats in enumerate(trans.categories_):
                fixed = []
                needs_fix = False
                for c in cats:
                    if isinstance(c, float) and np.isnan(c):
                        fixed.append("__nan__")
                        needs_fix = True
                    else:
                        fixed.append(c)
                if needs_fix:
                    trans.categories_[i] = np.array(fixed, dtype=object)


def load_models():
    global titanic_model, housing_model
    titanic_model = joblib.load("titanic_pipeline.joblib")
    fix_ohe_nans(titanic_model)
    log.info("titanic model loaded")

    housing_model = joblib.load("housing_pipeline.joblib")
    log.info("housing model loaded")


def check_api_key():
    key = request.headers.get("X-API-Key")
    if key is not None and key != api_key:
        return jsonify({"error": "invalid API key"}), 401
    return None


def check_numeric(data, fields):
    bad = [f for f in fields if f in data and not isinstance(data[f], (int, float))]
    return bad


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "titanic_model_loaded": titanic_model is not None,
        "housing_model_loaded": housing_model is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    auth_err = check_api_key()
    if auth_err:
        return auth_err

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    if data is None:
        return jsonify({"error": "invalid JSON"}), 400

    missing = [f for f in titanic_required if f not in data]
    if missing:
        return jsonify({"error": f"missing required fields: {missing}"}), 400

    bad = check_numeric(data, titanic_numeric)
    if bad:
        return jsonify({"error": f"expected numeric values for: {bad}"}), 400

    # keep only known fields, ignore extras
    clean = {k: data[k] for k in titanic_required}

    log.info(f"predict input: {clean}")

    # derive extra features the pipeline needs
    clean["adult_male"] = bool((clean["sex"] == "male") and (clean["age"] >= 18))
    clean["alone"] = bool((clean["sibsp"] + clean["parch"]) == 0)
    clean["deck"] = "__nan__"

    threshold = data.get("threshold", 0.5)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.5

    try:
        df = pd.DataFrame([clean])
        df["adult_male"] = df["adult_male"].astype(bool)
        df["alone"] = df["alone"].astype(bool)
        proba = titanic_model.predict_proba(df)
        pred = int(proba[0][1] >= threshold)

        result = {"prediction": pred}

        if data.get("return_proba", False):
            result["probability"] = {
                "died": round(float(proba[0][0]), 4),
                "survived": round(float(proba[0][1]), 4),
            }

        log.info(f"predict result: {result}")
        return jsonify(result)

    except Exception as e:
        log.error(f"predict failed: {e}")
        return jsonify({"error": f"prediction failed: {str(e)}"}), 500


@app.route("/predict_price", methods=["POST"])
def predict_price():
    auth_err = check_api_key()
    if auth_err:
        return auth_err

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    if data is None:
        return jsonify({"error": "invalid JSON"}), 400

    missing = [f for f in housing_required if f not in data]
    if missing:
        return jsonify({"error": f"missing required fields: {missing}"}), 400

    bad = check_numeric(data, housing_required)
    if bad:
        return jsonify({"error": f"expected numeric values for: {bad}"}), 400

    clean = {k: data[k] for k in housing_required}

    log.info(f"predict_price input: {clean}")

    try:
        df = pd.DataFrame([clean])
        pred = housing_model.predict(df)
        price = round(float(pred[0]), 4)

        result = {"predicted_price": price}
        log.info(f"predict_price result: {result}")
        return jsonify(result)

    except Exception as e:
        log.error(f"predict_price failed: {e}")
        return jsonify({"error": f"prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
