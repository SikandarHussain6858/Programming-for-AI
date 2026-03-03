import requests
import sys

base = "http://localhost:5000"


def run_test(name, method, url, expected_status, **kwargs):
    r = getattr(requests, method)(url, **kwargs)
    status = "PASS" if r.status_code == expected_status else "FAIL"
    print(f"[{status}] {name} -> {r.status_code} {r.json()}")
    return r.status_code == expected_status


def main():
    results = []

    # health check
    results.append(run_test("health check", "get", f"{base}/health", 200))

    # valid titanic prediction
    valid = {"pclass": 3, "sex": "male", "age": 22, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"}
    results.append(run_test("valid predict", "post", f"{base}/predict", 200, json=valid))

    # missing field
    no_fare = {"pclass": 3, "sex": "male", "age": 22, "sibsp": 1, "parch": 0, "embarked": "S"}
    results.append(run_test("missing fare", "post", f"{base}/predict", 400, json=no_fare))

    # wrong type
    bad_type = {"pclass": 3, "sex": "male", "age": "young", "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"}
    results.append(run_test("wrong type", "post", f"{base}/predict", 400, json=bad_type))

    # invalid json
    results.append(run_test("invalid json", "post", f"{base}/predict", 400,
                            data="not json", headers={"Content-Type": "application/json"}))

    # extra fields ignored
    extra = {"pclass": 1, "sex": "female", "age": 30, "sibsp": 0, "parch": 0,
             "fare": 100.0, "embarked": "C", "cabin": "B22", "ticket": "12345"}
    results.append(run_test("extra fields", "post", f"{base}/predict", 200, json=extra))

    # with probabilities
    with_proba = {"pclass": 1, "sex": "female", "age": 30, "sibsp": 0, "parch": 0,
                  "fare": 100.0, "embarked": "C", "return_proba": True}
    results.append(run_test("with proba", "post", f"{base}/predict", 200, json=with_proba))

    # custom threshold
    with_thresh = {"pclass": 1, "sex": "female", "age": 30, "sibsp": 0, "parch": 0,
                   "fare": 100.0, "embarked": "C", "return_proba": True, "threshold": 0.8}
    results.append(run_test("custom threshold", "post", f"{base}/predict", 200, json=with_thresh))

    # valid housing
    house = {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984, "AveBedrms": 1.024,
             "Population": 322.0, "AveOccup": 2.556, "Latitude": 37.88, "Longitude": -122.23}
    results.append(run_test("valid housing", "post", f"{base}/predict_price", 200, json=house))

    # housing missing field
    house_bad = {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984, "AveBedrms": 1.024,
                 "AveOccup": 2.556, "Latitude": 37.88, "Longitude": -122.23}
    results.append(run_test("housing missing", "post", f"{base}/predict_price", 400, json=house_bad))

    # housing wrong type
    house_type = {"MedInc": "high", "HouseAge": 41.0, "AveRooms": 6.984, "AveBedrms": 1.024,
                  "Population": 322.0, "AveOccup": 2.556, "Latitude": 37.88, "Longitude": -122.23}
    results.append(run_test("housing wrong type", "post", f"{base}/predict_price", 400, json=house_type))

    # invalid api key
    results.append(run_test("invalid api key", "post", f"{base}/predict", 401,
                            json=valid, headers={"X-API-Key": "wrong-key"}))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
