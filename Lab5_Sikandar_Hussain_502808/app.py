from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model_pipeline = None

def load_model():
    global model_pipeline
    try:
        model_pipeline = joblib.load('best_classifier.joblib')
        print("model loaded successfully")
    except Exception as error:
        print(f"error loading model: {error}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_pipeline is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model_pipeline is None:
            return jsonify({
                'error': 'model not loaded'
            }), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'no input data provided'
            }), 400
        
        required_fields = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
        
        if isinstance(data, dict):
            data = [data]
        
        for record in data:
            missing = [field for field in required_fields if field not in record]
            if missing:
                return jsonify({
                    'error': f'missing required fields: {missing}'
                }), 400
        
        input_dataframe = pd.DataFrame(data)
        
        predictions = model_pipeline.predict(input_dataframe)
        probabilities = model_pipeline.predict_proba(input_dataframe)
        
        results = []
        for idx, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                'index': idx,
                'prediction': int(pred),
                'survival_status': 'survived' if pred == 1 else 'did not survive',
                'confidence': {
                    'died': float(probs[0]),
                    'survived': float(probs[1])
                }
            })
        
        return jsonify({
            'success': True,
            'predictions': results
        })
    
    except ValueError as error:
        return jsonify({
            'error': f'invalid input format: {str(error)}'
        }), 400
    
    except Exception as error:
        return jsonify({
            'error': f'prediction failed: {str(error)}'
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'logistic regression pipeline',
        'dataset': 'titanic survival',
        'metrics': {
            'accuracy': 0.8204,
            'precision': 0.7770,
            'recall': 0.7512,
            'f1': 0.7623,
            'roc_auc': 0.8626
        },
        'cross_validation': '5-fold',
        'training_config': {
            'random_state': 42,
            'test_size': 0.2
        }
    })

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
