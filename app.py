from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('sales_prediction_lr_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_sales():
    data = request.json

    # Extract input data
    try:
        month = data['month']  # Month index (0-11)
        product_code = data['product_code']  # Product code identifier

        if not isinstance(month, int) or month < 0 or month > 11:
            return jsonify({'error': 'Month must be an integer between 0 and 11.'}), 400

        if not isinstance(product_code, int):
            return jsonify({'error': 'Product code must be an integer.'}), 400
    except KeyError:
        return jsonify({'error': 'Missing "month" or "product_code" field in request body.'}), 400

    # Simulated data lookup (replace with actual database or data structure)
    dummy_sales_data = {
        69263: [100, 200, 150, 300, 400, 250, 350, 300, 200, 100, 150, 300],
        69266: [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
    }

    if product_code not in dummy_sales_data:
        return jsonify({'error': f'Product code {product_code} not found.'}), 404

    # Retrieve the sales data for the given product code
    sales_data = dummy_sales_data[product_code]

    # Prepare input for the model
    input_data = np.zeros(12)
    input_data[month] = sales_data[month]
    input_data = input_data.reshape(1, -1)
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)[0]

    return jsonify({
        'month': month,
        'product_code': product_code,
        'predicted_sales': prediction
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
