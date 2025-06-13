# app.py
from flask import Flask, request, jsonify, render_template, send_file
import pickle
import pandas as pd
from datetime import datetime
import io
import os
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

app = Flask(__name__)

# Konfigurasi upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the saved model
with open('fraud.pkl', 'rb') as f:
    model = pickle.load(f)

# Predefined lists for dropdown menus
LOCATIONS = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
TRANSACTION_TYPES = ['Online', 'In-Person']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(file):
    # Simpan file sementara
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    try:
        # Baca file berdasarkan ekstensi
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
        else:
            df = pd.read_excel(filename)
        
        # Proses setiap baris data
        results = []
        for index, row in df.iterrows():
            try:
                # Konversi timestamp
                timestamp = pd.to_datetime(row['Timestamp'])
                
                # Proses fitur
                features = [[
                    float(row['Amount']),
                    timestamp.hour,
                    timestamp.weekday(),
                    timestamp.month,
                    LOCATIONS.index(row['Location']),
                    TRANSACTION_TYPES.index(row['TransactionType'])
                ]]
                
                # Prediksi
                probability = model.predict_proba(features)[0][1]
                prediction = model.predict(features)[0]
                result = f'FRAUD ({probability:.2%})' if prediction == 1 else f'BUKAN FRAUD ({probability:.2%})'
                
                # Tambahkan hasil ke DataFrame
                results.append({
                    # 'row': index + 2,
                    # 'trasactionID': row['TransactionID'],
                    # 'customerID': row['CustomerID'],
                    'amount': row['Amount'],
                    'location': row['Location'],
                    'transaction_type': row['TransactionType'],
                    'timestamp': timestamp,
                    'prediction': result
                })
            except Exception as e:
                results.append({
                    'trasactionID': row['TransactionID'],
                    'customerID': row['CustomerID'],
                    'error': f'Error pada baris {index + 2}: {str(e)}'
                })
    
    finally:
        # Hapus file setelah selesai diproses
        if os.path.exists(filename):
            os.remove(filename)
    
    return pd.DataFrame(results)

@app.route('/')
def home():
    return render_template('index.html', 
                         locations=LOCATIONS,
                         transaction_types=TRANSACTION_TYPES)

@app.route('/upload')
def upload():
    return render_template('upload.html',
                         locations=LOCATIONS,
                         transaction_types=TRANSACTION_TYPES)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return render_template('upload.html',
                                error='No file uploaded',
                                locations=LOCATIONS,
                                transaction_types=TRANSACTION_TYPES)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html',
                                error='No file selected',
                                locations=LOCATIONS,
                                transaction_types=TRANSACTION_TYPES)
        
        if not allowed_file(file.filename):
            return render_template('upload.html',
                                error='Format file tidak didukung. Gunakan CSV atau Excel.',
                                locations=LOCATIONS,
                                transaction_types=TRANSACTION_TYPES)
        
        # Proses file
        results_df = process_file(file)
        
        # Convert hasil ke Excel
        output = io.BytesIO()
        results_df.to_excel(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='fraud_prediction_results.xlsx'
        )
    
    except Exception as e:
        return render_template('upload.html',
                             error=f'Error: {str(e)}',
                             locations=LOCATIONS,
                             transaction_types=TRANSACTION_TYPES)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Proses input form
        amount = float(request.form['amount'])
        location = request.form['location']
        transaction_type = request.form['transaction_type']
        timestamp = datetime.strptime(request.form['timestamp'], '%Y-%m-%dT%H:%M')
        
        # Process timestamp
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Convert categorical variables to numeric
        location_encoded = LOCATIONS.index(location)
        transaction_type_encoded = TRANSACTION_TYPES.index(transaction_type)
        
        # Create feature array
        features = [[
            amount,
            hour,
            day_of_week,
            month,
            location_encoded,
            transaction_type_encoded
        ]]
        
        # Make prediction
        probability = model.predict_proba(features)[0][1]
        prediction = model.predict(features)[0]
        
        # Determine result text
        result = f'FRAUD ({probability:.2%})' if prediction == 1 else f'BUKAN FRAUD ({probability:.2%})'
        alert_class = 'danger' if prediction == 1 else 'success'
        
        return render_template('index.html',
                             prediction_text=result,
                             alert_class=alert_class,
                             locations=LOCATIONS,
                             transaction_types=TRANSACTION_TYPES)
    
    except Exception as e:
        return render_template('index.html',
                             error=f'Error: {str(e)}',
                             locations=LOCATIONS,
                             transaction_types=TRANSACTION_TYPES)

if __name__ == '__main__':
    app.run(debug=True)
