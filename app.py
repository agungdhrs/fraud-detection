from flask import Flask, request, jsonify, render_template, send_file
import pickle
import pandas as pd
from datetime import datetime
import io
import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Set backend sebelum import pyplot
import matplotlib.pyplot as plt
import base64

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
                    'amount': row['Amount'],
                    'location': row['Location'],
                    'transaction_type': row['TransactionType'],
                    'timestamp': timestamp,
                    'prediction': result
                })
            except Exception as e:
                results.append({
                    'trasactionID': row.get('TransactionID', 'N/A'),
                    'customerID': row.get('CustomerID', 'N/A'),
                    'error': f'Error pada baris {index + 2}: {str(e)}'
                })
    
    finally:
        # Hapus file setelah selesai diproses
        if os.path.exists(filename):
            os.remove(filename)
    
    return pd.DataFrame(results)

def generate_plot(results_df):
    try:
        # Filter out rows with errors
        valid_predictions = results_df[~results_df['prediction'].str.contains('Error', na=False)]
        
        if valid_predictions.empty:
            return None, None, None
            
        # Extract prediction labels (FRAUD atau BUKAN FRAUD)
        prediction_labels = []
        for pred in valid_predictions['prediction']:
            if 'FRAUD' in pred and 'BUKAN' not in pred:
                prediction_labels.append('FRAUD')
            else:
                prediction_labels.append('BUKAN FRAUD')
        
        from collections import Counter
        
        # 1. PREDICTION DISTRIBUTION CHART
        counts = Counter(prediction_labels)
        
        plt.figure(figsize=(8, 6))
        ordered_data = []
        ordered_colors = []
        if 'BUKAN FRAUD' in counts:
            ordered_data.append(('BUKAN FRAUD', counts['BUKAN FRAUD']))
            ordered_colors.append('#28a745')
        if 'FRAUD' in counts:
            ordered_data.append(('FRAUD', counts['FRAUD']))
            ordered_colors.append('#dc3545')
        
        prediction_chart = None
        if ordered_data:
            labels, sizes = zip(*ordered_data)
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                   startangle=90, colors=ordered_colors,
                   textprops={'fontsize': 12})
            plt.title('Distribusi Hasil Prediksi Fraud Detection', fontsize=14, pad=20)
            plt.axis('equal')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            prediction_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close()
        
        # 2. LOCATION DISTRIBUTION CHART
        location_counts = Counter(valid_predictions['location'])
        location_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        plt.figure(figsize=(8, 6))
        if location_counts:
            labels = list(location_counts.keys())
            sizes = list(location_counts.values())
            colors = location_colors[:len(labels)]
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                   startangle=90, colors=colors,
                   textprops={'fontsize': 12})
            plt.title('Distribusi Transaksi Berdasarkan Lokasi', fontsize=14, pad=20)
            plt.axis('equal')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            location_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close()
        else:
            location_chart = None
        
        # 3. TRANSACTION TYPE DISTRIBUTION CHART
        type_counts = Counter(valid_predictions['transaction_type'])
        type_colors = ['#6C5CE7', '#A29BFE']
        
        plt.figure(figsize=(8, 6))
        if type_counts:
            labels = list(type_counts.keys())
            sizes = list(type_counts.values())
            colors = type_colors[:len(labels)]
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                   startangle=90, colors=colors,
                   textprops={'fontsize': 12})
            plt.title('Distribusi Transaksi Berdasarkan Tipe', fontsize=14, pad=20)
            plt.axis('equal')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            type_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close()
        else:
            type_chart = None
            
        return prediction_chart, location_chart, type_chart
            
    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        plt.close()
        return None, None, None

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
        
        # Buat plot hasil (3 charts)
        prediction_chart, location_chart, type_chart = generate_plot(results_df)

        # Convert hasil ke Excel
        output = io.BytesIO()
        results_df.to_excel(output, index=False)
        output.seek(0)

        # Encode file untuk unduhan sementara (base64)
        excel_base64 = base64.b64encode(output.getvalue()).decode('utf-8')

        # Tampilkan di halaman
        return render_template(
            'result.html',
            table=results_df.to_html(classes='table table-bordered', index=False),
            chart_data=prediction_chart,
            location_chart=location_chart,
            type_chart=type_chart,
            excel_data=excel_base64
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