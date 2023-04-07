import os
from flask import Flask, request, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True, allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"])

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided.'}), 400
        
        # Get file and filename
        file = request.files['file']
        filename = secure_filename(file.filename)

        # Check if file format is supported
        file_format = os.path.splitext(filename)[1]
        if file_format not in ['.csv', '.txt', '.xlsx']:
            return jsonify({'error': 'Unsupported file format.'}), 400

        # Read file into pandas DataFrame
        if file_format == '.xlsx':
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        # Perform data preprocessing, cleaning, visualization, and model training
        # ...

        # Convert DataFrame to CSV format
        csv_data = df.to_csv(index=False)

        # Return success response
        comment=request.form.get('comment')
        return jsonify({'message': f'File {filename} uploaded successfully.', 'comment': comment, 'data': csv_data}), 200
    
    except FileNotFoundError:
        return jsonify({'error': 'No file found.'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    

if __name__ == '__main__':
    app.run(debug=True)
