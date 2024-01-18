from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        user_input = request.form.to_dict()

        user_data = pd.DataFrame([user_input])

        prediction = model.predict(user_data)

        return render_template('display.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)