from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

# Home route to check if the server is running
@app.route("/")
def home():
    return "Sentiment Analysis API is running! Use /predict endpoint."

# Sentiment prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = [data["text"]]
    X_test = vectorizer.transform(text)
    prediction = model.predict(X_test)
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

