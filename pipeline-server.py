from sklearn.externals import joblib
from flask import Flask, request
from jinja2 import Template

p = joblib.load('sentiment-model.pkl')

app = Flask(__name__)

def pred(text):
    return p.predict([text])[0]

@app.route('/')
def index():
    text = request.args.get('text')
    if text:
        prediction = pred(text)
    else:
        prediction = ""

    template = Template("""
    <html>
        <body>
            <h1>Sentiment Analysis</h1>
            <h3>Type a mesage here:</h3>
            <form>
                <input type="text" name="text" size="100">
                <input type="submit">
            </form>
            <p>Your input is: {{ text }}</p>
            <p>Prediction: {{ prediction }}</p>
        </body>
    </html>
    """)

    return template.render(prediction=prediction, text=text)


if __name__ == '__main__':
    app.run(port=8000)
