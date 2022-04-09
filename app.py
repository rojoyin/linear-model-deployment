from flask import Flask, render_template, request
import pandas as pd
from sklearn import linear_model
import pickle
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelDeployment")

model = pickle.load(open("model/model.pkl", "rb"))


@app.route("/")
def hello_world():
    return "<p>Hello world</p>"


@app.route("/retrain")
def regenerate_train_model():
    logger.info("Reading csv file")
    training_df = pd.read_csv("model/training.csv")
    # predictors independent variables
    predictors = training_df[["Rooms", "Distance"]]
    # outcomes dependent variables
    outcomes = training_df["Value"]
    logger.info("Creating model")
    lm = linear_model.LinearRegression()
    lm.fit(predictors, outcomes)
    logger.info("Saving model as pkl file")
    pickle.dump(lm, open("model/model.pkl", "wb"))
    return render_template("trained.html")


@app.route("/predict", methods=["POST"])
def predict():
    body = request.json
    if "Rooms" not in list(body.keys()) or "Distance" not in list(body.keys()):
        logger.error("Incorrect parameters in the body")
        return "Bad request, include 'Rooms' or 'Distance'", 400

    rooms = body["Rooms"]
    distance = body["Distance"]
    prediction = model.predict([[rooms, distance]])
    output = round(prediction[0], 2)
    return {"prediction": output}, 200


if __name__ == "__main__":
    app.run()
