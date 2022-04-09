from flask import Flask, render_template
import pandas as pd
from sklearn import linear_model
import pickle
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model app")


@app.route("/")
def hello_world():
    return "<p>Hello world</p>"


@app.route("/train")
def train_model():
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


if __name__ == "__main__":
    app.run()
