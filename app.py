from flask import Flask
import pandas as pd
from sklearn import linear_model
import pickle

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello world</p>"


@app.route("/train")
def train_model():
    training_df = pd.read_csv("model/training.csv")
    # predictors independent variables
    predictors = training_df[["Rooms", "Distance"]]
    # outcomes dependent variables
    outcomes = training_df["Value"]
    lm = linear_model.LinearRegression()
    lm.fit(predictors, outcomes)
    pickle.dump(lm, open("model/model.pkl", "wb"))


if __name__ == "__main__":
    app.run()
