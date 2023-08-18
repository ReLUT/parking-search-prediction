from model import *
from train import *
import tensorflow as tf

def load_model(path="path/to/load/from"):
    return tf.keras.models.load_model(path)

def load_new_data():
    # Load new data for prediction
    pass

def preprocess_new_data(data):
    # Preprocess the new data
    pass

def make_prediction(model, data):
    return model.predict(data)

if __name__ == "__main__":
    model = load_model()
    new_data = load_new_data()
    processed_data = preprocess_new_data(new_data)
    predictions = make_prediction(model, processed_data)
    print(predictions)
