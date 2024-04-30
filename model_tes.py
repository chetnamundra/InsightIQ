from keras.models import load_model
import joblib

keras_model = load_model("D:\Hackathon\model.h5")
joblib.dump(keras_model, 'emotion_model.pkl')