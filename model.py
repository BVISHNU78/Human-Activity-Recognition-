from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
video_path=r"D:\coding\horse fitness\Walking With Phone Stock Footage #Stock Footage.mp4"
cap =cv2.VideoCapture(video_path)
model =load_model(r"D:\coding\horse fitness\Video_Classfication_model.h5")

frames=[]
num_frames_to_read =20

while len(frames)<num_frames_to_read:
    ret,frame =cap.read()
    if not ret:
        break
    frame_resized =cv2.resize(frame,(244,244))
    frame_resized =frame_resized/255.0
    frames.append(frame_resized) 
cap.release()
def build_feature_extractor():
        feature_extractor = keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(244,244,3),
            )
        preprocess_input =keras.applications.inception_v3.preprocess_input
        inputs = keras.Input((244,244,3))
        preprocessed = preprocess_input(inputs)
        outputs= feature_extractor(preprocessed)
        return keras.Model(inputs,outputs,name="feature_extractor")
feature_extractor = build_feature_extractor()

def extract_features(video_path,max_frames=30):
    frames = load_model(video_path,max_frames=max_frames)
    features=[]
    for frames in frame:
        frame=np.expand_dims(frame,axis=0)
        feature=feature_extractor.predict(frame)
        features.append(feature[0])
    return np.array(features)
features =np.array(frames)
features =np.expand_dims(features,axis=0)
predictions = model.predict(features)
predictions_class =np.argmax(predictions,axis=-1)
