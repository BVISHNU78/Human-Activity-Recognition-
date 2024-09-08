import cv2
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import pathlib
import random
import pandas as pd
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt 
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from keras import layers
import numpy as np
from IPython.display import Video
from tensorflow_docs.vis import embed
dataset_path =r"C:\Users\Dell\Downloads\Human Activity Recognition - Video Dataset"
label_types = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,d))]
Image_size=244
Batch_size = 62
Epochs = 100
Max_seq_length =20
Num_Features =2048
def crop_centre_square(frame):
    y,x=frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(dataset_path,max_frames=30,resize=(Image_size,Image_size)):
    cap=cv2.VideoCapture(dataset_path)
    frames=[]
    try:
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            frame = crop_centre_square(frame)
            frame = cv2.resize(frame,resize)
            frame=frame[:,:,[2,1,0]]
            frames.append(frame)
            if len(frames)==max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

plt.figure(figsize=(20,20))
all_classes = os.listdir(dataset_path)
random_range =random.sample(range(len(all_classes)),7)
for count , random_index in enumerate(random_range,1):
    select_class_name = all_classes[random_index]
    print(select_class_name)
    video_files_names_list = os.listdir(f'{dataset_path}/{select_class_name}')
    selected_video_file_name =random.choice(video_files_names_list)
    Video_reader=cv2.VideoCapture(f'{dataset_path}/{select_class_name}/{selected_video_file_name}')
    __,bgr_frame =Video_reader.read()
    Video_reader.release()
    rgb_frame =cv2.cvtColor(bgr_frame,cv2.COLOR_BGR2RGB)
    cv2.putText(rgb_frame,select_class_name,(10,30),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),2)
    plt.subplot(6,5,count)
    plt.imshow(rgb_frame)
    plt.axis('off')


    def build_feature_extractor():
        feature_extractor = keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(Image_size,Image_size,3),
            )
        preprocess_input =keras.applications.inception_v3.preprocess_input
        inputs = keras.Input((Image_size,Image_size,3))
        preprocessed = preprocess_input(inputs)
        outputs= feature_extractor(preprocessed)
        return keras.Model(inputs,outputs,name="feature_extractor")
feature_extractor = build_feature_extractor()

def extract_features(Video_path,max_frames=30):
    frames = load_video(Video_path,max_frames=max_frames)
    features=[]
    for frame in frames:
        frame=np.expand_dims(frame,axis=0)
        feature=feature_extractor.predict(frame)
        features.append(feature[0])
    return np.array(features)

video_path = f'{dataset_path}/{select_class_name}/{selected_video_file_name}'
features = extract_features(video_path)
print(features)
print(video_path)

features_list=[]
lables_list=[]

for label in label_types:
    label_folder = os.path.join(dataset_path,label)
    video_files = os.listdir(label_folder)

    for video_file in video_files:
        video_path=os.path.join(label_folder,video_file)
        features =extract_features(video_path,max_frames=Max_seq_length)
        features_list.append(features)
        lables_list.append(label)
        print(len(features_list))

        if len(features) == Max_seq_length:
            features_list.append(features)
            lables_list.append(label)

x=np.array(features_list)
y=np.array(lables_list)
label_encoder = LabelEncoder()
y_encoded =label_encoder.fit_transform(y)
print(f"Length of y_encoded: {len(y_encoded)}")
features_padded =pad_sequences(x,maxlen=Max_seq_length,padding='post',truncating='post')
print(f"Length of features_padded: {len(features_padded)}")
x_reshaped = features_padded.reshape(features_padded.shape[0],Max_seq_length, Num_Features)
#x_padded = np.squeeze(features_padded, axis=2)
x_train,x_test,y_train,y_test=train_test_split(features_padded,y_encoded,train_size=0.5,test_size=0.5,random_state=42,stratify=y_encoded)
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
x_train_flat = x_train.reshape(x_train.shape[0],x_train.shape[1], -1)
x_test_flat = x_test.reshape(x_test.shape[0],x_test.shape[1], -1)
#train_data = pd.DataFrame(x_train_flat)
#train_data['label'] = y_train
#test_data = pd.DataFrame(x_test_flat)
#test_data['label'] = y_test
#train_data.to_csv('train_data.csv', index=False)
#test_data.to_csv('test_data.csv', index=False)
#print("CSV files for train and test data saved successfully.")
def get_sequence_model(input_shape,select_class_name):
    model = keras.Sequential()
    model.add(layers.Masking(mask_value=0.0, input_shape=(Max_seq_length,Num_Features)))
    model.add(layers.LSTM(64,return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(select_class_name ,activation='softmax'))
    return model
input_shape =(Max_seq_length,Num_Features)
select_class_name  = len(label_types)
nn_model =get_sequence_model(input_shape,select_class_name)
nn_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
nn_model.summary()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = nn_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=Epochs, batch_size=Batch_size,callbacks=[early_stopping])
model_evaluation=nn_model.evaluate( x_test,y_test)

nn_model.save('Video_Classfication_model.h5')

def prepare_single_video(video_file_path, max_seq_length, label_types):
    Video_reader = cv2.VideoCapture(video_file_path)
    frames_list = []
    Video_frames_count = int(Video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(Video_frames_count / max_seq_length), 1)

    for frame_counter in range(max_seq_length):
        Video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = Video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (Image_size, Image_size))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    Video_reader.release()

    # Extract features
    features = []
    for frame in frames_list:
        frame = np.expand_dims(frame, axis=0)
        feature = feature_extractor.predict(frame)
        features.append(feature[0])

    features = np.array(features)
    if len(features) == max_seq_length:
        features_padded = pad_sequences([features], maxlen=max_seq_length, padding='post', truncating='post')
        predicted_labels_probabilities = nn_model.predict(features_padded)
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = label_types[predicted_label]

        print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[0][predicted_label]}')
input_video_file_path = "D:\\coding\\horse fitness\\Walking With Phone Stock Footage #Stock Footage.mp4"
prepare_single_video(input_video_file_path, Max_seq_length,label_types)
Video(input_video_file_path, embed=True, width=600)
os.system(f'start "" "{input_video_file_path}"')

