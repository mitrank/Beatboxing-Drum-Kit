from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from datetime import datetime as dt
from matplotlib import pyplot as plt
import tensorflow as tf
import librosa
import librosa.display as disp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

########## Feature Extraction Function ##########
def features_extractor(file_name):
  audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
  audio,_ = librosa.effects.trim(audio,top_db=15)
  mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
  mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
  return mfccs_scaled_features

########## Storing the kic, snare and hi-hats to the list as per their classes ##########
extracted_features = []
kick_path = 'audio/kick/'
for file_name in os.listdir(kick_path):
  if file_name.endswith('.wav'):
    f = os.path.join(kick_path, file_name)
  class_label = 'kick'
  data = features_extractor(f)
  extracted_features.append([data, class_label])
snare_path = 'audio/snare/'
for file_name in os.listdir(snare_path):
  if file_name.endswith('.wav'):
    f = os.path.join(snare_path, file_name)
  class_label = 'snare'
  data = features_extractor(f)
  extracted_features.append([data, class_label])
hat_path = 'audio/hats/'
for file_name in os.listdir(hat_path):
  if file_name.endswith('.wav'):
    f = os.path.join(hat_path, file_name)
  class_label = 'hi-hat'
  data = features_extractor(f)
  extracted_features.append([data, class_label])

extracted_features_df = pd.DataFrame(extracted_features, columns = ['feature', 'class'])
# print(extracted_features_df)

X = np.array(extracted_features_df['feature'].tolist())
Y = np.array(extracted_features_df['class'].tolist())
labelEncoder = LabelEncoder()
Y = to_categorical(labelEncoder.fit_transform(Y))
x_train, x_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=0)
# print(X.shape)
# print(Y.shape)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
num_labels = Y.shape[1]

########## Start making the ANN ##########
model = Sequential()
# First Layer
model.add(Dense(100, input_shape = (128, )))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Second Layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Third Layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Final Layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# model.summary()
model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

########## Training the model ##########
num_epochs = 70
num_batch_size = 5
checkPointer = ModelCheckpoint(filepath='audio/bb_classification.hdf5', verbose=1, save_best_only=True)
start = dt.now()
history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkPointer])
dur = dt.now() - start
print('Training completed in ', dur, ' seconds')

########## Test the accuracy ##########
test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test Accuracy: ', test_accuracy[1])

plt.plot(history.epoch, history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Value Accuracy')
plt.show()

file_name1 = 'audio/Kick-test.wav'
prediction_feature = features_extractor(file_name1)
prediction_feature = prediction_feature.reshape(1, -1)
prediction_feature = model.predict(prediction_feature)
classes_1 = np.argmax(prediction_feature, axis=-1)
print(labelEncoder.inverse_transform(classes_1))

file_name2 = 'audio/Snare-test.wav'
prediction_feature = features_extractor(file_name2)
prediction_feature = prediction_feature.reshape(1, -1)
prediction_feature = model.predict(prediction_feature)
classes_2 = np.argmax(prediction_feature, axis=-1)
print(labelEncoder.inverse_transform(classes_2))

file_name3 = 'audio/hihat-test.wav'
prediction_feature = features_extractor(file_name3)
prediction_feature = prediction_feature.reshape(1, -1)
prediction_feature = model.predict(prediction_feature)
classes_3 = np.argmax(prediction_feature, axis=-1)
print(labelEncoder.inverse_transform(classes_3))

