import os
import numpy as np
import pandas as pd
import librosa
import librosa.feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras import regularizers, callbacks
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, LeakyReLU, BatchNormalization, \
    AvgPool1D
from tensorflow import losses
from sklearn.preprocessing import MinMaxScaler

callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
label_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
          'jackhammer',
          'siren', 'street_music']
le = LabelEncoder()
labels_encoded = np.reshape(le.fit_transform(label_names), [len(label_names), 1])
OHEncoder = OneHotEncoder()
OHEncoder.fit(labels_encoded)


def load_data(data_path, metadata_path):
    features = np.empty((0, 40, 40))
    labels = []

    metadata = pd.read_csv(metadata_path)

    for index, row in metadata.iterrows():
        file_path = os.path.join(data_path, f"fold{row['fold']}", f"{row['slice_file_name']}")

        # Load the audio file and resample it

        target_sr = 22050
        factor = 0.4
        audio, sample_rate = librosa.load(file_path, sr=target_sr)

        audio_white_noise = audio + 0.009 * np.random.normal(0, 1, len(audio))
        audio_roll = np.roll(audio, int(target_sr / 10))
        audio_time_stch = librosa.effects.time_stretch(audio, rate=factor)
        audio_pitch_sf = librosa.effects.pitch_shift(audio, sr=target_sr, n_steps=-5)

        segment_length = int(len(audio) / 40)
        segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]
        # Extract MFCC features
        mfccs_2d = np.empty((0, 40))
        mfccs_wn_2d = np.empty((0, 40))
        mfccs_r_2d = np.empty((0, 40))
        mfccs_ts_2d = np.empty((0, 40))
        mfccs_ps_2d = np.empty((0, 40))
        for segment in segments:
            mfccs = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
            mfccs_scaled = np.reshape(np.mean(mfccs.T, axis=0), (1, 40))
            mfccs_2d = np.append(mfccs_2d, mfccs_scaled, axis=0)

            mfccs_white_noise = librosa.feature.mfcc(y=audio_white_noise, sr=target_sr, n_mfcc=40)
            mfccs_wn_scaled = np.reshape(np.mean(mfccs_white_noise.T, axis=0), (1, 40))
            mfccs_wn_2d = np.append(mfccs_wn_2d, mfccs_wn_scaled, axis=0)

            mfccs_roll = librosa.feature.mfcc(y=audio_roll, sr=target_sr, n_mfcc=40)
            mfccs_r_scaled = np.reshape(np.mean(mfccs_roll.T, axis=0), (1, 40))
            mfccs_r_2d = np.append(mfccs_r_2d, mfccs_r_scaled, axis=0)

            mfccs_ts = librosa.feature.mfcc(y=audio_time_stch, sr=target_sr, n_mfcc=40)
            mfccs_ts_scaled = np.reshape(np.mean(mfccs_ts.T, axis=0), (1, 40))
            mfccs_ts_2d = np.append(mfccs_ts_2d, mfccs_ts_scaled, axis=0)

            mfccs_ps = librosa.feature.mfcc(y=audio_pitch_sf, sr=target_sr, n_mfcc=40)
            mfccs_ps_scaled = np.reshape(np.mean(mfccs_ps.T, axis=0), (1, 40))
            mfccs_ps_2d = np.append(mfccs_ps_2d, mfccs_ps_scaled, axis=0)

        try:
            features = np.insert(features, np.shape(features)[0], mfccs_2d, axis=0)
            features = np.insert(features, np.shape(features)[0], mfccs_wn_2d, axis=0)
            features = np.insert(features, np.shape(features)[0], mfccs_r_2d, axis=0)
            features = np.insert(features, np.shape(features)[0], mfccs_ts_2d, axis=0)
            features = np.insert(features, np.shape(features)[0], mfccs_ps_2d, axis=0)

        except ValueError:
            while np.shape(mfccs_2d)[0]!=40:
                mfccs_2d = np.delete(mfccs_2d, np.shape(mfccs_2d)[0] - 1, axis=0)

            while np.shape(mfccs_wn_2d)[0]!=40:
                mfccs_wn_2d = np.delete(mfccs_wn_2d, np.shape(mfccs_wn_2d)[0] - 1, axis=0)

            while np.shape(mfccs_r_2d)[0]!=40:
                mfccs_r_2d = np.delete(mfccs_r_2d, np.shape(mfccs_r_2d)[0] - 1, axis=0)

            while np.shape(mfccs_ts_2d)[0]!=40:
                mfccs_ts_2d = np.delete(mfccs_ts_2d, np.shape(mfccs_ts_2d)[0] - 1, axis=0)

            while np.shape(mfccs_ps_2d)[0]!=40:
                mfccs_ps_2d = np.delete(mfccs_ps_2d, np.shape(mfccs_ps_2d)[0] - 1, axis=0)

            features = np.insert(features, np.shape(features)[0], mfccs_2d, axis=0)
            features = np.insert(features, np.shape(features)[0], mfccs_wn_2d, axis=0)
            features = np.insert(features, np.shape(features)[0], mfccs_r_2d, axis=0)
            features = np.insert(features, np.shape(features)[0], mfccs_ts_2d, axis=0)
            features = np.insert(features, np.shape(features)[0], mfccs_ps_2d, axis=0)



        labels.append(row['class'])
        labels.append(row['class'])
        labels.append(row['class'])
        labels.append(row['class'])
        labels.append(row['class'])

    labels_encoded = le.transform(labels)
    labels_onehot = to_categorical(labels_encoded)
    np.save('Test3DArray1.npy', features)
    np.save('TestLabelsArray.npy', labels_onehot)

    # return features, labels_onehot


val_loss = 1
train_loss = 0.001


def custom_loss(y_true, y_pred):
    penalty_coefficient = 0.001
    cross_entropy_loss = losses.categorical_crossentropy(y_true, y_pred)
    return cross_entropy_loss

features = np.load('Test3DArray1.npy', allow_pickle=True)
labels = np.load('TestLabelsArray.npy', allow_pickle=True)
print('Data Loaded')

X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.1, random_state=42, stratify=labels_onehot)
num_rows, num_columns, num_channels = 40, 40, 1
x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = labels.shape[1]
filter_size = 2

# Construct model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)
from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 72
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='Best2DCNN.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
          callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

#-------------------------Ignore the part below--------------------------


# Outlier detection and removal
# from sklearn.ensemble import IsolationForest
# iforest = IsolationForest(random_state=0, n_estimators=100)
# iforest.fit(features)
# pos = iforest.predict(features)
# pos = np.asarray(np.where((pos==-1)), dtype='int')
# # print(np.shape(pos))
# features = np.delete(features, pos, 0)
# labels_onehot = np.delete(labels_onehot , pos, 0)
# print(f"Outliers: {np.shape(np.where(iforest.predict(features)==-1))}")


# Building the CNN model

# input_shape = (int(features.shape[1]), 1)
#
#
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
# from keras.optimizers import AdamW
# opt = AdamW()
#
# kf = KFold(n_splits=10, shuffle=False)
# acc = []
# j=0
# for train_idx, val_idx in kf.split(features):
#     features_kf_train, features_kf_val = features[train_idx, :] , features[val_idx, :]
#     labels_kf_train , labels_kf_val = labels_onehot[train_idx], labels_onehot[val_idx]
#     features_kf_train = np.reshape(features_kf_train, (features_kf_train.shape[0], features_kf_train.shape[1], 1))
#     features_kf_val = np.reshape(features_kf_val, (features_kf_val.shape[0], features_kf_val.shape[1], 1))
#
#
#     model = Sequential()
#     model.add(Conv1D(56, kernel_size=4, padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001),))
#     model.add(LeakyReLU(alpha=0.05))
#     model.add(BatchNormalization())
#     model.add(AvgPool1D(pool_size=2))
#     #model.add(Dropout(0.25))
#     model.add(Conv1D(96, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001)))
#     model.add(LeakyReLU(alpha=0.05))
#     model.add(BatchNormalization())
#     model.add(AvgPool1D(pool_size=2))
#     model.add(Conv1D(96, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001)))
#     model.add(LeakyReLU(alpha=0.05))
#     model.add(BatchNormalization())
#     model.add(AvgPool1D(pool_size=2))
#     #model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(BatchNormalization())
#     model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001)))
#     model.add(LeakyReLU(alpha=0.05))
#     #model.add(Dropout(0.5))
#     model.add(Dense(len(le.classes_), activation='softmax'))
#
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit(features_kf_train, labels_kf_train, batch_size=50, epochs=100,
#               validation_data=(features_kf_val, labels_kf_val),
#               use_multiprocessing=1)
#
#     y_pred = model.predict(features_kf_val)
#
#     labels_kf_val = np.argmax(labels_kf_val, axis=1)
#     labels_pred = np.argmax(y_pred, axis=1)
#     acc.append(accuracy_score(labels_kf_val, labels_pred))
#     print(f"Fold {j} accuracy: {np.mean(acc)}")
#     acc = []
#     j += 1

#
#
# from keras.layers import LSTM
#
# model = Sequential()
# model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
# model.add(LeakyReLU(alpha=0.05))
# model.add(LSTM(128, return_sequences=True))
# model.add(LeakyReLU(alpha=0.05))
# model.add(Flatten())
# model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001)))
# model.add(LeakyReLU(alpha=0.05))
# model.add(Dense(len(le.classes_), activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(features_kf_train, labels_kf_train, batch_size=16, epochs=100,
#           validation_data=(features_kf_val, labels_kf_val),
#           use_multiprocessing=1)

# custom training
# num_epochs = 100
# for epoch in range(num_epochs):
#     # Training step
#     train_loss, train_accuracy = model.train_on_batch(features_kf_train, labels_kf_train)
#
#     # Validation step
#     val_loss, val_accuracy = model.evaluate(features_kf_val, labels_kf_val)
#
#     print(
#         f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Training accuracy: {train_accuracy}"
#         f" Validation Loss: {val_loss}, Validation accuracy: {val_accuracy}")

