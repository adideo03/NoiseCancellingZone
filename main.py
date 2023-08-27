import os
import numpy as np
import pandas as pd
import librosa
import librosa.feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras import regularizers, callbacks
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, LeakyReLU, BatchNormalization
import tensorflow
from sklearn.preprocessing import MinMaxScaler
callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
def load_data(data_path, metadata_path):
    features = []
    labels = []


    metadata = pd.read_csv(metadata_path)


    for index, row in metadata.iterrows():
     file_path = os.path.join(data_path, f"fold{row['fold']}", f"{row['slice_file_name']}")


     # Load the audio file and resample it
     target_sr = 22050; factor = 0.4
     audio, sample_rate = librosa.load(file_path, sr=target_sr)
     audio_white_noise = audio + 0.009 * np.random.normal(0, 1, len(audio))
     audio_roll = np.roll(audio, int(target_sr / 10))
     audio_time_stch = librosa.effects.time_stretch(audio, rate=factor)
     audio_pitch_sf = librosa.effects.pitch_shift(audio, sr=target_sr, n_steps=-5)


     # Extract MFCC features
     mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
     mfccs_scaled = np.mean(mfccs.T, axis=0)
     mfccs_white_noise = librosa.feature.mfcc(y=audio_white_noise, sr=target_sr, n_mfcc=40)
     mfccs_wn_scaled = np.mean(mfccs_white_noise.T, axis=0)
     mfccs_roll = librosa.feature.mfcc(y=audio_roll, sr=target_sr, n_mfcc=40)
     mfccs_r_scaled = np.mean(mfccs_roll.T, axis=0)
     mfccs_ts = librosa.feature.mfcc(y=audio_time_stch, sr=target_sr, n_mfcc=40)
     mfccs_ts_scaled = np.mean(mfccs_ts.T, axis=0)
     mfccs_ps = librosa.feature.mfcc(y=audio_pitch_sf, sr=target_sr, n_mfcc=40)
     mfccs_ps_scaled = np.mean(mfccs_ps.T, axis=0)


     #Why do we need the mean of the mfccs?

     # Append features and labels
     features.append(mfccs_scaled)
     features.append(mfccs_wn_scaled)
     features.append(mfccs_r_scaled)
     features.append(mfccs_ts_scaled)
     features.append(mfccs_ps_scaled)
     labels.append(row['class'])
     labels.append(row['class'])
     labels.append(row['class'])
     labels.append(row['class'])
     labels.append(row['class'])


    return np.array(features), np.array(labels)

#
# data_path = "D:/Advay/SY BTech/EDI3/archive"
# metadata_path = "D:/Advay/SY BTech/EDI3/archive/UrbanSound8K.csv"
# features, labels = load_data(data_path, metadata_path)
# print(features)
# print(labels)
#
#
labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
'siren', 'street_music']
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
# labels_onehot = to_categorical(labels_encoded)
# conc_arr = np.concatenate((features, labels_onehot), axis=1)
#
# pd.DataFrame(conc_arr).to_csv("Augmented_data2.csv")
dataset = pd.read_csv("Augmented_data2.csv")
features = np.array(dataset.iloc[:, :41])
labels_onehot = np.array(dataset.iloc[:, 41:])

# feature scaling
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
print(f"Max val in 0th feature: {np.max(features[:,0])}")
print(features)
print(labels_onehot)
# print(np.argmax(labels_onehot, axis=1))
# X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.1, random_state=42, stratify=labels_onehot)
#
# Building the CNN model

input_shape = (int(features.shape[1]), 1)


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

kf = KFold(n_splits=10, shuffle=False)
acc = []
j=0
for train_idx, val_idx in kf.split(features):
    features_kf_train, features_kf_val = features[train_idx, :] , features[val_idx, :]
    labels_kf_train , labels_kf_val = labels_onehot[train_idx], labels_onehot[val_idx]
    features_kf_train = np.reshape(features_kf_train, (features_kf_train.shape[0], features_kf_train.shape[1], 1))
    features_kf_val = np.reshape(features_kf_val, (features_kf_val.shape[0], features_kf_val.shape[1], 1))
    model = Sequential()
    model.add(Conv1D(35, kernel_size=3, padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001),))
    model.add(LeakyReLU(alpha=0.05))
    #model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.25))
    model.add(Conv1D(70, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(LeakyReLU(alpha=0.05))
    #model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    #model.add(BatchNormalization())
    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001)))
    model.add(LeakyReLU(alpha=0.05))
    #model.add(Dropout(0.5))
    model.add(Dense(len(le.classes_), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(features_kf_train, labels_kf_train, batch_size=16, epochs=100, validation_data=(features_kf_val, labels_kf_val),
              use_multiprocessing=1)

    y_pred = model.predict(features_kf_val)
    labels_kf_val = np.argmax(labels_kf_val, axis=1)
    labels_pred = np.argmax(y_pred, axis=1)
    acc.append(accuracy_score(labels_kf_val, labels_pred))
    print(f"Fold {j} accuracy: {np.mean(acc)}")
    acc = []
    j += 1

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
