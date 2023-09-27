import os
import pyaudio
import keras.src.saving.legacy.hdf5_format
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
# Tentative classification of good and bad categories
classification = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])
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
        # print(audio.shape)

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
            while np.shape(mfccs_2d)[0] != 40:
                mfccs_2d = np.delete(mfccs_2d, np.shape(mfccs_2d)[0] - 1, axis=0)

            while np.shape(mfccs_wn_2d)[0] != 40:
                mfccs_wn_2d = np.delete(mfccs_wn_2d, np.shape(mfccs_wn_2d)[0] - 1, axis=0)

            while np.shape(mfccs_r_2d)[0] != 40:
                mfccs_r_2d = np.delete(mfccs_r_2d, np.shape(mfccs_r_2d)[0] - 1, axis=0)

            while np.shape(mfccs_ts_2d)[0] != 40:
                mfccs_ts_2d = np.delete(mfccs_ts_2d, np.shape(mfccs_ts_2d)[0] - 1, axis=0)

            while np.shape(mfccs_ps_2d)[0] != 40:
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

    return features, labels_onehot


def predict_binary(input):
    global model
    # pred = model.predict(np.expand_dims(input, axis=0))
    pred = model.predict(input)
    print('Shape of prediction: ', np.shape(pred))
    print(np.argmax(pred), ' ', label_names[np.argmax(pred)])
    y_pred_binary = np.empty([0, 1])
    prob_good_sum = 0
    prob_bad_sum = 0
    for i in range(pred.shape[1]):
        if classification[i] == 0:
            prob_good_sum += pred[0][i]
        else:
            prob_bad_sum += pred[0][i]
    if classification[np.argmax(pred)] == 0:  # Meaning the prediction belongs to the good category
        if np.max(pred) >= 0.4 or prob_good_sum - prob_bad_sum >= 0.4:
            y_pred_binary = np.insert(y_pred_binary, y_pred_binary.shape[0], np.array([0]), axis=0)

        else:
            y_pred_binary = np.insert(y_pred_binary, y_pred_binary.shape[0], np.array([1]), axis=0)

    else:
        if np.max(pred) >= 0.4 or prob_bad_sum - prob_good_sum >= 0.4:
            y_pred_binary = np.insert(y_pred_binary, y_pred_binary.shape[0], np.array([1]), axis=0)

        else:
            y_pred_binary = np.insert(y_pred_binary, y_pred_binary.shape[0], np.array([0]), axis=0)

    return y_pred_binary

def doFinal():
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 22050
    CHUNK_SIZE = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    sample = np.empty((0))
    features = np.empty((0, 40, 40))
    try:
        print("Recording... (Press Ctrl+C to stop)")
        i = 0
        input_array = np.empty((0))
        mfccs_2d = np.empty((0, 40))
        while True:
            audio_data = stream.read(CHUNK_SIZE)

            numerical_data = np.frombuffer(audio_data, dtype=np.float32)
            #print(numerical_data)
            sample = numerical_data
            if i<=80:                          # Replace with the correct value to make it 0.5 secs
                input_array = np.concatenate((input_array, numerical_data), axis=0)
                i+=1

            else:
                i=0
                segment_length = int(len(input_array) / 40)
                segments = [input_array[i:i + segment_length] for i in range(0, len(input_array), segment_length)]
                for segment in segments:
                    mfccs = librosa.feature.mfcc(y=segment, sr=RATE, n_mfcc=40)
                    mfccs_scaled = np.reshape(np.mean(mfccs.T, axis=0), (1, 40))
                    mfccs_2d = np.append(mfccs_2d, mfccs_scaled, axis=0)

                try:
                    features = np.insert(features, np.shape(features)[0], mfccs_2d, axis=0)
                except ValueError:
                    while np.shape(mfccs_2d)[0] != 40:
                        mfccs_2d = np.delete(mfccs_2d, np.shape(mfccs_2d)[0] - 1, axis=0)

                features = np.insert(features, np.shape(features)[0], mfccs_2d, axis=0)
                # print(features,'\n', features.shape)
                pred = predict_binary(features)
                if pred[0, 0].item()==0:
                    print('prediction: Good\n\n')
                else:
                    print('prediction: Bad\n\n')
                features = np.empty((0, 40, 40))
                mfccs_2d = np.empty((0, 40))
                input_array = np.empty((0))



    except KeyboardInterrupt:
        print("Recording stopped.")
        print(sample.shape)

    stream.stop_stream()
    stream.close()
    audio.terminate()



# Construct model

from keras.layers import Conv2D, GlobalAveragePooling2D, AvgPool2D

# model = Sequential()
# model.add(Conv2D(filters=24, kernel_size=2, input_shape=(num_rows, num_columns, num_channels),
#                  kernel_regularizer=regularizers.l2(0.001)))
# model.add(LeakyReLU(alpha=0.05))
# model.add(AvgPool2D(pool_size=2))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(filters=48, kernel_size=2, kernel_regularizer=regularizers.l2(0.001)))
# model.add(LeakyReLU(alpha=0.05))
# model.add(AvgPool2D(pool_size=2))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(filters=96, kernel_size=2, kernel_regularizer=regularizers.l2(0.001)))
# model.add(LeakyReLU(alpha=0.05))
# model.add(AvgPool2D(pool_size=2))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(filters=194, kernel_size=2, kernel_regularizer=regularizers.l2(0.001)))
# model.add(LeakyReLU(alpha=0.05))
# model.add(AvgPool2D(pool_size=2))
# model.add(Dropout(0.2))
#
# model.add(GlobalAveragePooling2D())
#
# model.add(Dense(num_labels, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = model.evaluate(x_test, y_test, verbose=1)
# accuracy = 100 * score[1]
#
# print("Pre-training accuracy: %.4f%%" % accuracy)
# from keras.callbacks import ModelCheckpoint
# from datetime import datetime
#
# num_epochs = 72
# num_batch_size = 256
#
# checkpointer = ModelCheckpoint(filepath='Best2DCNN1.hdf5',
#                                verbose=1, save_best_only=True)
# start = datetime.now()
#
# model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
#           callbacks=[checkpointer], verbose=1)
#
#
# duration = datetime.now() - start
# print("Training completed in time: ", duration)


#-------------------------------loading the best CNN and binary classification--------------------------------



from datetime import datetime
model = keras.src.saving.legacy.hdf5_format.load_model_from_hdf5('AvgPoolCNN.hdf5')
print('Model loaded')
doFinal()
