import os
import pyaudio
import keras.src.saving.legacy.hdf5_format
import numpy as np
import pandas as pd
import librosa
import librosa.feature
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from random import randint


label_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
               'jackhammer',
               'siren', 'street_music']
# Tentative classification of good and bad categories
classification = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])
le = LabelEncoder()
labels_encoded = np.reshape(le.fit_transform(label_names), [len(label_names), 1])

def load_data(data_path, metadata_path):
    features = np.empty((0, 40, 40))
    labels = []
    mfcc_means = np.empty((0, 40))
    transformation_labels = []

    metadata = pd.read_csv(metadata_path)
    count = 0
    for index, row in metadata.iterrows():
        count += 1
        file_path = os.path.join(data_path, f"fold{row['fold']}", f"{row['slice_file_name']}")

        # Load the audio file and resample it

        target_sr = 22050
        factor = 0.4
        audio, sample_rate = librosa.load(file_path, sr=target_sr)
        print(np.mean(audio))
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
        temp_means = np.empty((0))
        temp_means_wn = np.empty((0))
        temp_means_r = np.empty((0))
        temp_means_ts = np.empty((0))
        temp_means_ps = np.empty((0))
        for segment in segments:

            mfccs = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
            mfccs_scaled = np.reshape(np.mean(mfccs.T, axis=0), (1, 40))
            if mfccs_2d.shape[0] < 40:
                mfccs_2d = np.append(mfccs_2d, mfccs_scaled, axis=0)
            # print(mfccs_scaled, np.mean(mfccs_scaled))
            if temp_means.shape[0] < 40:
                temp_means = np.append(temp_means, np.mean(mfccs_scaled))



        segment_length = int(len(audio_white_noise) / 40)
        segments = [audio_white_noise[i:i + segment_length] for i in range(0, len(audio_white_noise), segment_length)]
        for segment in segments:
            mfccs_white_noise = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
            mfccs_wn_scaled = np.reshape(np.mean(mfccs_white_noise.T, axis=0), (1, 40))
            if mfccs_wn_2d.shape[0] < 40:
                mfccs_wn_2d = np.append(mfccs_wn_2d, mfccs_wn_scaled, axis=0)
            if temp_means_wn.shape[0] < 40:
                temp_means_wn = np.append(temp_means_wn, np.mean(mfccs_wn_scaled))


        segment_length = int(len(audio_roll) / 40)
        segments = [audio_roll[i:i + segment_length] for i in
                    range(0, len(audio_roll), segment_length)]
        for segment in segments:
            mfccs_roll = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
            mfccs_r_scaled = np.reshape(np.mean(mfccs_roll.T, axis=0), (1, 40))
            if mfccs_r_2d.shape[0] < 40:
                mfccs_r_2d = np.append(mfccs_r_2d, mfccs_r_scaled, axis=0)
            if temp_means_r.shape[0] < 40:
                temp_means_r = np.append(temp_means_r, np.mean(mfccs_r_scaled))



        segment_length = int(len(audio_time_stch) / 40)
        segments = [audio_time_stch[i:i + segment_length] for i in
                    range(0, len(audio_time_stch), segment_length)]
        for segment in segments:
            mfccs_ts = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
            mfccs_ts_scaled = np.reshape(np.mean(mfccs_ts.T, axis=0), (1, 40))
            if mfccs_ts_2d.shape[0] < 40:
                mfccs_ts_2d = np.append(mfccs_ts_2d, mfccs_ts_scaled, axis=0)
            if temp_means_ts.shape[0] < 40:
                temp_means_ts = np.append(temp_means_ts, np.mean(mfccs_ts_scaled))

        segment_length = int(len(audio_pitch_sf) / 40)
        segments = [audio_pitch_sf[i:i + segment_length] for i in
                    range(0, len(audio_pitch_sf), segment_length)]
        for segment in segments:
            mfccs_ps = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
            mfccs_ps_scaled = np.reshape(np.mean(mfccs_ps.T, axis=0), (1, 40))
            if mfccs_ps_2d.shape[0] < 40:
                mfccs_ps_2d = np.append(mfccs_ps_2d, mfccs_ps_scaled, axis=0)
            if temp_means_ps.shape[0] < 40:
                temp_means_ps = np.append(temp_means_ps, np.mean(mfccs_ps_scaled))


        features = np.insert(features, np.shape(features)[0], mfccs_2d, axis=0)
        features = np.insert(features, np.shape(features)[0], mfccs_wn_2d, axis=0)
        features = np.insert(features, np.shape(features)[0], mfccs_r_2d, axis=0)
        features = np.insert(features, np.shape(features)[0], mfccs_ts_2d, axis=0)
        features = np.insert(features, np.shape(features)[0], mfccs_ps_2d, axis=0)


        mfcc_means = np.insert(mfcc_means, mfcc_means.shape[0], temp_means, axis=0)
        mfcc_means = np.insert(mfcc_means, mfcc_means.shape[0], temp_means_wn, axis=0)
        mfcc_means = np.insert(mfcc_means, mfcc_means.shape[0], temp_means_r, axis=0)
        mfcc_means = np.insert(mfcc_means, mfcc_means.shape[0], temp_means_ts, axis=0)
        mfcc_means = np.insert(mfcc_means, mfcc_means.shape[0], temp_means_ps, axis=0)

        labels.append(row['class'])
        labels.append(row['class'])
        labels.append(row['class'])
        labels.append(row['class'])
        labels.append(row['class'])

        transformation_labels.append('raw')
        transformation_labels.append('wn')
        transformation_labels.append('r')
        transformation_labels.append('ts')
        transformation_labels.append('ps')
        # print("\n\nRaw: \tNegative values count: ", np.shape(np.where(temp_means < 0))[1],
        #       "\tMaximum Vale: ", np.max(temp_means), "\tMin Val: ", np.min(temp_means))
        #
        # print("\nWhite Noise: \tNegative values count: ", np.shape(np.where(temp_means_wn < 0))[1],
        #       "\tMaximum Vale: ", np.max(temp_means_wn), "\tMin Val: ", np.min(temp_means_wn))
        #
        # print("\nRolled: \tNegative values count: ", np.shape(np.where(temp_means_r < 0))[1],
        #       "\tMaximum Vale: ", np.max(temp_means_r), "\tMin Val: ", np.min(temp_means_r))
        #
        # print("\nTime Shift: \tNegative values count: ", np.shape(np.where(temp_means_ts < 0))[1],
        #       "\tMaximum Vale: ", np.max(temp_means_ts), "\tMin Val: ", np.min(temp_means_ts))
        #
        # print("\nPitch shift: \tNegative values count: ", np.shape(np.where(temp_means_ps < 0))[1],
        #       "\tMaximum Vale: ", np.max(temp_means_ps), "\tMin Val: ", np.min(temp_means_ps))
        if len(labels) != features.shape[0]:
            print(f"Sizes mismatched at {count+1}. Terminated.")
            return None, None
        print(f"{count+1} files done.\n")

    labels_encoded = le.transform(labels)
    labels_onehot = to_categorical(labels_encoded)
    transformation_labels_encoded = le.fit_transform(transformation_labels)
    transformation_labels_onehot = to_categorical(transformation_labels_encoded)
    np.save('Features.npy', features)
    np.save('Labels.npy', labels_onehot)
    np.save('MFCC_Means.npy', mfcc_means)
    pd.Series(transformation_labels).to_csv('Transformation_Labels.csv')
    np.save('Transformation_Labels_categorical.npy', transformation_labels_encoded)
    # np.save('Transformation_Labels.npy', transformation_labels)
    return features, labels_onehot



def load_human_speech_data(data_path, metadata_path, features_array, labels_onehot):
    features = features_array
    target_sr = 22050
    modified_labels = np.empty([0, labels_onehot.shape[1] + 1])
    for row in labels_onehot:
        modified_row = np.append(row, 0)
        modified_labels = np.insert(modified_labels, modified_labels.shape[0], modified_row, axis=0)
    print(labels_onehot.shape, modified_labels.shape)

    # We are only taking the first 10 folders to maintain class balance
    for folder in range(1, 11):
        for i in range(10):
            for j in range(50):
                file_path = f"{data_path}/0{folder}/{i}_0{folder}_{j}.wav"
                try:
                    audio, _ = librosa.load(file_path, sr=target_sr)
                except FileNotFoundError:
                    continue
                segment_length = int(len(audio) / 40)
                segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]
                # Extract MFCC features
                mfccs_2d = np.empty((0, 40))
                for segment in segments:
                    mfccs = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
                    mfccs_scaled = np.reshape(np.mean(mfccs.T, axis=0), (1, 40))
                    mfccs_2d = np.append(mfccs_2d, mfccs_scaled, axis=0)

                try:
                    idx = randint(0, features.shape[0] - 1)
                    features = np.insert(features, idx, mfccs_2d, axis=0)
                    label = np.concatenate((np.zeros(modified_labels.shape[1] - 1), np.array([1])), axis=0)
                    print(label)
                    modified_labels = np.insert(modified_labels, idx, label, axis=0)

                except ValueError:
                    while np.shape(mfccs_2d)[0] != 40:
                        mfccs_2d = np.delete(mfccs_2d, np.shape(mfccs_2d)[0] - 1, axis=0)

                    idx = randint(0, features.shape[0] - 1)
                    features = np.insert(features, idx, mfccs_2d, axis=0)
                    label = np.concatenate((np.zeros(modified_labels.shape[1] - 1), np.array([1])), axis=0)
                    modified_labels = np.insert(modified_labels, idx, label, axis=0)

                print(f"0{folder}/{i}_0{folder}_{j}.wav Done.")
    return features, modified_labels


def predict_binary(input, model):
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


def doFinal1():
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

            print("Mean of input audio: ", np.mean(numerical_data), "\n")
            sample = numerical_data
            if i <= 80:  # Replace with the correct value to make it 0.5 secs
                input_array = np.concatenate((input_array, numerical_data), axis=0)
                i += 1

            else:
                low_counter = 0
                i = 0
                segment_length = int(len(input_array) / 40)
                segments = [input_array[i:i + segment_length] for i in range(0, len(input_array), segment_length)]
                for segment in segments:
                    print(np.mean(segment))
                    if low_counter > 30:
                        break
                    if np.mean(segment) < 20:
                        low_counter += 1
                    mfccs = librosa.feature.mfcc(y=segment, sr=RATE, n_mfcc=40)
                    mfccs_scaled = np.reshape(np.mean(mfccs.T, axis=0), (1, 40))
                    mfccs_2d = np.append(mfccs_2d, mfccs_scaled, axis=0)

                if low_counter > 30:
                    print("Low amplitude sound, hence ignored.")
                    continue

                try:
                    features = np.insert(features, np.shape(features)[0], mfccs_2d, axis=0)
                except ValueError:
                    while np.shape(mfccs_2d)[0] != 40:
                        mfccs_2d = np.delete(mfccs_2d, np.shape(mfccs_2d)[0] - 1, axis=0)

                features = np.insert(features, np.shape(features)[0], mfccs_2d, axis=0)
                # print(features,'\n', features.shape)
                pred = predict_binary(features)
                if pred[0, 0].item() == 0:
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


def load_labels(data_path, metadata_path):
    labels = []

    metadata = pd.read_csv(metadata_path)

    for index, row in metadata.iterrows():
        labels.append(row['class'])
        labels.append(row['class'])
        labels.append(row['class'])
        labels.append(row['class'])
        labels.append(row['class'])
    labels_enocded = le.transform(labels)
    labels_onehot = to_categorical(labels_enocded)
    np.save('Labels.npy', labels_onehot)


def doFinal(model):
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 22050
    CHUNK_SIZE = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    stream_out = audio.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=RATE,
                        output=True,
                        frames_per_buffer=CHUNK_SIZE)

    features = np.empty((1, 40, 40))
    try:
        print("Recording... (Press Ctrl+C to stop)")
        i = 0
        input_array = np.empty((0))
        mfccs_2d = np.empty((40, 40))
        pred_history = [0, 0]
        while True:
            audio_data = stream.read(CHUNK_SIZE)
            numerical_data = np.frombuffer(audio_data, dtype=np.float32)
            if pred_history[0] == 1 and pred_history[1] == 1:
            # Invert the audio
                inverted_audio = -10 * numerical_data

            # Convert the inverted audio back to bytes
                inverted_data = inverted_audio.tobytes()

            # Play the inverted audio
                stream_out.write(inverted_data)

            # print("Mean of input audio: ", np.mean(numerical_data), "\n")
            if i <= 80:  # Replace with the correct value to make it 0.5 secs
                input_array = np.concatenate((input_array, numerical_data), axis=0)
                i += 1

            else:
                input_array *= 10
                print("Mean of input audio: ", np.mean(input_array))
                low_counter = 0
                i = 0
                segment_length = int(len(input_array) / 40)
                segments = [input_array[i:i + segment_length] for i in range(0, len(input_array), segment_length)]

                k = 0
                for segment in segments:
                    # print(np.mean(segment))
                    # if low_counter > 30:
                    #     break
                    # if np.mean(segment) < 20:
                    #     low_counter += 1
                    mfccs = librosa.feature.mfcc(y=segment, sr=RATE, n_mfcc=40)
                    mfccs_scaled = np.reshape(np.mean(mfccs.T, axis=0), (1, 40))
                    if k < 40:
                        mfccs_2d[k] = mfccs_scaled
                    k+=1

                # if low_counter > 30:
                #     print("Low amplitude sound, hence ignored.")
                #     continue

                try:
                    features[0] = mfccs_2d
                except ValueError:
                    while np.shape(mfccs_2d)[0] != 40:
                        mfccs_2d = np.delete(mfccs_2d, np.shape(mfccs_2d)[0] - 1, axis=0)
                finally:
                    features[0] = mfccs_2d
                # print(features,'\n', features.shape)
                pred = predict_binary(features, model)
                if pred[0, 0].item() == 0:
                    print('prediction: Good\n\n')
                    pred_history[0], pred_history[1] = pred_history[1], 0
                else:
                    print('prediction: Bad\n\n')
                    pred_history[0], pred_history[1] = pred_history[1], 1
                features = np.empty((1, 40, 40))
                mfccs_2d = np.empty((40, 40))
                input_array = np.empty((0))



    except KeyboardInterrupt:
        print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    stream_out.stop_stream()
    stream_out.close()
    audio.terminate()


