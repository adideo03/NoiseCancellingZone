import os
from collections import deque
from datetime import datetime
from random import randint
import librosa
import librosa.feature
import numpy as np
import pandas as pd
import pyaudio
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical
from scipy.signal import wiener
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

label_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
               'jackhammer',
               'siren', 'street_music']
# Tentative classification of good and bad categories
classification = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])
le = LabelEncoder()
labels_encoded = np.reshape(le.fit_transform(label_names), [len(label_names), 1])
noisy_sounds = [
    # Add noisy classes here
    'Music', 'Alarm', 'Drill', 'Jackhammer', 'Sawing', 'Power tool', 'Engine', 'Heavy engine (low frequency)',
    'Engine knocking',
    'Idling', 'Accelerating, revving, vroom', 'Vehicle horn, car horn, honking', 'Skidding', 'Tire squeal',
    'Car passing by', 'Race car, auto racing', 'Truck', 'Air brake', 'Air horn, truck horn', 'Reversing beeps',
    'Ice cream truck, ice cream van', 'Bus', 'Emergency vehicle', 'Police car (siren)', 'Ambulance (siren)',
    'Fire engine, fire truck (siren)', 'Motorcycle', 'Traffic noise, roadway noise', 'Train whistle', 'Train horn',
    'Aircraft engine', 'Jet engine', 'Propeller, airscrew', 'Helicopter', 'Fixed-wing aircraft, airplane',
    'Bicycle bell', 'Motorboat, speedboat', 'Ship', 'Lawn mower', 'Chainsaw', 'Dental drill, dentist\'s drill',
    'Power tool', 'Jackhammer', 'Sawing', 'Chopping (food)', 'Frying (food)', 'Microwave oven', 'Blender',
    'Water tap, faucet', 'Sink (filling or washing)', 'Bathtub (filling or washing)', 'Hair dryer', 'Toilet flush',
    'Vacuum cleaner', 'Keys jangling', 'Coin (dropping)', 'Scissors', 'Electric shaver, electric razor', 'Typewriter',
    'Computer keyboard', 'Writing', 'Alarm clock', 'Siren', 'Civil defense siren', 'Buzzer',
    'Smoke detector, smoke alarm',
    'Fire alarm', 'Foghorn', 'Whistle', 'Steam whistle', 'Mechanisms', 'Ratchet, pawl', 'Clock', 'Gears', 'Pulleys',
    'Sewing machine', 'Mechanical fan', 'Air conditioning', 'Cash register', 'Printer', 'Camera', 'Hammer',
    'Power tool', 'Drill', 'Explosion', 'Gunshot, gunfire', 'Machine gun', 'Fusillade', 'Artillery fire', 'Cap gun',
    'Fireworks', 'Firecracker', 'Burst, pop', 'Eruption', 'Boom', 'Wood', 'Chop', 'Splinter', 'Crack', 'Glass',
    'Chink, clink', 'Shatter', 'Splash, splatter', 'Slosh', 'Squish', 'Drip', 'Pour', 'Trickle, dribble', 'Gush',
    'Fill (with liquid)', 'Spray', 'Pump (liquid)', 'Stir', 'Boiling', 'Arrow', 'Whoosh, swoosh, swish', 'Thump, thud',
    'Thunk', 'Basketball bounce', 'Bang', 'Slap, smack', 'Whack, thwack', 'Smash, crash', 'Breaking', 'Bouncing',
    'Whip', 'Flap', 'Scratch', 'Scrape', 'Rub', 'Roll', 'Crushing', 'Crumpling, crinkling', 'Tearing', 'Beep, bleep',
    'Ping', 'Ding', 'Clang', 'Squeal', 'Creak', 'Rustle', 'Whir', 'Clatter', 'Sizzle', 'Clicking', 'Clickety-clack',
    'Rumble', 'Plop', 'Jingle, tinkle', 'Hum', 'Zing', 'Boing', 'Crunch']


def load_data(data_path, metadata_path):
    features = np.empty((0, 40, 40))
    labels = []
    mfcc_means = np.empty((0, 40))
    transformation_labels = []

    metadata = pd.read_csv(metadata_path)
    count = 0
    # audio = pyaudio.PyAudio()
    # stream_out = audio.open(format=pyaudio.paFloat32,
    #                         channels=1,
    #                         rate=22050,
    #                         output=True,
    #                         frames_per_buffer=1024)
    for index, row in metadata.iterrows():
        count += 1
        file_path = os.path.join(data_path, f"fold{row['fold']}", f"{row['slice_file_name']}")

        # Load the audio file and resample it

        target_sr = 16000
        factor = 0.4
        audio, sample_rate = librosa.load(file_path, sr=target_sr, dtype=np.float32)

        # stream_out.write(audio.tobytes())
        print(audio.shape)

        audio_white_noise = audio + 0.009 * np.random.normal(0, 1, len(audio))
        audio_roll = np.roll(audio, int(target_sr / 10))
        audio_time_stch = librosa.effects.time_stretch(audio, rate=factor)
        audio_pitch_sf = librosa.effects.pitch_shift(audio, sr=target_sr, n_steps=-5)
        #
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

        segment_length = int(len(audio_white_noise) / 40)
        segments = [audio_white_noise[i:i + segment_length] for i in range(0, len(audio_white_noise), segment_length)]
        for segment in segments:
            mfccs_white_noise = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
            mfccs_wn_scaled = np.reshape(np.mean(mfccs_white_noise.T, axis=0), (1, 40))
            if mfccs_wn_2d.shape[0] < 40:
                mfccs_wn_2d = np.append(mfccs_wn_2d, mfccs_wn_scaled, axis=0)
            # if temp_means_wn.shape[0] < 40:
            #     temp_means_wn = np.append(temp_means_wn, np.mean(mfccs_wn_scaled))

        segment_length = int(len(audio_roll) / 40)
        segments = [audio_roll[i:i + segment_length] for i in
                    range(0, len(audio_roll), segment_length)]
        for segment in segments:
            mfccs_roll = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
            mfccs_r_scaled = np.reshape(np.mean(mfccs_roll.T, axis=0), (1, 40))
            if mfccs_r_2d.shape[0] < 40:
                mfccs_r_2d = np.append(mfccs_r_2d, mfccs_r_scaled, axis=0)
            # if temp_means_r.shape[0] < 40:
            #     temp_means_r = np.append(temp_means_r, np.mean(mfccs_r_scaled))

        segment_length = int(len(audio_time_stch) / 40)
        segments = [audio_time_stch[i:i + segment_length] for i in
                    range(0, len(audio_time_stch), segment_length)]
        for segment in segments:
            mfccs_ts = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
            mfccs_ts_scaled = np.reshape(np.mean(mfccs_ts.T, axis=0), (1, 40))
            if mfccs_ts_2d.shape[0] < 40:
                mfccs_ts_2d = np.append(mfccs_ts_2d, mfccs_ts_scaled, axis=0)
            # if temp_means_ts.shape[0] < 40:
            #     temp_means_ts = np.append(temp_means_ts, np.mean(mfccs_ts_scaled))

        segment_length = int(len(audio_pitch_sf) / 40)
        segments = [audio_pitch_sf[i:i + segment_length] for i in
                    range(0, len(audio_pitch_sf), segment_length)]
        for segment in segments:
            mfccs_ps = librosa.feature.mfcc(y=segment, sr=target_sr, n_mfcc=40)
            mfccs_ps_scaled = np.reshape(np.mean(mfccs_ps.T, axis=0), (1, 40))
            if mfccs_ps_2d.shape[0] < 40:
                mfccs_ps_2d = np.append(mfccs_ps_2d, mfccs_ps_scaled, axis=0)
            # if temp_means_ps.shape[0] < 40:
            #     temp_means_ps = np.append(temp_means_ps, np.mean(mfccs_ps_scaled))

        features = np.insert(features, np.shape(features)[0], mfccs_2d, axis=0)
        features = np.insert(features, np.shape(features)[0], mfccs_wn_2d, axis=0)
        features = np.insert(features, np.shape(features)[0], mfccs_r_2d, axis=0)
        features = np.insert(features, np.shape(features)[0], mfccs_ts_2d, axis=0)
        features = np.insert(features, np.shape(features)[0], mfccs_ps_2d, axis=0)
        print(f"{count + 1} files done.\n")

    return features


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
                file_path = f"{data_path}/0{folder}/{i}0{folder}{j}.wav"
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

                print(f"0{folder}/{i}0{folder}{j}.wav Done.")
    return features, modified_labels


def predict_binary(input, model):
    # pred = model.predict(np.expand_dims(input, axis=0))
    pred = model.predict(input)
    # print(np.argmax(pred), ' ', label_names[np.argmax(pred)])

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

    return y_pred_binary, np.max(pred)


def doFinal1():
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 22050
    chunkSize = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=chunkSize)
    sample = np.empty((0))
    features = np.empty((0, 40, 40))
    try:
        print("Recording... (Press Ctrl+C to stop)")
        i = 0
        input_array = np.empty((0))
        mfccs_2d = np.empty((0, 40))
        while True:
            audio_data = stream.read(chunkSize)

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


class NumpyArrayQueue:
    def __init__(self):
        self.queue = deque(maxlen=5)  # change when necessary

    def enqueue(self, numpy_array):
        # Add a NumPy array to the end of the deque
        self.queue.append(numpy_array)

    def dequeue(self):
        # Remove and return the NumPy array from the front of the deque
        return self.queue.popleft()

    def is_empty(self):
        # Check if the deque is empty
        return len(self.queue) == 0

    def get_length(self):
        return len(self.queue)

    def peek(self, index):
        # Access the NumPy array at the specified index without dequeuing
        return self.queue[index]


def plot_waveform(data_list, title_list):
    plt.figure(figsize=(12, 6))

    for i, data in enumerate(data_list):
        plt.plot(data, label=title_list[i])

    plt.title('Audio Waveforms')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


def plot_final(input, output, i):
    plt.close()
    plt.figure(figsize=(12, 6))
    print("Input: ", input)
    print("Output: ", output)
    plt.plot(np.arange(input.shape[0]), input, color='blue')

    # Plot the second array in red
    plt.plot(np.arange(output.shape[0]), output, color='red')
    plt.legend()
    plt.savefig(f'plot{i}.png')
    plt.show()


ratio = 1


def expt_plot(input, closest_array, output):
    plt.close()
    plt.figure(figsize=(12, 6))
    print("Input: ", input)
    print("Output: ", output)
    plt.plot(np.arange(input.shape[0]), input, color='blue')

    plt.plot(np.arange(closest_array.shape[0]), closest_array, color='green')
    # Plot the second array in red
    plt.plot(np.arange(output.shape[0]), output, color='red')
    plt.legend()
    plt.show()


def doFinal(model):
    # Load YAMNet model
    yamnet_model_handle = 'https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1'
    model = hub.load(yamnet_model_handle)
    print('YAMNet model loaded.')
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    class_names = list(pd.read_csv(class_map_path)['display_name'])

    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    chunkSize = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=chunkSize)
    stream_out = audio.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=RATE,
                            output=True,
                            frames_per_buffer=chunkSize)

    features = np.empty((1, 40, 40))
    recent_arrays = NumpyArrayQueue()
    input_queue = [None, None]
    parent_array_queue = [None, None]
    inverted_queue = [None, None]
    try:
        print("Recording...")
        i = 0
        input_array = np.empty((1024 * 87), dtype=np.float32)  # change to 1024*80 when needed
        mfccs_2d = np.empty((40, 40))
        pred_history = 0
        pos = 0
        while True:
            num_plots = 0
            start_time = datetime.now()
            audio_data = stream.read(chunkSize)
            end_time = datetime.now()
            difference = end_time - start_time
            global ratio
            try:
                ratio = 1024 / difference.total_seconds()
            except ZeroDivisionError:
                # print('ZeroDivisionError')
                pass
            start_time = datetime.now()
            numerical_data = np.frombuffer(audio_data, dtype=np.float32)
            max_similarity = -1
            max_index = []
            input_queue[0], input_queue[1] = input_queue[1], numerical_data

            if pred_history == 1:  # Meaning that the last sound is bad
                # Calculate similarity between all subparts of the recent arrays and the input array
                if recent_arrays.get_length() > 0:
                    for array in range(recent_arrays.get_length()):
                        for j in range(87):
                            similarity = cosine_similarity(np.expand_dims(numerical_data, axis=0),
                                                           np.expand_dims(recent_arrays.peek(array)[j: j + 1024],
                                                                          axis=0))[0, 0]
                            if similarity > max_similarity:
                                max_index = [array, j]

                    j = max_index[1]
                try:
                    end_time = datetime.now()

                    calc_time = (end_time - start_time).total_seconds()
                    ratio = 87 * recent_arrays.get_length() * 1024 / calc_time
                    delay = int(ratio * calc_time)
                    print('Delay: ', delay)
                    # Invert the part of the array that follows the part with the maximum similarity
                    inverted_audio = -1 * recent_arrays.peek(max_index[0])[
                                          j + delay: j + delay + 1024]  # give the delay here by
                    # inverted_audio = -input_array
                    print('Inverted_audio: ', inverted_audio)

                    parent_array_queue[0], parent_array_queue[1] = parent_array_queue[1], recent_arrays.peek(
                        max_index[0])
                    inverted_queue[0], inverted_queue[1] = inverted_queue[1], inverted_audio

                    # This was to check whch part of the parent array it matched the most but it's not the right way

                    try:
                        if input_queue[0] is not None:
                            expt_plot(input_queue[1], -1 * parent_array_queue[0], inverted_queue[0])
                    except TypeError:
                        print('Type error')
                        pass
                    # plot_final(numerical_data, inverted_audio)

                    # We need to take the inverted audio and maybe also the parent array and plot it with the next input
                    # How do we determine the delay still?


                except IndexError:
                    print('Max similarity sound out of range.')

            # print("Mean of input audio: ", np.mean(numerical_data), "\n")
            if i < 87:  # Replace with the correct value to make it 0.5 secs
                input_array[pos: pos + 1024] = numerical_data
                i += 1
                pos += 1024

            else:
                recent_arrays.enqueue(input_array)

                low_counter = 0
                i = 0
                segment_length = int(len(input_array) / 40)
                segments = [input_array[i:i + segment_length] for i in range(0, len(input_array), segment_length)]

                k = 0
                for segment in segments:
                    mfccs = librosa.feature.mfcc(y=segment, sr=RATE, n_mfcc=40)
                    mfccs_scaled = np.reshape(np.mean(mfccs.T, axis=0), (1, 40))
                    if k < 40:
                        mfccs_2d[k] = mfccs_scaled
                    k += 1
                # print("\nMeans of MFCC features of segments:\n", np.mean(mfccs_2d, axis=0))

                try:
                    features[0] = mfccs_2d
                except ValueError:
                    while np.shape(mfccs_2d)[0] != 40:
                        mfccs_2d = np.delete(mfccs_2d, np.shape(mfccs_2d)[0] - 1, axis=0)
                finally:
                    features[0] = mfccs_2d
                # print(features,'\n', features.shape)
                pred, prob = predict_binary(features, model)
                global noisy_sounds
                scores, embeddings, spectrogram = model(input_array)
                class_scores = tf.reduce_mean(scores, axis=0)
                top_class_yamnet = tf.math.argmax(class_scores)
                inferred_class_yamnet = class_names[top_class_yamnet]
                top_score_yamnet = class_scores[top_class_yamnet]

                if inferred_class_yamnet in noisy_sounds:
                    # If YAMnet says bad and the custom model says good:
                    # if pred[0, 0].item() == 0:
                    #     if top_score_yamnet * 521 - prob * 10 > 0:
                    print(f'prediction: Bad  {inferred_class_yamnet}\n\n')
                    pred_history = 1


                else:
                    print(f'prediction: Good  {inferred_class_yamnet}\n\n')
                    pred_history = 0
                    # continue
                    # quit()

                features = np.empty((1, 40, 40))
                mfccs_2d = np.empty((40, 40))
                input_array = np.empty((1024 * 87), dtype=np.float32)  # change to 1024*80 when needed
                pos = 0

    except KeyboardInterrupt:
        print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    stream_out.stop_stream()
    stream_out.close()
    audio.terminate()

    # except KeyboardInterrupt:
    #     print("Recording stopped.")

    # stream.stop_stream()
    # stream.close()
    # stream_out.stop_stream()
    # stream_out.close()
    # audio.terminate()


from annoy import AnnoyIndex
import numpy as np


def initialize_index(dimension):
    # Specify the number of dimensions for the Annoy index
    annoy_index = AnnoyIndex(dimension, 'angular')  # Use 'angular' metric for cosine similarity
    return annoy_index


def update_index(annoy_index, new_data_point, max_size, save_interval):
    # Add the new data to the Annoy index
    index_position = annoy_index.get_n_items()
    annoy_index.add_item(index_position, new_data_point.flatten())

    # Check if it's time to manage the index size
    if index_position % save_interval == 0:
        # Save the index to a temporary file
        annoy_index.save('temp_index.ann')

        # Create a new Annoy index and load from the saved file
        new_annoy_index = AnnoyIndex(annoy_index.f, 'angular')
        new_annoy_index.load('temp_index.ann')

        # Trim the new index to the desired size
        # new_annoy_index.build(int(max_size))

        # Update the original index with the new one
        annoy_index = new_annoy_index


import numpy as np
from annoy import AnnoyIndex


def convert_and_trim_index(annoy_index, max_size):
    # Check if trimming is required

    index_array = np.array([annoy_index.get_item_vector(i) for i in range(annoy_index.get_n_items())])
    # Trim the array to the desired size

    trimmed_array = index_array[index_array.shape[0] - max_size: ]

    # Create a new Annoy index
    new_annoy_index = AnnoyIndex(len(trimmed_array[0]), 'angular')

    # Add vectors from the trimmed array to the new index
    for i, vector in enumerate(trimmed_array):
        new_annoy_index.add_item(i, vector)

    # Build the new index
    new_annoy_index.build(int(max_size))
    return new_annoy_index


def print_neighbors_stats(annoy_index, new_data_point, k):
    # Query for approximate nearest neighbors
    indices = annoy_index.get_nns_by_vector(vector=new_data_point.flatten(), n=k)

    # Retrieve the data of the neighbor with the highest similarity
    for neighbor in indices:
        neighbor_data = annoy_index.get_item_vector(neighbor)

        # Print the mean and std of the retrieved neighbor
        print("Mean of Retrieved Neighbor:", np.mean(neighbor_data))
        print("Std of Retrieved Neighbor:", np.std(neighbor_data))


# Example continuous update loop
dimension = 1024  # Adjust based on your data dimensionality
max_index_size = 1000  # Maximum number of items in the index
index = initialize_index(dimension)

window_size = 1000  # Adjust the window size as needed
# for i in range(1000):
#     # Receive new data
#     new_data_point = np.random.rand(dimension).astype('float32')
#     index.add_item(i=i, vector=new_data_point)
# index.build(100, -1)
#
# for i in range(100):
#     # Receive new data
#     new_data_point = np.random.rand(1, dimension).astype('float32')
#     if index.get_n_items() > max_index_size + 80:
#         index = convert_and_trim_index(index, max_index_size)
#     # update_index(index, new_data_point, max_index_size, 10)
#
# for i in range(10):
#     new_data_point = np.random.rand(1, dimension).astype('float32')
#     print('Mean of new data point: ', np.mean(new_data_point), 'Std: ', np.std(new_data_point),'\n\n')
#
#     # Print mean and std of neighbors with the highest similarity
#     print_neighbors_stats(index, new_data_point, k=5)
#     print('\n\n================================')


def doFinalFastSearching():
    # Load YAMNet model
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    # yamnet_model_handle = 'https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1'
    model = hub.load(yamnet_model_handle)
    # model = tf.saved_model.load('D:/Advay/SY BTech/EDI3 Folder/archive.tar/archive/saved_model.pb/1/')
    print('YAMNet model loaded.')
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    class_names = list(pd.read_csv(class_map_path)['display_name'])
    
    # Constants
    maxNumArrays = 5
    numChunks = 87
    format = pyaudio.paFloat32
    channels = 1
    rate = 16000
    chunkSize = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunkSize)
    stream_out = audio.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=rate,
                            output=True,
                            frames_per_buffer=chunkSize)

    
    # Initializations
    recent_arrays = NumpyArrayQueue()
    input_queue = [None, None]
    inverted_queue = [None, None]
    map = {}
    annoy_index = AnnoyIndex(1024, 'angular')

    try:
        print("Recording...")
        iter = 0
        i = 0
        input_array = np.empty((1024 * 87), dtype=np.float32)  # change to 1024*80 when needed
        pred_history = 1
        pos = 0
        while True:
            start_time = datetime.now()
            audio_data = stream.read(chunkSize)
            end_time = datetime.now()
            difference = end_time - start_time
            global ratio
            try:
                ratio = 1024 / difference.total_seconds()
            except ZeroDivisionError:
                # print('ZeroDivisionError')
                pass
            start_time = datetime.now()
            numerical_data = np.frombuffer(audio_data, dtype=np.float32)

            input_queue[0], input_queue[1] = input_queue[1], numerical_data

            if pred_history == 1:  # Meaning that the last sound is bad
                # Calculate similarity between all subparts of the recent arrays and the input array
                if annoy_index.get_n_items() > 0:
                    indices = annoy_index.get_nns_by_vector(vector=numerical_data.flatten(), n=1)

                    # Retrieve the data of the neighbor with the highest similarity
                    neighbor = indices[0]
                    neighbor_data = tuple(annoy_index.get_item_vector(neighbor))
                    print(map[neighbor_data])
                    parent_array_index, index = map[neighbor_data][0], map[neighbor_data][1]
                    end_time = datetime.now()
                    calc_time = (end_time - start_time).total_seconds()
                    delay = int(ratio * calc_time)
                    print('parent array index: ', parent_array_index)
                    print("effective index: ", index+delay)

                    # Get the actual chunk which is to be inverted
                    try:
                        output = -1 * recent_arrays.peek(parent_array_index)[index+delay : index+delay+1024]
                        inverted_queue[0], inverted_queue[1] = inverted_queue[1], output

                        try:
                            # Plot the result
                            if inverted_queue[0] is not None:
                                plot_final(input_queue[1], inverted_queue[0], iter)
                        except TypeError:
                            print('Type error')

                    except IndexError:
                        print("Index Out of range")

            if i < numChunks:  # Replace with the correct value to make it 0.5 secs
                input_array[pos: pos + 1024] = numerical_data
                i += 1
                pos += 1024

            # When the input array is of length numChunks*chunkSize
            else:
                recent_arrays.enqueue(input_array)
                # Update the ANNOY index and map
                annoy_index = AnnoyIndex(chunkSize, 'angular')
                for i in range(recent_arrays.get_length()):
                    array = recent_arrays.peek(i)
                    for j in range(0, len(array), chunkSize):
                        subarray = input_array[j:j + chunkSize]
                        annoy_index.add_item(j+i*numChunks, subarray)

                # Build the new index
                annoy_index.build(int(recent_arrays.get_length()*numChunks))

                # If the map has less entries than 5 arrays
                if len(map) == numChunks*maxNumArrays:
                    map = {key: value for key, value in list(map.items())[numChunks:]}
                    for key in map.keys():
                        map[key][0] -= 1

                queue_len = recent_arrays.get_length()
                for i in range(0, len(input_array), chunkSize):
                    subarray = input_array[i:i + chunkSize]
                    map[tuple(subarray)] = [queue_len - 1, i]

                print(map.values())

                # Update the ANNOY Index
                # if annoy_index.get_n_items() > numChunks*maxNumArrays*chunkSize:
                #     annoy_index = convert_and_trim_index(annoy_index, numChunks * maxNumArrays * chunkSize)


            ################################     Data handling is over     #############################

                global noisy_sounds
                scores, embeddings, spectrogram = model(input_array)
                class_scores = tf.reduce_mean(scores, axis=0)
                top_class_yamnet = tf.math.argmax(class_scores)
                inferred_class_yamnet = class_names[top_class_yamnet]
                top_score_yamnet = class_scores[top_class_yamnet]

                if inferred_class_yamnet in noisy_sounds:
                    print(f'prediction: Bad  {inferred_class_yamnet}\n\n')
                    pred_history = 1


                else:
                    print(f'prediction: Good  {inferred_class_yamnet}\n\n')
                    pred_history = 0

                input_array = np.empty((1024 * 87), dtype=np.float32)  # change to 1024*80 when needed
                pos = 0
                iter +=1

    except KeyboardInterrupt:
        print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    stream_out.stop_stream()
    stream_out.close()
    audio.terminate()
