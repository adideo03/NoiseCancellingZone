import hashlib
import sounddevice as sd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def callback(indata, frames, time, status):
    if status:
        print(f"Error in audio input: {status}")
    if any(indata):
        # Process the audio
        audio_data.append(indata.copy())


def calculate_cosine_similarity(array1, array2):
    # Reshape arrays to have a single dimension
    array1_flat = array1.flatten()
    array2_flat = array2.flatten()

    # Ensure the arrays have the same length
    min_len = min(len(array1_flat), len(array2_flat))
    array1_flat = array1_flat[:min_len]
    array2_flat = array2_flat[:min_len]

    # Calculate cosine similarity
    similarity = cosine_similarity([array1_flat], [array2_flat])[0][0]

    # Handle edge case when both arrays are all zeros
    if np.all(array1_flat == 0) and np.all(array2_flat == 0):
        return 1.0  # Both arrays are the same (all zeros)

    return max(0, min(1, similarity))  # Ensure similarity is within [0, 1]


sample_rate = 44100
duration = 5

audio_data = []
with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
    print(f"Recording for {duration} seconds...")
    sd.sleep(duration * 1000)  # Convert seconds to milliseconds
    print("Recording complete.")

sound = np.concatenate(audio_data, axis=0)

print(f"Shape of the audio array: {sound.shape}")
print(f"Sample rate: {sample_rate}")

repeated_subparts = {}
pattern_sequence = []

# Split sound file into overlapping windows
window_size = 2
step_size = 2
windows = [sound[i:i + window_size] for i in range(0, len(sound) - window_size + 1, step_size)]

# Iterate over the windows and calculate their MD5 hashes
for window in windows:
    md5_hash = hashlib.md5(window.tobytes()).hexdigest()

    # If the MD5 hash is already in the hash table, then the window is a repeated subpart
    if md5_hash in repeated_subparts:
        repeated_subparts[md5_hash].append(window.tolist())
    else:
        # If the MD5 hash is not in the hash table, add the pattern to the sequence
        pattern_sequence.append(md5_hash)
        repeated_subparts[md5_hash] = [window.tolist()]

# Consolidate repeated patterns
consolidated_result = {}
for pattern_hash in pattern_sequence:
    if len(repeated_subparts[pattern_hash]) > 1:
        if pattern_hash not in consolidated_result:
            consolidated_result[pattern_hash] = repeated_subparts[pattern_hash]

# Represent in numerical values
sound_representation = []
for window in windows:
    md5_hash = hashlib.md5(window.tobytes()).hexdigest()

    if md5_hash in consolidated_result:
        numerical_value = list(consolidated_result.keys()).index(md5_hash) + 1  # +1 to avoid 0
        sound_representation.extend([numerical_value] * window_size)
    else:
        sound_representation.extend([0] * window_size)

print(f"Sound represented by numerical values: {sound_representation}")

num_parts = 40
divided_sound = np.array_split(sound_representation, num_parts)
# # Print each part
for i, part in enumerate(divided_sound, start=1):
    print(f"Audio array {i}: {part}")


similarities_list = []

for i in range(num_parts):
    for j in range(i + 1, num_parts):
        similarity = calculate_cosine_similarity(divided_sound[i], divided_sound[j]) * 100
        similarities_list.append((i + 1, j + 1, similarity))

# descending order
similarities_list.sort(key=lambda x: x[2], reverse=True)


for i, j, similarity in similarities_list:
    print(f"Similarity between array {i} and array {j}: {similarity}%")
