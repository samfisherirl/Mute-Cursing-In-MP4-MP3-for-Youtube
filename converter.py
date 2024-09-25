import speech_recognition as sr
import librosa
import pandas as pd
import soundfile as sf
import numpy as np
# Load your CSV file with words to mute
words_to_mute_df = pd.read_csv('words_to_mute.csv', header=None)
words_to_mute = words_to_mute_df[0].tolist()

# Load your audio file
audio_path = 'your_audio_file.wav'  # Change to your audio file path
audio_data, sample_rate = librosa.load(
    audio_path, sr=None)  # Renamed variable here

# Function to mute desired words


def mute_words(audio_chunk, sample_rate, words):
    recognizer = sr.Recognizer()  # Now it should work correctly
    with sr.AudioData(audio_data.tobytes(), sample_rate, audio_data.dtype.itemsize) as source:
        try:
            text = recognizer.recognize_google(source)
            print(f"Recognized text: {text}")
            for word in words:
                if word.lower() in text.lower():
                    return (audio_data * 0).astype(audio_data.dtype)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}")
    return audio_data


# Process the audio file and mute words
processed_audio = []
for i in range(0, len(y), sr):  # Process in chunks of 1 second
    print(f"Processing chunk {i//sr}/{len(y)//sr}...")
    chunk = y[i:i+sr]
    processed_chunk = mute_words(chunk, sr, words_to_mute)
    processed_audio.append(processed_chunk)

# Convert the processed audio list back to an array
processed_audio = np.concatenate(processed_audio, axis=0)

# Save the processed audio
sf.write('processed_audio.wav', processed_audio, sr)
print("Audio processing complete. Saved as 'processed_audio.wav'.")
