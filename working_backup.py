```py
# Ensure librosa is installed:
# pip install librosa

import librosa
import librosa.display
import numpy as np
import json
import soundfile as sf

# Function to load a saved transcript from a JSON file


def load_saved_transcript(json_file_path):
    with open(json_file_path, 'r') as file:
        transcript = json.load(file)
    return transcript

# Function to mute curse words in the audio

def mute_curse_words(audio_data, sample_rate, transcription_result, curse_words_list):
    # Create a copy of the audio data to avoid modifying the original
    audio_data_muted = np.copy(audio_data)
    # Go through each word in the transcription result
    for segment in transcription_result['segments']:
        word = [word for word in curse_words_list if word.lower() in segment['text']]
        if word != []:
            for word in segment['words']:
                if word['word'].lower().strip() in curse_words_list:
                    # Calculate the start and end samples
                    start_sample = int(word['start'] * sample_rate)
                    end_sample = int(word['end'] * sample_rate)
                    # Mute the curse words by setting the amplitude to zero
                    audio_data_muted[start_sample:end_sample] = 0
    return audio_data_muted


if __name__ == '__main__':
    # Load your audio file
    audio_data, sample_rate = librosa.load('Mixdown.mp3', sr=None)

    # Define a list of curse words to mute
    curse_words = ['fucking', 'shit']  # Replace with actual curse words

    # Path to your saved JSON transcript file
    transcript_path = 'transcription0.json'

    # Load the transcript
    transcript = load_saved_transcript(transcript_path)

    # Mute the curse words in the audio
    muted_audio_data = mute_curse_words(
        audio_data, sample_rate, transcript, curse_words)

    # Export the muted audio to a new file
    sf.write('muted_audio.wav', muted_audio_data, sample_rate)
```