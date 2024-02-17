# Ensure librosa is installed:
# pip install librosa

import librosa
import stable_whisper
import librosa.display
import numpy as np
import json
import soundfile as sf
import csv
from tkinter import Tk as Tk
from tkinter import filedialog
from tkinter import messagebox
import csv
import random
from pathlib import Path
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def load_saved_transcript(json_file_path):
    with open(json_file_path, 'r') as file:
        transcript = json.load(file)
    return transcript


def read_curse_words_from_csv(csv_file_path):
  curse_words_list = []
  with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      curse_words_list.append(row[0])  # Assuming curse words are in column A
  return curse_words_list
# Function to mute curse words in the audio


def load_transcript():
    # Ask the user if they want to load an existing transcript
    if messagebox.askyesno('Load Transcript', 'Do you want to load an existing transcript?'):
        # File dialog to select a transcript JSON file
        transcript_path = filedialog.askopenfilename(
            title='Select Transcript File',
            filetypes=[('JSON files', '*.json')]
        )
        if transcript_path:
            print(f'Transcript file selected: {transcript_path}')
            return transcript_path
    return None


def select_audio_file():
    # File dialog to select an audio file
    audio_path = filedialog.askopenfilename(
        title='Select Audio File',
        filetypes=[('Audio files', '*.mp3 *.wav')]
    )
    if audio_path:
        print(f'Audio file selected: {audio_path}')
        return audio_path
    return None


def load_saved_transcript(json_file_path):
    with open(json_file_path, 'r') as file:
        transcript = json.load(file)
    return transcript


def read_curse_words_from_csv(csv_file_path):
  curse_words_list = []
  with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      curse_words_list.append(row[0])  # Assuming curse words are in column A
  return curse_words_list
# Function to mute curse words in the audio
# Function to check if the base form of a word is in the curse words set


def is_curse_word(word, curse_words_set):
    # Find the base form of the word
    lemma = lemmatizer.lemmatize(word.lower().strip())
    # Check if the base form is in the curse words set
    return lemma in curse_words_set

def mute_curse_words(audio_data, sample_rate, transcription_result, curse_words_list):
    # Create a copy of the audio data to avoid modifying the original
    audio_data_muted = np.copy(audio_data)
        # Go through each word in the transcription result
    # Create a set for faster membership testing
    curse_words_set = set(word.lower() for word in curse_words_list)

    # Generate the start and end sample indices for muting
    mute_indices = [
        (int(word['start'] * sample_rate), int(word['end'] * sample_rate))
        for segment in transcription_result['segments']
        for word in segment['words']
        if word['word'].lower().strip() in curse_words_set
    ]

    # Create a copy of the audio data to mute
    audio_data_muted = np.copy(audio_data)

    # Mute the curse words by setting the amplitude to zero
    for start_sample, end_sample in mute_indices:
        audio_data_muted[start_sample:end_sample] = 0
                
    return audio_data_muted


def main(transcript_file, audio_file):
    if not transcript_file:
        # Load your audio file
        model = stable_whisper.load_model('base')
        # Transcribe the audio file
        result = model.transcribe(audio_file)
        r = random.randint(0, 100)
        # Save the transcription result as a JSON file for future use
        result.save_as_json(f'transcription{r}.json')
        # Define a list of curse words to mute
        # Path to your saved JSON transcript file
        transcript_file = f'transcription{r}.json'
    results = load_saved_transcript(transcript_file)
    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    # Mute the curse words in the audio
    curses = read_curse_words_from_csv(Path.cwd() / "curse_words.csv")
    curse_words_set = set(curses)

    muted_audio = mute_curse_words(
        audio_data, sample_rate, results, curse_words_set)
    outfile = Path(audio_file).stem
    parent = Path(audio_file).parent
    complete_name = f"{outfile}muted_audio.mp3"
    complete = parent / complete_name
    sf.write(complete, muted_audio, sample_rate)

if __name__ == '__main__':
    # Load the model
    root = Tk()
    root.withdraw()  # Hide the main window
    transcript_file = load_transcript()
    audio_file = select_audio_file()
    # Bind the close event to the on_close function
    main(transcript_file, audio_file)
