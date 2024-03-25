# Ensure stable-ts is installed:
# pip install -U stable-ts

import stable_whisper
import random
import json
from pathlib import Path
from tkinter import Tk as Tk
from tkinter import filedialog
from tkinter import messagebox
import csv

# Hide the main Tkinter window
Tk().withdraw()
# Use a GUI module to display the file select dialog
file_path = filedialog.askopenfilename(
    filetypes=[("mp3", "*.*")]
)

# Check if a file was selected
if not Path(file_path).is_file():
    exit()



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

# Function to mute curse words in the audio
def mute_curse_words(audio_segment, transcription_result, curse_words_list):
    # Go through each word in the transcription result
    try:
        words = transcription_result.segments
    except Exception as e:
        print(str(e))
        words = transcription_result['segments']
    for word in words:
        if word['text'].lower() in curse_words_list:
            # Calculate the start and end times in milliseconds
            start_time = (word.start * 1000)+66000
            end_time = (word.end * 1000)+116000
            # Create a silent segment for the duration of the curse word
            silent_segment = AudioSegment.silent(
                duration=end_time - start_time)
            # Overlay the silent segment onto the original audio
            audio_segment = audio_segment.overlay(
                silent_segment, position=start_time)
    return audio_segment

if __name__ == '__main__':
