import librosa
import stable_whisper
import librosa.display
import numpy as np
import json
import soundfile as sf
import csv
from tkinter import Tk, messagebox, filedialog
import tkinter as tk
import csv
import random
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from moviepy.editor import *
import torch
from datetime import datetime
from progress.bar import Bar



# Define paths and file names
OUTPUT_AUDIO_PATH = 'output.mp3'
NEW_AUDIO_PATH = 'clean_audio.mp3'
OUTPUT_VIDEO_PATH = 'clean_video.mp4'
CURSE_WORD_FILE = 'curse_words.csv'

day = datetime.now().strftime('%d')
mo = datetime.now().strftime('%m')
time = datetime.now().strftime('%H-%M-%S')

lemmatizer = WordNetLemmatizer()
# Function to update the progress bar

 
def load_saved_transcript(json_file_path):
    with open(json_file_path, 'r') as file:
        transcript = json.load(file)
    return transcript
# Function to recursively convert all strings in a JSON object to lowercase


def to_lowercase(input):
    if isinstance(input, dict):
        return {k.lower().strip(): to_lowercase(v) for k, v in input.items()}
    elif isinstance(input, list):
        return [to_lowercase(element) for element in input]
    elif isinstance(input, str):
        return input.lower().strip()
    else:
        return input

def process_json(infile): 
    # Read the original JSON file
    with open(infile, 'r') as file:
        data = json.load(file)
    # Convert all strings to lowercase
    lowercase_data = to_lowercase(data)
    # Write the modified JSON to a new file
    with open(infile, 'w') as file:
        json.dump(lowercase_data, file, indent=4)
    # Read the original JSON file
    return lowercase_data

def read_curse_words_from_csv(CURSE_WORD_FILE):
  curse_words_list = []
  with open(CURSE_WORD_FILE, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      curse_words_list.append(row[0])  # Assuming curse words are in column A
  return curse_words_list
# Function to mute curse words in the audio


def load_transcript():
    # Ask the user if they want to load an existing transcript
    if messagebox.askyesno('Load Transcript', 'If this program crashed, it saves the transcript to ensure it doesn\'t require restarting. Do you want to load an existing transcript?'):
        # File dialog to select a transcript JSON file
        transcript_path = filedialog.askopenfilename(
            title='Select Transcript File',
            filetypes=[('JSON files', '*.json')]
        )
        if transcript_path:
            print(f'Transcript file selected: {transcript_path}')
            return transcript_path
    return None


def select_audio_or_video():
    # File dialog to select an audio file
    av_path = filedialog.askopenfilename(
        title='Select A/V files',
        filetypes=[('A/V files', '*.mp3 *.wav *.mp4')]
    )
    if av_path:
        print(f'Audio/Video file selected: {av_path}')
        return av_path
    return None


def load_saved_transcript(json_file_path):
    with open(json_file_path, 'r') as file:
        transcript = json.load(file)
    return transcript


def read_curse_words_from_csv(CURSE_WORD_FILE):
  curse_words_list = []
  with open(CURSE_WORD_FILE, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      curse_words_list.append(row[0])  # Assuming curse words are in column A
  return curse_words_list
# Function to mute curse words in the audio
# Function to check if the base form of a word is in the curse words set


def is_curse_word(word, curse_words_set):
    # Find the base form of the word
    lemma = lemmatizer.lemmatize(word.strip())
    # Check if the base form is in the curse words set
    return lemma in curse_words_set


def split_silence(sample_rate, word):
    # Calculate the start and end sample indices
    start_sample = int(word['start'] * sample_rate)
    end_sample = int(word['end'] * sample_rate)

    # Check if the duration is less than 0.3 seconds
    if (end_sample - start_sample) < int(0.3 * sample_rate):
        # Calculate the middle point of the word
        middle_sample = (start_sample + end_sample) // 2
        # Expand the start and end to cover 0.5 seconds total duration
        start_sample = max(0, middle_sample - int(0.25 * sample_rate))
        end_sample = middle_sample + int(0.25 * sample_rate)

    return start_sample, end_sample

def mute_curse_words(audio_data, sample_rate, transcription_result, curse_words_list):
    # Create a copy of the audio data to avoid modifying the original
    audio_data_muted = np.copy(audio_data)
    # Create a set for faster membership testing
    curse_words_set = set(word.lower() for word in curse_words_list)
    bar = Bar('Processing', max=len(transcription_result['segments']))

    # Initialize an empty list to store the start and end sample indices for muting
    mute_indices = []
    # Go through each segment in the transcription result
    for segment in transcription_result['segments']:
        bar.next()
        # Go through each word in the segment
        for word in segment['words']:
            # Check if the word is in the curse words set
            if word['word'] in curse_words_set:
                start_sample, end_sample = split_silence(sample_rate, word)
                mute_indices.append((start_sample, end_sample)) 
    # Mute the curse words by setting the amplitude to zero
    for start_sample, end_sample in mute_indices:
        audio_data_muted[start_sample:end_sample] = 0
    bar.finish()
    return audio_data_muted


def transcribe_audio(audio_file, device_type):
    model = stable_whisper.load_faster_whisper(
        'large-v2', compute_type="float16", device=device_type)
    # model = stable_whisper.load_model('large-v3', device=device_type)
    result = model.transcribe_stable(audio_file)
    transcript_path = f'transcript{random.randint(0, 100)}.json'
    result.save_as_json(transcript_path)
    return transcript_path


def find_curse_words(audio_data, sample_rate, transcript_file, CURSE_WORD_FILE=CURSE_WORD_FILE):
    results = process_json(transcript_file)
    curses = read_curse_words_from_csv(CURSE_WORD_FILE)
    curse_words_set = set(curses)
    return mute_curse_words(audio_data, sample_rate, results, curse_words_set)


def process_audio(audio_file, transcript_file=None):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if not transcript_file:
        transcript_file = transcribe_audio(audio_file, device_type)
    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    muted_audio = find_curse_words(
        audio_data, sample_rate, transcript_file, CURSE_WORD_FILE)
    outfile = Path(audio_file).stem + 'muted_audio.mp3'
    sf.write(outfile, muted_audio, sample_rate)
    return outfile


def process_video(input_video_path, transcript_file):
    video_clip = VideoFileClip(input_video_path)
    OUTPUT_AUDIO_PATH = input_video_path.replace('.mp4', '.mp3')
    video_clip.audio.write_audiofile(OUTPUT_AUDIO_PATH)
    new_audio_path = process_audio(
        OUTPUT_AUDIO_PATH, CURSE_WORD_FILE, transcript_file)  # fix audio path
    with AudioFileClip(new_audio_path) as new_audio_clip:
        final_video = video_clip.set_audio(new_audio_clip)
        final_video.write_videofile(
            OUTPUT_VIDEO_PATH, codec='libx264', audio_codec='aac')

def main():
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)
    transcript_file = load_transcript()
    input_video_path = select_audio_or_video()

    if Path(input_video_path).suffix == '.mp4':
        process_video(input_video_path, transcript_file)
    else:
        # Process audio only
        process_audio(input_video_path, transcript_file)


if __name__ == "__main__":
    main()
