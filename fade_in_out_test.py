import stable_whisper
import numpy as np
import json
import soundfile as sf
import csv
from tkinter import Tk, messagebox, filedialog
import subprocess
import csv
import random
import videoxt
from pathlib import Path
from nltk.stem import WordNetLemmatizer
import torch
from datetime import datetime
from progress.bar import Bar


cwd = Path(__file__).parent

# Define paths and file names
CURSE_WORD_FILE = 'curse_words.csv'

day = datetime.now().strftime('%d')
mo = datetime.now().strftime('%m')
time = datetime.now().strftime('%H-%M-%S')

lemmatizer = WordNetLemmatizer()
# Function to update the progress bar


def make_dirs():
    Path(cwd / "tscript").mkdir(parents=True, exist_ok=True)
    Path(cwd / "exports").mkdir(parents=True, exist_ok=True)
    
 
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

# def mute_curse_words(audio_data, sample_rate, transcription_result, curse_words_list):
#     # Create a copy of the audio data to avoid modifying the original
#     audio_data_muted = np.copy(audio_data)
#         # Go through each word in the transcription result
#     # Create a set for faster membership testing
#     curse_words_set = set(word.lower() for word in curse_words_list)

#     # Generate the start and end sample indices for muting
#     mute_indices = [
#         (int(word['start'] * sample_rate), int(word['end'] * sample_rate))
#         for segment in transcription_result['segments']
#         for word in segment['words']
#         if word['word'].strip() in curse_words_set
#     ]

#     # Create a copy of the audio data to mute
#     audio_data_muted = np.copy(audio_data)

#     # Mute the curse words by setting the amplitude to zero
#     for start_sample, end_sample in mute_indices:
#         audio_data_muted[start_sample:end_sample] = 0
                
#     return audio_data_muted


def apply_fade(audio_data, start_sample, end_sample, sample_rate, fade_duration=0.001):
    # Calculate the number of samples for the fade duration
    fade_samples = int(fade_duration * sample_rate)

    # Apply an exponential fade-in
    for i in range(fade_samples):
        # Calculate the exponential fade-in factor
        fade_in_factor = 1 - np.exp(-i / fade_samples)
        # Apply fade-in to the starting sample
        audio_data[start_sample + i] *= fade_in_factor

    # Apply an exponential fade-out
    for i in range(fade_samples):
        # Calculate the exponential fade-out factor
        fade_out_factor = np.exp(-(fade_samples - i) / fade_samples)
        # Apply fade-out to the ending sample
        audio_data[end_sample - i] *= fade_out_factor

    return audio_data


def split_silence(sample_rate, word):
    # Calculate the start and end sample indices  
    start_sample = int(word['start'] * sample_rate)
    end_sample = int(word['end'] * sample_rate)


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
            if word['word'] in curse_words_set:
                # Check if the word is in the curse words set
                start_sample, end_sample = split_silence(sample_rate, word)
                # Apply fade-in before muting
                apply_fade(audio_data_muted, start_sample,
                        end_sample, sample_rate) 
                # Mute the curse words by setting the amplitude to zero
                audio_data_muted[start_sample:end_sample] = 0
                # Apply fade-out after muting
                apply_fade(audio_data_muted, start_sample,
                        end_sample, sample_rate)

    return audio_data_muted



def transcribe_audio(audio_file, device_type):
    model = stable_whisper.load_faster_whisper(
        'large-v3', compute_type="float16", device=device_type)
    # model = stable_whisper.load_model('large-v3', device=device_type)
    result = model.transcribe_stable(
        audio_file, beam_size=5, word_timestamps=True)
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
    audio_data, sample_rate = sf.read(audio_file, samplerate=None)
    muted_audio = find_curse_words(
        audio_data, sample_rate, transcript_file)
    outfile = Path(audio_file).parent / \
        str(Path(audio_file).name + '_muted_audio.mp3')
    sf.write(outfile, muted_audio, sample_rate)
    return outfile


def process_video(input_video_path, transcript_file):
    # Define output paths
    output_audio_path = Path(input_video_path).with_suffix('.mp3') 
    suf = str(Path(output_audio_path).suffix)
    
    output_video_path = str(output_audio_path).replace(suf, "clean_video.mp4")
    audio = videoxt.extract_audio(
        input_video_path, audio_format='mp3', destdir=Path(input_video_path).parent)
    
    # Process audio (assuming process_audio returns a path to the processed audio file)
    return process_audio(
        str(audio.destpath), transcript_file)


def main():
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)
    transcript_file = load_transcript()
    input_video_path = select_audio_or_video()
    cwd = Path(input_video_path).parent

    if Path(input_video_path).suffix == '.mp4':
        result = process_video(input_video_path, transcript_file)
        song = Path(input_video_path).parent / result

    else:
        # Process audio only
        process_audio(input_video_path, transcript_file)


if __name__ == "__main__":
    make_dirs()
    main()
