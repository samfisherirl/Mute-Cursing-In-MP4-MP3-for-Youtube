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
from moviepy.editor import *
import torch

lemmatizer = WordNetLemmatizer()

def load_saved_transcript(json_file_path):
    with open(json_file_path, 'r') as file:
        transcript = json.load(file)
    return transcript
# Function to recursively convert all strings in a JSON object to lowercase


def to_lowercase(input):
    if isinstance(input, dict):
        return {k.lower(): to_lowercase(v) for k, v in input.items()}
    elif isinstance(input, list):
        return [to_lowercase(element) for element in input]
    elif isinstance(input, str):
        return input.lower()
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
    with open(infile, 'r') as file:
        data = json.load(file)

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

    # Initialize an empty list to store the start and end sample indices for muting
    mute_indices = []
    # Go through each segment in the transcription result
    for segment in transcription_result['segments']:
        # Go through each word in the segment
        for word in segment['words']:
            # Check if the word is in the curse words set
            if word['word'].strip() in curse_words_set:
                start_sample, end_sample = split_silence(sample_rate, word)
                mute_indices.append((start_sample, end_sample)) 

    # Mute the curse words by setting the amplitude to zero
    for start_sample, end_sample in mute_indices:
        audio_data_muted[start_sample:end_sample] = 0

    return audio_data_muted

def censor(transcript_file, audio_file):
    if not transcript_file:
        # Load your audio file
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = stable_whisper.load_model('medium', device=device)
        except Exception as e:
            print(str(e))
            model = stable_whisper.load_model('medium')
        # Transcribe the audio file
        result = model.transcribe(audio_file)
        r = random.randint(0, 100)
        # Save the transcription result as a JSON file for future use
        result.save_as_json(f'transcription{r}.json')
        # Define a list of curse words to mute
        # Path to your saved JSON transcript file 
        transcript_file = f'transcription{r}.json'
        
    process_json(transcript_file)
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
    return complete_name

if __name__ == '__main__':
    # Load the model
    root = Tk()
    root.withdraw()  # Hide the main window
    transcript_file = load_transcript()
    input_video_path = select_audio_or_video()
    if Path(input_video_path).name.endswith('.mp4'):
        # Replace 'output.mp3' with the desired output MP3 file path
        output_audio_path = 'output.mp3'
        # Replace 'input_video.mp4' with the path to your MP4 file
        new_audio_path = 'new_audio.mp3'
        # Replace 'output_video.mp4' with the desired output MP4 file path
        output_video_path = 'output_video.mp4'
        # Load the video file
        video_clip = VideoFileClip(input_video_path)
        # Load the new audio file
        new_audio_path = censor(transcript_file, output_audio_path)
        skip_video = False
    else:
        new_audio_path = censor(
            transcript_file, input_video_path)
        skip_video = True
    if not skip_video:
        new_audio_clip = AudioFileClip(new_audio_path)
        final_video = video_clip.set_audio(new_audio_clip)
        final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
        # Close the clips
        new_audio_clip.close()
        video_clip.close()
        # Bind the close event to the on_close function
