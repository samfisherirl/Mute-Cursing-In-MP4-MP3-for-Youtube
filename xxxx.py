import librosa
import speech_recognition as sr
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
from pydub import AudioSegment
import subprocess
import wave

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
        if ".mp3" in av_path:
            subprocess.call(
                ['ffmpeg', '-i', av_path, av_path+'.wav'])
            return av_path+'.wav'
            
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


def chunk_audio(audio_data, chunk_length_s, sample_rate):
    # Ensure audio_data is a NumPy array and has a shape attribute
    if isinstance(audio_data, np.ndarray) and audio_data.ndim == 1:
        # Calculate the number of samples per chunk
        chunk_size = int(chunk_length_s * sample_rate)
        # Calculate the total number of chunks
        num_chunks = int(np.ceil(len(audio_data) / chunk_size))
        # Yield successive chunks of audio data
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            yield audio_data[start:end]
    else:
        raise ValueError(
            "audio_data must be a one-dimensional NumPy array with a defined shape")


def combine_audio_chunks(chunks_filenames, output_filename):
    # Create a new audio segment for the combined audio
    combined_audio = AudioSegment.empty()
    # Loop through the list of chunk filenames
    for filename in chunks_filenames:
        # Load the audio chunk
        chunk_audio = AudioSegment.from_wav(filename)
        # Append the chunk to the combined audio
        combined_audio += chunk_audio
    # Export the combined audio to a single WAV file
    combined_audio.export(output_filename, format="wav")
    
    
# Function to mute curse words in the audio
def mute_curse_words(audio_data, sample_rate, transcription_result, curse_words_list):
    # Create a set for faster membership testing
    curse_words_set = set(word.lower() for word in curse_words_list)

    # Generate the start and end sample indices for muting
    mute_indices = [
        (int(word['start'] * sample_rate), int(word['end'] * sample_rate))
        for word in transcription_result['words']
        if word['text'].lower().strip() in curse_words_set
    ]

    # Mute the curse words by setting the amplitude to zero
    for start_sample, end_sample in mute_indices:
        audio_data[start_sample:end_sample] = 0

    return audio_data


# Function to save a chunk of audio data to a WAV file
def save_wav(filename, audio_data, sample_rate):
    # Open the file in 'write bytes' mode
    with wave.open(filename, 'wb') as wav_file:
        # Set the number of channels
        wav_file.setnchannels(1)
        # Set the sample width to 2 bytes (16 bits)
        wav_file.setsampwidth(2)
        # Set the frame rate to the sample rate
        wav_file.setframerate(sample_rate)
        # Write the frames to the file
        wav_file.writeframes(audio_data.tobytes())


def censor(audio_file_path, curse_words_list, chunk_length_s=10):
    # Initialize recognizer
    r = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(audio_file_path) as source:
        audio_data = np.array(r.record(source).get_raw_data())
        sample_rate = source.DURATION

    # List to keep track of chunk filenames
    chunks_filenames = []

    # Process audio in chunks
    for i, chunk in enumerate(chunk_audio(audio_data, chunk_length_s, sample_rate)):
        try:
            # Recognize audio chunk using Google Web Speech API
            transcription_result = r.recognize_google(chunk, show_all=True)
            # Mute curse words in the chunk
            chunk_muted = mute_curse_words(
                chunk, sample_rate, transcription_result, curse_words_list)
            # Save the muted chunk
            chunk_filename = f'muted_chunk_{i}.wav'
            save_wav(chunk_filename, chunk_muted, sample_rate)
            chunks_filenames.append(chunk_filename)

        except sr.UnknownValueError:
            print(
                f"Chunk {i}: Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(
                f"Chunk {i}: Could not request results from Google Speech Recognition service; {e}")

    # Combine muted chunks into a single audio file
    combine_audio_chunks(chunks_filenames, 'combined_output.wav')


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
        new_audio_path = censor(output_audio_path, transcript_file)
        skip_video = False
    else:
        new_audio_path = censor(input_video_path,
                                read_curse_words_from_csv('curse_words.csv'))
        skip_video = True
    if not skip_video:
        new_audio_clip = AudioFileClip(new_audio_path)
        final_video = video_clip.set_audio(new_audio_clip)
        final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
        # Close the clips
        new_audio_clip.close()
        video_clip.close()
        # Bind the close event to the on_close function
