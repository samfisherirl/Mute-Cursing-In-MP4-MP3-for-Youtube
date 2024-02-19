import stable_whisper
import numpy as np
import json
import soundfile as sf
import csv
from tkinter import Tk, messagebox, filedialog
import subprocess
import csv
import random
from audio_extract import extract_audio
from pathlib import Path
from nltk.stem import WordNetLemmatizer
import torch
from datetime import datetime
from progress.bar import Bar
import wave
from os import remove


cwd = Path(__file__).parent

# Define paths and file names
CURSE_WORD_FILE = 'curse_words.csv'
sample_audio_path = 'looperman.wav'

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
def load_wav_as_np_array(wav_file_path):
    # Open the audio file
    with wave.open(wav_file_path, "rb") as wav_file:
        # Ensure that the audio file is mono
        if wav_file.getnchannels() != 1:
            raise ValueError("Only mono audio files are supported.")
        
        # Extract audio frames
        frames = wav_file.readframes(wav_file.getnframes())
        
        # Convert audio frames to float32 NumPy array
        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        
        # Normalize the audio data
        audio_data /= np.iinfo(np.int16).max
        
        # Return the audio data and the sample rate
        return audio_data, wav_file.getframerate()


def main_file_audio(wav_file_path):
    # Read the audio file
    audio_data, sample_rate = sf.read(wav_file_path, dtype='float32')

    # Ensure that the audio file is mono
    if audio_data.ndim > 1:
        # Average the channels if more than one channel (i.e., stereo)
        audio_data = np.mean(audio_data, axis=1)

    # 'soundfile' already reads into 'float32', and the data is typically normalized
    # If you need to ensure normalization between -1.0 and 1.0, uncomment the following lines:
    # peak = np.max(np.abs(audio_data))
    # if peak > 1:
    #     audio_data /= peak

    return audio_data, sample_rate



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
    if messagebox.askyesno('Load Transcript', 'If this program crashed, this saves the transcript to ensure it doesn\'t require restarting.\n\nDo you want to load an existing transcript?'):
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


def _load_wav_as_np_array(file_path):
    # Open the .wav file
    with wave.open(file_path, 'rb') as wav_file:
        # Get parameters
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        # Read frames
        frames = wav_file.readframes(n_frames)

    # Convert the byte data to numpy array based on the sample width
    if sample_width == 1:  # 8-bit unsigned
        np_audio = np.frombuffer(frames, dtype=np.uint8) - 128
    elif sample_width == 2:  # 16-bit signed
        np_audio = np.frombuffer(frames, dtype=np.int16)
    elif sample_width == 3:  # 24-bit signed
        np_audio = np.frombuffer(frames, '<i3').view('<i4')
    elif sample_width == 4:  # 32-bit signed or unsigned
        np_audio = np.frombuffer(frames, dtype=np.int32)
    else:
        raise ValueError("Unsupported sample width")

    # Convert stereo to mono by averaging channels
    if n_channels == 2:
        np_audio = np.mean(np_audio.reshape(-1, 2), axis=1)

    # Normalize audio to range [-1, 1]
    np_audio = np_audio.astype(np.float32) / np.iinfo(np_audio.dtype).max

    return np_audio, sample_rate


def linear_crossfade(audio_data, start_sample, end_sample, sample_audio_data):
    required_sample_length = end_sample - start_sample
    # Ensuring the sample is the correct length
    if len(sample_audio_data) < required_sample_length:
        loop_count = required_sample_length // len(sample_audio_data) + 1
        sample_audio_data = np.tile(sample_audio_data, loop_count)[
            :required_sample_length]

    # Make sure sample_audio_data matches the number of channels in audio_data
    if audio_data.ndim > 1 and sample_audio_data.ndim == 1:
        sample_audio_data = np.tile(
            sample_audio_data[:, None], (1, audio_data.shape[1]))

    # Linear crossfade
    for t in range(required_sample_length):
        u = t / float(required_sample_length)
        audio_data[start_sample + t] = audio_data[start_sample +
                                                  t] * (1 - u) + sample_audio_data[t] * u

    return audio_data


def remove_clicks(audio_data, sample_rate, threshold=0.1, window_size=200):
    """
    A simple click removal function that scans for sudden changes in the
    audio signal amplitude and smoothes them out by interpolating the waveform.

    Parameters:
    audio_data : numpy.array
        The audio data.
    sample_rate : int
        The sample rate of the audio data.
    threshold : float
        The threshold for detecting a click (relative to max amplitude).
    window_size : int
        The number of samples used for interpolation around the click.

    Returns:
    cleaned_audio : numpy.array
        The audio data with clicks removed.
    """

    max_amplitude = np.max(np.abs(audio_data))
    click_threshold = max_amplitude * threshold
    cleaned_audio = np.copy(audio_data)

    # Iteratively scan for abrupt changes that may indicate clicks
    for i in range(1, len(audio_data) - 1):
        if np.abs(audio_data[i] - audio_data[i-1]) > click_threshold and \
                np.abs(audio_data[i] - audio_data[i+1]) > click_threshold:
            # Detected a click, interpolate to remove
            start = max(0, i - window_size // 2)
            end = min(len(audio_data), i + window_size // 2)
            cleaned_audio[start:end] = np.interp(
                np.arange(start, end),
                np.array([start, end]),
                np.array([audio_data[start], audio_data[end]])
            )

    return cleaned_audio

def get_word_samples(word, sample_rate):
    """
    Calculates the start and end sample indices for a given
    word with its associated transcription timestamp.

    Parameters:
    word : dict
        A dictionary containing the word and its timestamps.
    sample_rate : int
        The sample rate of the audio data.

    Returns:
    (start_sample, end_sample) : tuple
        A tuple containing the start and end sample indices for the word.
    """

    # Extract start and end times from the word timestamps, assume they are in seconds
    start_time = word['start']
    end_time = word['end']

    # Convert the start and end times to sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    return (start_sample, end_sample)


def split_silence(audio_data, silence_threshold, min_silence_length, sample_rate):
    """
    Splits audio data into segments based on silence.

    Parameters:
    audio_data : numpy.array
        The audio data.
    silence_threshold : float
        The value below which audio is considered silent.
    min_silence_length : float
        The minimum length of silence intervals to be used for splitting, in seconds.
    sample_rate : int
        The sample rate of the audio data.

    Returns:
    segments : list of tuples
        A list of tuples where each tuple contains start and end samples of non-silent segments.
    """

    # Normalize audio data to range [-1, 1] if not already
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(
            np.float32) / np.iinfo(audio_data.dtype).max

    # Convert min_silence_length to sample counts
    min_silence_samples = int(min_silence_length * sample_rate)

    # Initialize variables
    samples_above_threshold = np.abs(audio_data) > silence_threshold
    silent_samples = np.where(samples_above_threshold == False)[0]
    segments = []
    segment_start = 0

    # Identify non-silent segments
    for i in range(1, len(silent_samples)):
        if silent_samples[i] - silent_samples[i - 1] > min_silence_samples:
            segment_end = silent_samples[i - 1]
            if segment_end - segment_start > min_silence_samples:
                segments.append((segment_start, segment_end))
            segment_start = silent_samples[i]
    if segment_start < len(audio_data):
        segments.append((segment_start, len(audio_data)))

    return segments


def mute_curse_words(audio_path, sample_rate, transcription_result, curse_words_list):
    sample_audio_data, _ = load_wav_as_np_array(sample_audio_path)
    audio_data_crossfaded, _ = main_file_audio(audio_path)
    # audio_data_crossfaded = np.copy(audio_data_crossfaded).astype(np.float32)
    sample_audio_data = sample_audio_data.astype(np.float32)
    curse_words_set = set(word.lower() for word in curse_words_list)

    bar = Bar('Processing', max=len(transcription_result['segments']))

    for segment in transcription_result['segments']:
        bar.next()
        for word in segment['words']:
            if word['word'].lower() in curse_words_set:
                start_sample, end_sample = get_word_samples(word, sample_rate)

                
                # Perform linear crossfade with the sample audio
                linear_crossfade(audio_data_crossfaded, start_sample, end_sample, sample_audio_data)

    bar.finish()
    return audio_data_crossfaded


def convert_stereo(f):
    # Read the stereo audio file
    data, sample_rate = sf.read(f)
    # Check if the file is indeed stereo
    if data.ndim > 1 and data.shape[1] == 2:
        # Average the stereo channels to convert to mono
        mono_data = data.mean(axis=1)
    else:
        # If it's already mono, just assign it as is
        mono_data = data
    # Ensure the data is in float32 format
    mono_data = mono_data.astype('float32')
    # Write the mono data to a new audio file
    sf.write(f, mono_data, sample_rate,
            format='WAV', subtype='FLOAT')


def transcribe_audio(audio_file, device_type):
    model = stable_whisper.load_faster_whisper(
        'large-v3', compute_type="float16", device=device_type)
    # model = stable_whisper.load_model('large-v3', device=device_type)
    result = model.transcribe_stable(
        audio_file, beam_size=5, word_timestamps=True)
    transcript_path = f'transcript{random.randint(0, 100)}.json'
    result.save_as_json(transcript_path)
    return transcript_path


def find_curse_words(audio_file, sample_rate, transcript_file, CURSE_WORD_FILE=CURSE_WORD_FILE):
    results = process_json(transcript_file)
    curses = read_curse_words_from_csv(CURSE_WORD_FILE)
    curse_words_set = set(curses)
    return mute_curse_words(audio_file, sample_rate, results, curse_words_set)


def process_audio(audio_file, transcript_file=None):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if not transcript_file:
        transcript_file = transcribe_audio(audio_file, device_type)
    convert_stereo(audio_file)
    audio_data, sample_rate = sf.read(audio_file, samplerate=None)
    
    muted_audio = find_curse_words(
        audio_file, sample_rate, transcript_file)
    outfile = Path(audio_file).parent / \
        str(Path(audio_file).name + '_muted_audio.wav')
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # remove_clicks(muted_audio, sample_rate, 0.2)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    sf.write(outfile, muted_audio, sample_rate)
    return outfile


def process_video(input_video_path, transcript_file):
    # Define output paths
    output_audio_path = Path(input_video_path).with_suffix('.wav') 
    suf = str(Path(output_audio_path).suffix)
    audio_out = Path(Path(input_video_path).parent / "audio.wav")
    output_video_path = str(output_audio_path).replace(suf, "clean_video.mp4")
    remove(str(audio_out))
    extract_audio(input_path=input_video_path,
                  output_path=str(audio_out), output_format="wav")
    remove_clicks
    # Process audio (assuming process_audio returns a path to the processed audio file)
    return process_audio(
        str(audio_out), transcript_file)


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
