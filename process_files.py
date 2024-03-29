from moviepy.editor import VideoFileClip, AudioFileClip
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import soundfile as sf
import csv
from pydub import AudioSegment
import math


cwd = Path(__file__).parent


def split_mp3(file_path):
    # Convert MP3 to WAV for processing
    audio = AudioSegment.from_mp3(file_path)
    audio.export("temp.wav", format="wav")

    # Load the full audio file
    data, samplerate = sf.read("temp.wav")
    duration_in_seconds = len(data) / samplerate
    segment_duration = 3600  # 1 hour in seconds
    number_of_segments = math.ceil(duration_in_seconds / segment_duration)
    segment_paths = []

    for i in range(number_of_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, duration_in_seconds)
        start_sample = int(start_time * samplerate)
        end_sample = int(end_time * samplerate)
        segment_data = data[start_sample:end_sample]

        segment_file_path = f"segment_{i+1}.wav"
        sf.write(segment_file_path, segment_data, samplerate)
        segment_paths.append(segment_file_path)
        print(f"Segment {i+1} written to {segment_file_path}")

    # Remove the temporary WAV file
    Path("temp.wav").unlink()

    return segment_paths


def load_saved_transcript(json_file_path):
    with open(json_file_path, 'r') as file:
        transcript = json.load(file)
    return transcript

# Function to recursively convert all strings in a JSON object to lowercase

def main_file_audio(wav_file_path):
    # Read the audio file
    audio_data, sample_rate = sf.read(wav_file_path, dtype='float32')

    # Ensure that the audio file is mono
    if audio_data.ndim > 1:
        # Average the channels if more than one channel (i.e., stereo)
        audio_data = np.mean(audio_data, axis=1)


def load_saved_transcript(json_file_path):
    with open(json_file_path, 'r') as file:
        transcript = json.load(file)
    return transcript

    # 'soundfile' already reads into 'float32', and the data is typically normalized
    # If you need to ensure normalization between -1.0 and 1.0, uncomment the following lines:
    # peak = np.max(np.abs(audio_data))
    # if peak > 1:
    #     audio_data /= peak
    return audio_data, sample_rate


def to_lowercase(input):
    if isinstance(input, dict):
        return {k.lower().strip("',.\"-_/` ").strip(): to_lowercase(v).strip("',.\"-_/` ").strip() for k, v in input.items()}
    elif isinstance(input, list):
        return [to_lowercase(element) for element in input]
    elif isinstance(input, str):
        return input.lower().strip("',.\"-_/` ").strip()
    else:
        return input


def process_json(infile):
    global new_trans_file
    # Read the original JSON file
    with open(infile, 'r', errors="replace", encoding='utf-8') as f:
        data = json.load(f)
    # Convert all strings to lowercase
    words = []
    try:
        words = [{'word': word['word'].strip("',.\"-_/`").lower().strip(), 'start': word['start'], 'end': word['end']}
                for segment in data['segments'] for word in segment['words']]
            # Write the modified JSON to a new file
        with open(infile, 'w') as file:
            json.dump(words, file, indent=4)
    except Exception as e:
        words = data
    # Read the original JSON file
    return words


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


def read_curse_words_from_csv(CURSE_WORD_FILE):
    curse_words_list = []
    with open(CURSE_WORD_FILE, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Assuming curse words are in column A
            curse_words_list.append(row[0])
    return curse_words_list

# Function to mute curse words in the audio


def replace_audio(mp4_path, wav_path):
    # Create a Path object from the mp4 path
    original_video_path = Path(mp4_path)

    # Get the directory and the name without extension
    directory = original_video_path.parent
    original_name = original_video_path.stem

    # Define the new output path
    output_path = directory / f"{original_name}_cleaned.mp4"

    # Convert Path objects to strings before passing to moviepy
    video_clip = VideoFileClip(str(mp4_path))
    audio_clip = AudioFileClip(str(wav_path))

    # Ensure the audio is the same duration as the video
    if audio_clip.duration != video_clip.duration:
        raise ValueError(
            "The durations of the video and audio files do not match.")

    # Set the audio of the video clip to the new audio
    video_clip = video_clip.set_audio(audio_clip)

    # Write the result to a file
    video_clip.write_videofile(
        str(output_path), codec='libx264', audio_codec='aac')

