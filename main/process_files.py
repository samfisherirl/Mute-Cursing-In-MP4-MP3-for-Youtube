from tkinter import filedialog
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip
from pathlib import Path
import json
import datetime
import numpy as np
import soundfile as sf
import csv
from pydub import AudioSegment
import math
import wave
import tkinter as tk
from tkinter import filedialog
import json
import subprocess
import re 

cwd = Path(__file__).parent


def load_wav_as_np_array(wav_file_path):
    """
     Load a WAV file and return the audio data as NumPy array. This function is used to load mono wav files that are stored in a file system.
     
     @param wav_file_path - The path to the WAV file
     
     @return A tuple containing the audio data and the sample
    """
    # Open the audio file
    try:
        with wave.open(wav_file_path, "rb") as wav_file:
            # Ensure that the audio file is mono
            if wav_file.getnchannels() != 1:
                raise ValueError("Only mono audio files are supported.")

            # Extract audio frames
            frames = wav_file.readframes(wav_file.getnframes())

            # Convert audio frames to float32 NumPy array
            audio_data = np.frombuffer(
                frames, dtype=np.int16).astype(np.float32)

            # Normalize the audio data
            audio_data /= np.iinfo(np.int16).max

            # Return the audio data and the sample rate
            return audio_data, wav_file.getframerate()
    except wave.Error as e:
        print(f"An error occurred while reading the WAV file: {wav_file_path}")
        print(e)
    return sf.read(wav_file_path, dtype='float64')


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


def create_new_subfolder_from_path(path):
    # Convert the path to a Path object if it's not already
    path = Path(path)

    # Extract the parent directory
    parent_dir = path.parent

    # Get the current time and format it as day-month-time
    # Note: for time, we're using hour-minute-second format to avoid using ':', which is not allowed in folder names
    timestamp = datetime.datetime.now().strftime("%d-%m-%H%M%S")

    # Extract the original filename (without extension)
    original_filename = path.stem

    # Combine everything to create the new folder name
    new_folder_name = f"{timestamp}-{original_filename}"
    new_folder_path = parent_dir / new_folder_name

    # Create the new subfolder
    new_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"New folder created at: {new_folder_path}")
    return new_folder_path


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
    words = []
    with open(infile, 'r', errors="replace", encoding='utf-8') as f:
        data = json.load(f)
    try:
        # Assuming the structure is a list of words with 'word', 'start', and 'end'
        words = [{'word': word['word'].strip(
            "',.\"-_/` ").lower(), 'start': word['start'], 'end': word['end']} for word in data]

        # Rewrite the modified JSON back to the same file
        with open(infile, 'w', encoding='utf-8') as file:
            json.dump(words, file, indent=4)
    except Exception as e:
        print(f"Error processing JSON file: {e}")
        words = data  # Fallback to the original data in case of error
    return words


def convert_json_format(input_filename, output_filename):
    """
    Converts a JSON file from a complex nested structure to a simplified
    structure focusing on words, their start, and end times.
    
    @param input_filename: Path to the input JSON file.
    @param output_filename: Path where the converted JSON is saved.
    """
    with open(input_filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    simplified_data = []
    for segment in data.get('segments', []):
        for word_info in segment.get('words', []):
            simplified_data.append({
                "word": word_info['word'].strip(r"',.\"-_/`?!; ").lower(),
                "start": word_info['start'],
                "end": word_info['end']
            })

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(simplified_data, outfile, indent=4)

    print(
        f'The data has been successfully converted and saved to: {output_filename}')
    return simplified_data


SILENCE_GAP = 100  # Silence gap threshold in milliseconds


def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()
    return file_path


def process_srt(srt_file):
    gaps = []
    with open(srt_file, 'r') as file:
        content = file.read()
        timestamps = re.findall(
            r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', content)
        prev_end_time = 0
        for timestamp in timestamps:
            start_time, end_time = timestamp.split(' --> ')
            start_time_ms = convert_to_ms(start_time)
            end_time_ms = convert_to_ms(end_time)
            if start_time_ms - prev_end_time > 200:
                gaps.append((convert_to_ffmpeg_time(prev_end_time),
                            convert_to_ffmpeg_time(start_time_ms)))
            prev_end_time = end_time_ms

    return gaps


def convert_to_ms(s):
    hours, minutes, seconds = s[:-4].split(':')
    milliseconds = int(s[-3:])
    return (int(hours) * 3600000) + (int(minutes) * 60000) + (int(seconds) * 1000) + milliseconds


def convert_to_ffmpeg_time(ms):
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    milliseconds = ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def merge_crops(video_file, gaps):
    commands = []
    for i, (start, end) in enumerate(gaps):
        commands.append(
            f"ffmpeg -i {video_file} -ss {start} -to {end} -c copy output_{i}.mp4")
    commands.append(
        "ffmpeg -f concat -safe 0 -i inputs.txt -c copy output.mp4")

    with open('inputs.txt', 'w') as f:
        for i in range(len(gaps)):
            f.write(f"file 'output_{i}.mp4'\n")

    for command in commands:
        subprocess.run(command, shell=True)

def crop_video(input_srt):
    print("Select the input video file to process:")
    input_video = select_file()
    if not input_video:
        print("No video file selected. Exiting.")
        return
    merge_crops(input_video,
                    process_srt(input_srt))



def load_json(infile):
    try:
        with open(infile, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [{'word': word['word'].strip(
            "',.\"-_/` ").lower(), 'start': word['start'], 'end': word['end']} for word in data]
    except Exception as e:
        print("Error reading JSON:", e)
        return []
    return data


if __name__ == "__main__":
    # Split the MP3 file into segments
    c = r'C:\Users\dower\Downloads\transcription.json'
    r = convert_json_format(c, f'{c}2.json')

    print()
    # Process the JSON file
