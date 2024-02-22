"""
This script provides functionality to mute curse words in audio or video files.
It uses a transcript file to identify curse words and applies audio processing techniques to mute them.
The script supports loading an existing transcript file and selecting an audio or video file for processing.
"""
import stable_whisper
import numpy as np
import soundfile as sf
import csv
from tkinter import Tk, messagebox, filedialog
import tkinter as ttk
import random
from audio_extract import extract_audio
from pathlib import Path
import torch
from progress.bar import Bar
import wave
from os import remove
from process_files import remove_clicks
from process_files import *


# Define paths and file names
CURSE_WORD_FILE = 'curse_words.csv'
sample_audio_path = 'looperman.wav'
transcripts = ""
exports = ""
new_trans_path = Path.cwd()
new_trans_path = Path(str(new_trans_path) + "\\transcripts")


def make_dirs():
    """returns (transcript_folder, export_folder)"""
    global new_trans_path, exports
    dmt_ = dmt()
    new_trans_path.mkdir(parents=True, exist_ok=True)
    new_trans_path = new_trans_path / (f"transcript{dmt_}.json")
    exports = Path(cwd / "exports").mkdir(parents=True, exist_ok=True)
    return (new_trans_path, exports)


def dmt():
    day = datetime.now().strftime('%d')
    mo = datetime.now().strftime('%m')
    time = datetime.now().strftime('%H-%M-%S')
    return f'{day}-{mo}-{time}'


def load_transcript():
    """
     Load Transcript if program crashed this saves the transcript to ensure it doesn't require restarting
     
     
     @return path to transcript or
    """
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
    """
     Select audio or video file. This function is used to select an audio or video file. The file is selected by the user and returned to the program
     
     
     @return path to audio or
    """
    # File dialog to select an audio file
    av_path = filedialog.askopenfilename(
        title='Select A/V files',
        filetypes=[('A/V files', '*.mp3 *.wav *.mp4')]
    )
    if av_path:
        print(f'Audio/Video file selected: {av_path}')
        return av_path
    return None



def read_curse_words_from_csv(CURSE_WORD_FILE):
    """
     Read curse words from CSV file. This is a list of words that are part of CURIE's word list
     
     @param CURSE_WORD_FILE - Path to file to read
     
     @return List of words in CURIE's word list ( column A ) as defined in CSV file
    """
    curse_words_list = []
    with open(CURSE_WORD_FILE, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Assuming curse words are in column A
            curse_words_list.append(row[0])
    return curse_words_list


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
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

            # Normalize the audio data
            audio_data /= np.iinfo(np.int16).max

            # Return the audio data and the sample rate
            return audio_data, wav_file.getframerate()
    except wave.Error as e:
        print(f"An error occurred while reading the WAV file: {wav_file_path}")
        print(e)
    return sf.read(wav_file_path, dtype='float64')


def get_word_samples(word, sample_rate):
    """
    Get start and end sample indices from a word. This is a helper function for get_word_samples and get_word_samples_with_time_range.
    
    @param word - The word to get samples from. Should have'start'and'end'fields.
    @param sample_rate - The sample rate in Hz.
    
    @return A tuple of start and end sample indices for the word in time units of the sample_rate passed
    """
    start_time = word['start']
    end_time = word['end']

    # Convert the start and end times to sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    return (start_sample, end_sample)


def apply_fade(audio_data, start_sample, end_sample, sample_rate, fade_duration=0.001):
    """
     Apply an exponential fade to the audio data. Fade is applied in the range [ start_sample end_sample ] and out the range [ start_sample end_sample + 1 ]
     
     @param audio_data - The audio data to be faded
     @param start_sample - The sample at which the fade starts
     @param end_sample - The sample at which the fade ends
     @param sample_rate - The sample rate of the audio data
     @param fade_duration - The duration of the fade in seconds
     
     @return The audio data with the fade applied to the start and end samples as well as the sample rate of
    """
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
    """
     Split silence into start and end indices. This is used to determine the sample indices that will be used for silencing a sound.
     
     @param sample_rate - Sampling rate of the sound in Hz.
     @param word - Word that is being split. Must contain'start'and'end'keys.
     
     @return Tuple of start and end indices. The first index is the start index the second is the end index
    """
    # Calculate the start and end sample indices
    start_sample = int(word['start'] * sample_rate)
    end_sample = int(word['end'] * sample_rate)
    if (end_sample - start_sample) < 3000:
        start_sample = start_sample - 1000
        end_sample = end_sample + 1000
    return start_sample, end_sample


def mute_curse_words(audio_data, sample_rate, transcription_result, curse_words_list):
    """
     Mute curse words by setting the amplitude. This is a helper function for : func : ` get_audio_data `
     
     @param audio_data - Audio data to be muted
     @param sample_rate - Sample rate of the audio data in Hz
     @param transcription_result - List of dictionaries containing information about the transcription
     @param curse_words_list - List of words that should be muted
     
     @return A list of indices for the audio data that have been muted in the time domain and the mute
    """
    # Create a copy of the audio data to avoid modifying the original
    audio_data_muted = np.copy(audio_data)
    # Create a set for faster membership testing
    curse_words_set = set(word.lower() for word in curse_words_list)
    bar = Bar('Processing', max=len(transcription_result))

    # Initialize an empty list to store the start and end sample indices for muting
    mute_indices = []
    # Go through each segment in the transcription result
    for word in transcription_result:
        bar.next()
        if word['word'] in curse_words_set:
            # Check if the word is in the curse words set
            start_sample, end_sample = split_silence(sample_rate, word)
            # Apply fade-in before muting
            audio_data_muted = apply_fade(audio_data_muted, start_sample,
                        end_sample, sample_rate)
            # Mute the curse words by setting the amplitude to zero
            audio_data_muted[start_sample:end_sample] = 0
            # Apply fade-out after muting
            audio_data_muted = apply_fade(audio_data_muted, start_sample,
                        end_sample, sample_rate)
    bar.finish()
    return audio_data_muted


def convert_stereo(f):
    """
     Convert a stereo WAV file to mono. This is a convenience function for reading and writing a stereo WAV file.
     
     @param f - Path to the file to be converted ( must be in WAV format
    """
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
    """
     Transcribe audio to a device. This is a wrapper around whisper. load_faster_whisper and whisper. load_model to get a model and transcribe the audio to the device.
     
     @param audio_file - Path to audio file. Must be a file - like object.
     @param device_type - Device type to transcribe to.
     
     @return Path to JSON file that contains the transcript of the audio file. If there is no transcript it will be None
    """
    global transcripts, exports, new_trans_path
    model = stable_whisper.load_faster_whisper(
        'large-v3', device=device_type)
    # model = stable_whisper.load_model('large-v3', device=device_type)
    result = model.transcribe_stable(
        audio_file, word_timestamps=True) 
    result.save_as_json(str(new_trans_path))
    return new_trans_path


def find_curse_words(audio_content, sample_rate, results, CURSE_WORD_FILE=CURSE_WORD_FILE):
    """
     Find Curse words in audio content. This is a wrapper around mute_curse_words that takes into account the sample rate in order to get an accurate set of cursors and returns a set of words that are present in the audio
     
     @param audio_content - The audio content to search
     @param sample_rate - The sample rate in Hz
     @param transcript_file - The file containing the transcripts.
     @param CURSE_WORD_FILE - The path to the CSV file containing curse words.
     
     @return The set of words that are present in the audio content and are not present in the transcript. This set is used to make sure that we don't accidentally miss a word
    """
    curses = read_curse_words_from_csv(CURSE_WORD_FILE)
    curse_words_set = set(curses)
    return mute_curse_words(audio_content, sample_rate, results, curse_words_set)


def process_audio(audio_file, transcript_file=None):
    """
     Process audio and transcribe it to wav. This is the main function of the program. It takes the audio file and transcribes it using transcript_file if it is not provided.
     
     @param audio_file - path to audio file to be transcribed
     @param transcript_file - path to transcript file. If not provided it will be transcribed
     
     @return path to audio file with processed
    """
    
    if not transcript_file:
        transcript_file = transcribe_audio(
            audio_file, device_type="cuda" if torch.cuda.is_available() else "cpu")
        
    convert_stereo(audio_file)
    audio_data, sample_rate = sf.read(audio_file, samplerate=None, dtype='float64')
    
    results = process_json(transcript_file)
    muted_audio = find_curse_words(
        audio_data, sample_rate, results)
    outfile = Path(audio_file).parent / \
        str(Path(audio_file).name + '_muted_audio.wav')
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    remove_clicks(muted_audio, sample_rate, 0.5)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    sf.write(outfile, muted_audio, sample_rate)
    return outfile


def process_video(input_video_path, transcript_file):
    """
     Process a video and return a path to the processed audio. This is a wrapper around process_audio that removes extraneous files and converts the audio to mp4
     
     @param input_video_path - The path to the video to process
     @param transcript_file - The path to the transcript file to process
     
     @return The path to the processed audio file or None if there was an error processing the audio ( in which case the error will be logged
    """
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
    """
     Main function of the program. Shows dialogue for fileselect video/audio and transcript files. Loads and processes the transcript file to determine the type of audio and / or video
    """
    global transcripts, exports 
    root = Tk()
    big_frame = ttk.Frame(root) 
    big_frame.pack(fill="both", expand=True)
    root.withdraw()
    root.attributes('-topmost', True)
    transcript_file = load_transcript()
    input_video_path = select_audio_or_video()
    transcripts, exports = make_dirs()
    cwd = Path(input_video_path).parent
    if Path(input_video_path).suffix == '.mp4':
        result = process_video(input_video_path, transcript_file)
        song = Path(input_video_path).parent / result
    else:
        # Process audio only
        process_audio(input_video_path, transcript_file)


if __name__ == "__main__":
    main()
