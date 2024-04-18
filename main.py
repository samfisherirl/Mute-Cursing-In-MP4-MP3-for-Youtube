"""
This script provides functionality to mute curse words in audio or video files.
It uses a transcript file to identify curse words and applies audio processing techniques to mute them.
The script supports loading an existing transcript file and selecting an audio or video file for processing.
"""
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
from mutagen.mp3 import MP3
import encrypt_cursewords_for_github as byte_curses
from split_segs import split_audio, segment_duration
import os
import threading
import scipy
import stable_whisper

segment_duration = 30
buff_ratio = 0.75

model = stable_whisper.load_faster_whisper(
    'large-v3', device="cuda" if torch.cuda.is_available() else "cpu")
print('finished loading...')

CURSE_WORD_FILE = 'curse_words.csv'
sample_audio_path = 'looperman.wav'
transcripts = ""
exports = ""
new_trans_path = Path.cwd()
new_trans_path = Path(str(new_trans_path) + "\\transcripts")
processed_paths = {}

def make_dirs():
    """returns (transcript_folder, export_folder)"""
    global new_trans_path, exports
    import shutil
    shutil.rmtree(str(new_trans_path))
    new_trans_path.mkdir(parents=True, exist_ok=True)
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
    init_dir = Path(__file__).parent / "transcripts"
    if messagebox.askyesno('Load Transcript', 'If this program crashed, this saves the transcript to ensure it doesn\'t require restarting.\n\nDo you want to load an existing transcript?'):
        # File dialog to select a transcript JSON file
        transcript_path = filedialog.askopenfilename(
            title='Select Transcript File',
            filetypes=[('JSON files', '*.json')],
            initialdir=str(init_dir)
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
    return sf.read(wav_file_path, dtype='float32')


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


def apply_fade(audio_data, start_time, end_time, sample_rate, fade_duration=0.01):
    """
    Correctly apply a fade out at the start_time and a fade in at the end_time.
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    audio_length = len(audio_data)
    fade_samples = int(fade_duration * sample_rate)

    start_sample = max(0, min(start_sample, audio_length - 1))
    end_sample = max(0, min(end_sample, audio_length - 1))

    # Correcting the fade ranges according to the conventional definitions
    # Fade out before the start (reducing volume leading into the curse)
    fade_out_start = max(0, start_sample - fade_samples)
    fade_out_range = range(fade_out_start, start_sample)

    # Fade in after the end (increasing volume after the curse)
    fade_in_start = end_sample
    fade_in_range = range(fade_in_start, min(
        fade_in_start + fade_samples, audio_length))


    # Apply fade out
    for i, sample in enumerate(fade_out_range):
        fade_out_factor = i / len(fade_out_range)  # Linear fade
        audio_data[sample] *= fade_out_factor

    # Apply fade in
    for i, sample in enumerate(fade_in_range):
        fade_in_factor = 1 - (i / len(fade_in_range))  # Linear fade
        audio_data[sample] *= fade_in_factor

    fade_timestamps = {
        'fade_out_start': max(0, start_time - fade_duration),
        'fade_out_end': start_time,
        'fade_in_start': end_time,
        'fade_in_end': min(end_time + fade_duration, audio_length / sample_rate)
    }

    return audio_data, fade_timestamps


def split_silence(sample_rate, word, audio_data, buffer_duration=0.01):
    """
    Split silence into start and end indices with a buffer. Adjust the buffer duration as needed.

    @param sample_rate - Sampling rate of the sound in Hz.
    @param word - Word being split. Must contain 'start' and 'end' keys.
    @param buffer_duration - Additional duration in seconds to add as a buffer before and after the curse word.

    @return Tuple of start and end indices including the buffer. 
    """
    global buff_ratio 
    # Example adjustment, ensure this is defined or adjusted as needed in your context.
    buff_ratio = 0.95

    start_time = word['start'] - buffer_duration
    end_time = word['end'] - buffer_duration
    audio_data_faded, fade_dict = apply_fade(
        audio_data,
        start_time,
        end_time,
        sample_rate
    )
    start_time = fade_dict['fade_out_end']
    end_time = fade_dict['fade_in_start']

    start_sample = max(0, int(start_time * sample_rate))
    end_sample = min(int(end_time * sample_rate), len(audio_data_faded) - 1)
    return start_sample, end_sample


def mute_curse_words(audio_data, sample_rate, transcription_result, curse_words_list):
    """
    Mute curse words in the audio data.
    """
    audio_data_muted = np.copy(audio_data)
    curse_words_set = set(word.lower() for word in curse_words_list)

    for word in transcription_result:
        if word['word'].lower() in curse_words_set:
            start_sample, end_sample = split_silence(
                sample_rate, word, audio_data_muted)
            # Mute the section
            audio_data_muted[start_sample:end_sample] = 0
            # Apply fade in and fade out to the modified section
    return audio_data_muted


def convert_stereo(f):
    """
     Convert a stereo WAV file to mono. This is a convenience function for reading and writing a stereo WAV file.

     @param f - Path to the file to be converted ( must be in WAV format
    """
    data, sample_rate = sf.read(f)
    if data.ndim > 1 and data.shape[1] == 2:
        mono_data = data.mean(axis=1)
    else:
        mono_data = data
    mono_data = mono_data.astype('float32')
    return mono_data, sample_rate


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


def transcribe_audio(audio_file, segnum):
    """
     Transcribe audio to a device. This is a wrapper around whisper. load_faster_whisper and whisper. load_model to get a model and transcribe the audio to the device.

     @param audio_file - Path to audio file. Must be a file - like object.
     @param device_type - Device type to transcribe to.

     @return Path to JSON file that contains the transcript of the audio file. If there is no transcript it will be None
    """
    global transcripts, exports, new_trans_path, model

    # model = stable_whisper.load_model('large-v3', device=device_type)
    result = model.transcribe_stable(
        audio_file,
        word_timestamps = True,
        language = 'en',
        beam_size = 5,
        vad_filter = True,
        vad_parameters = dict(min_silence_duration_ms=500)
    )

    dmt_ = dmt()
    new_trans_file = new_trans_path / (f"transcript{dmt_}_segnum_{segnum}.json")

    result.save_as_json(str(new_trans_file))
    process_json(new_trans_file)
    return new_trans_file


def manage_trans(audio_file, transcript_file=None, segnum=0):
    """
     Manage Transcripts. This function is used to manage the transcripts. It will save the transcript to a file and return the path to the file.

     @return Path to the transcript file
    """
    if not transcript_file:
        print('transcribing')
        transcript_file = transcribe_audio(
            audio_file, segnum)
    return audio_file, transcript_file


def process_audio(audio_file, iteration, transcript_file=None):
    """
     Process audio and transcribe it to wav. This is the main function of the program. It takes the audio file and transcribes it using transcript_file if it is not provided.

     @param audio_file - path to audio file to be transcribed
     @param transcript_file - path to transcript file. If not provided it will be transcribed

     @return path to audio file with processed
    """
    global processed_paths
    print('converting to stereo')

    print('reading audio')
    audio_data, sample_rate = convert_stereo(audio_file)
    print('process json')
    results = process_json(transcript_file)
    print('find curse words')
    muted_audio = find_curse_words(
        audio_data, sample_rate, results)
    outfile = Path(audio_file).parent / \
        str(Path(audio_file).name + '_muted_audio.wav')
    print('curse words removed, now removing clicks')
    print('exporting file now....')
    sf.write(outfile, muted_audio, sample_rate)
    processed_paths[iteration] = outfile


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
    if audio_out.exists():
        remove(str(audio_out))
    extract_audio(input_path=input_video_path,
                  output_path=str(audio_out), output_format="wav")
    # Process audio (assuming process_audio returns a path to the processed audio file)
    return process_audio(
        str(audio_out), transcript_file)


def process_audio_batch(trans_audio):
    max_threads = 5
    threads = []
    processed_paths = {}

    def wait_for_threads(threads):
        for thread in threads:
            thread.join()
        threads.clear()

    threadnumb = 0
    for trans, audio in trans_audio.items():
        threadnumb += 1

        if len(threads) >= max_threads:
            wait_for_threads(threads)

        # Sample way to track processed paths. Implement according to the actual process_audio
        processed_paths[threadnumb] = f"{audio}_processed"

        thread = threading.Thread(
            target=process_audio, args=(audio, threadnumb, trans))
        threads.append(thread)
        thread.start()

    # Wait for the remaining threads
    wait_for_threads(threads)
    return processed_paths


def main():
    """
     Main function of the program. Shows dialogue for fileselect video/audio and transcript files. Loads and processes the transcript file to determine the type of audio and / or video
    """
    global transcripts, exports, processed_paths, segment_duration, buff_ratio
    root = Tk()
    big_frame = ttk.Frame(root)
    big_frame.pack(fill="both", expand=True)
    root.withdraw()
    root.attributes('-topmost', True)
    transcript_file = load_transcript()
    in_av_path = select_audio_or_video()
    transcripts, exports = make_dirs()
    if Path(in_av_path).suffix == '.mp4':
        # process video involves extracting audio and reencoding
        result = process_video(in_av_path, transcript_file)
        song = Path(in_av_path).parent / result
        replace_audio(in_av_path, song)
    else:
        if in_av_path.endswith('.mp3') or in_av_path.endswith('.wav'):
            output_dir = os.path.dirname(os.path.abspath(in_av_path))
            output_dir = create_new_subfolder_from_path(
                os.path.abspath(in_av_path))
            segments = split_audio(in_av_path, output_dir, segment_duration)
            trans_audio = {}
            segnum = 0
            
            for seg in segments:
                segnum += 1
                if segment_duration*segnum > 6:
                    buff_ratio = 0.75
                audio, trans = manage_trans(seg, transcript_file, segnum)
                trans_audio[trans] = audio
            threads = []
            threadnumb = 0
            for trans, audio in trans_audio.items():
                threadnumb += 1
                # Create a new thread for each split audio file
                thread = threading.Thread(
                    target=process_audio, args=(audio, threadnumb, trans))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Combine all processed segments into a single file
            # for iteration, seg in processed_paths:
            combined_audio = []
            for i in range(1, threadnumb+1, 1):
                try:
                    seg = processed_paths[i]
                except Exception as e:
                    break
                try:
                    audio_data, sample_rate = sf.read(str(seg))
                    combined_audio.append(audio_data)
                except sf.LibsndfileError as e:
                    print(f"Error reading file {seg}: {str(e)}")
                    # Skip the file and continue with the next one

            if combined_audio:
                combined_audio = np.concatenate(combined_audio)
                combined_file = Path(in_av_path).with_stem(
                    Path(in_av_path).stem + '_combined_muted').with_suffix('.wav')
                sf.write(combined_file, combined_audio, sample_rate)
                print(f"Combined processed audio exported to: {combined_file}")
            else:
                print(
                    "No processed audio files were successfully read. Skipping concatenation.")


if __name__ == "__main__":
    main()
