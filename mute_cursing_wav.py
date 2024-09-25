from tkinter.colorchooser import askcolor
import stable_whisper
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import subprocess
import shutil
import os
import moviepy.editor as mp
from process_files import *
from censorship import *
import re
from datetime import datetime, timedelta

MODEL_SIZE = 'large-v3'
SPLIT_IN_MS = 60
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def clean_path(path_str):
    path = Path(path_str)
    clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', path.stem)
    clean_name = re.sub(r'_+', '_', clean_name)
    return path.with_stem(clean_name)

def split_audio(audio_file, output_dir, segment_duration=SPLIT_IN_MS):
    """
    Splits an audio file into segments of a specified duration using ffmpeg,
    and saves them in the provided output directory. Returns a list of paths
    to the generated segments.
    
    Args:
        audio_file (str): Path to the input audio file.
        output_dir (str): Path to the directory where the segments will be saved.
        segment_duration (int): Duration of each audio segment in seconds.
        
    Returns:
        List[str]: List of paths to the generated audio segment files.
    """

    audio_path = clean_path(audio_file)
    output_dir = audio_path.parent
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_pattern = str(
        output_dir / f"{audio_path.stem}_{timestamp}_%03d.wav")

    cmd = [
        'ffmpeg',
        '-i', str(audio_path),
        '-f', 'segment',
        '-y',
        '-segment_time', str(segment_duration),
        '-c', 'copy',
        '-vn',  # Exclude video
        output_pattern
    ]

    try:
        result = subprocess.run(
            cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print(f"Audio has been successfully split and saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to split audio: {e.stderr}")
        return []

    # Generate the list of file paths for the new audio segments
    segment_files = sorted(output_dir.glob(
        f"{audio_path.stem}_{timestamp}_*.wav"))
    return [str(file) for file in segment_files]


def video_to_audio(video_file, audio_file):
    # Load the video file
    video = mp.VideoFileClip(video_file)

    # Extract the audio
    audio = video.audio

    # Write the audio to an audio file
    audio.write_audiofile(audio_file)


def choose_color_hex():
    """
    Opens a color chooser dialog and returns the selected color as a hex string.
    If the user cancels the dialog, returns None.
    """
    # Initialize Tkinter's root window to avoid explicit window creation
    root = tk.Tk()
    root.withdraw()  # Hide the main window as we only need the dialog

    # Open the color chooser and capture the selected color
    color = askcolor()

    # Destroy the root window after selection
    root.destroy()

    if color[1]:
        return color[1]  # Return the hex value of the selected color
    else:
        return None  # Return None if the dialog was canceled


def select_audio_or_video():
    """
    Select audio or video file. This function is used to select an audio or video 
    file. The file is selected by the user and returned to the program.

    @return path to audio or video file.
    """
    while True:
        root = tk.Tk()
        root.withdraw()  # Hide main window
        root.call('wm', 'attributes', '.', '-topmost', '1')

        # File dialog to select an audio or video file
        av_path = filedialog.askopenfilename(
            title='Select A/V files',
            filetypes=[('A/V files', '*.mp3 *.wav *.mp4')]
        )
        root.destroy()
        video_bi = {}
        video_bi['status'] = False
        video_path = ''
        if "mp4" in av_path or "mov" in av_path:
            video_path = av_path
            av_path = convert_video_to_audio(
                av_path, av_path.replace(".mp4", ".wav"))
            video_bi['status'] = True
            video_bi['path'] = video_path
        if av_path:
            print(f'Audio/Video file selected: {av_path}')
            folder = Path(av_path).parent / Path(av_path).stem
            folder.mkdir(parents=True, exist_ok=True)
            try:
                os.remove(folder)
            except Exception as e:
                print(str(e))
            folder.mkdir(parents=True, exist_ok=True)
            av_new = str(folder / Path(av_path).name)
            shutil.copy(av_path, clean_path(av_new))
            return av_new, video_bi
        return None, video_bi['status']


def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return Path(folder_path)



def convert_video_to_audio(video_file, audio_output):
    cmd = [
        "ffmpeg",
        '-y',
        "-i", video_file,
        "-vn",  # No video.
        "-acodec", "pcm_s16le",  # Set codec to PCM s16 le
        "-ar", "44100",  # Audio sample rate
        "-ac", "2",  # Stereo
        audio_output
    ]
    subprocess.run(cmd, check=True)
    return audio_output


def remove_audio_from_video(video_file, video_output):
    cmd = [
        "ffmpeg",
        '-y',  # Automatically overwrite output files without asking
        "-i", video_file,
        "-c:v", "copy",  # Copy video without re-encoding
        "-an",  # Remove audio
        video_output
    ]
    subprocess.run(cmd, check=True)


def add_audio_to_video(video_file, audio_file, output_video):
    # Generate a temporary video file path without audio
    video_no_audio = video_file.replace('.mp4', "temp_.mp4")

    # First, remove audio from the original video
    remove_audio_from_video(video_file, video_no_audio)

    # Then, add new audio to the video without audio
    if os.path.exists(output_video):
        os.remove(output_video)
    cmd = [
        "ffmpeg",
        '-y',  # Automatically overwrite output files without asking
        "-i", video_no_audio,
        "-i", audio_file,
        "-c:v", "copy",  # Copy video without re-encoding
        "-c:a", "aac",  # Encode audio to AAC
        "-strict", "experimental",
        output_video
    ]
    subprocess.run(cmd, check=True)

    # Clean up temporary video file
    os.remove(video_no_audio)

class AudioTranscriber:
    def __init__(self, model_size='large-v3', device='cuda'):
        """
        Initialize the transcriber with a specific model size and device.
        """
        try:
            self.model = stable_whisper.load_model(model_size,
                                      device=device)
        except Exception as e:
            print(f"Error loading model: {e}\n\nDont panick, I got this. \n\nYou should really use nvidia for good results.")
            self.model = stable_whisper.load_model('base', device='cpu')
        self.audio_paths = []
        self.index = len(self.audio_paths) - 1
        self.clean_audio_paths = []
        self.srt_paths = []
        self.srt_paths_small = []
        self.clean_json = ""
        self.clean_json_paths = []
        self.srt_small = ""

    def add_time(self, time_str, minutes=1):
        """Add minutes to SRT timestamp, adjusting for fractional seconds."""
        base_time = datetime.strptime(time_str.split(',')[0], '%H:%M:%S')
        milliseconds = int(time_str.split(',')[1]) if ',' in time_str else 0
        added_time = base_time + \
            timedelta(minutes=minutes, milliseconds=milliseconds)
        return added_time.strftime('%H:%M:%S,') + f"{milliseconds:03d}"

    def srt_combine(self, paths):
        combined_content = ''
        subtitle_number = 1
        additional_minutes = 0
        for index, file_path in enumerate(paths):
            if index > 0:
                combined_content += '\n'
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.strip().isdigit():
                    lines[i] = f"{subtitle_number}\n"
                    subtitle_number += 1
                elif '-->' in line:
                    start_time, end_time = line.split(' --> ')
                    lines[i] = f"{self.add_time(start_time, additional_minutes)} --> {self.add_time(end_time, additional_minutes)}\n"
                combined_content += lines[i]
                i += 1
            additional_minutes += 1
            combined_content += '\n'
        name = Path(paths[0]).stem
        output_file_prt = Path(paths[0]).parent.parent / f'{name}.srt'
        plain_text = ""
        with open(str(output_file_prt), 'w', encoding='utf-8') as file:
            file.write(combined_content.strip())

    def transcribe_audio(self, audio_path, language='en', beam_size=5):
        """
        Transcribe the given audio file and return the transcription result.
        """
        self.audio_paths.append(audio_path)
        self.json_paths = []
        return self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language=language
            # beam_size=beam_size
            # vad_filter=False,
        )

    def save_transcription(self, audio_path, result, small=False):
        """
        Save the transcription to .srt and .json files based on the audio file path.
        """
        # Determine file paths for the outputs
        if small:
            audio_path = audio_path.replace(".wav", "_small.wav")
        if "wav" in audio_path:
            srt_path = audio_path.replace(".wav", ".srt")
            json_path = audio_path.replace(".wav", ".json")
            ass_path = audio_path.replace(".wav", ".ass")
        else:
            srt_path = audio_path.replace(".mp3", ".srt")
            json_path = audio_path.replace(".mp3", ".json")
            ass_path = audio_path.replace(".mp3", ".ass")
        print('outputting transcript files')
        # Write transcription to .srt file
        tag = (
            '<font color="#f9a100">', '</font>')
        if not small:
            result.to_srt_vtt(srt_path, tag=tag)
            
        else: 
            result.to_srt_vtt(srt_path, word_level=False)
            self.srt_small = srt_path
        # Prepare transcription data for JSON export
        result.to_txt(f'{srt_path}'.replace('.srt', '.txt'))
        
        result.save_as_json(json_path)
        self.json_path = json_path
        print('completed transcript files')
        if small:
            self.srt_paths_small.append(srt_path)
        else:
            self.srt_paths.append(srt_path)

    def censor_cursing(self, audio_path):
        return process_audio(audio_path, self.json_path)

    def transcribe_and_censor(self, audio_path):
        """
        Process an audio file, transcribe it and save the results.
        """
        result = self.transcribe_audio(audio_path)
        resultSmall = result
        result.split_by_length(max_chars=55)
        self.save_transcription(audio_path, result)
        resultSmall.split_by_length(max_chars=42)
        self.save_transcription(audio_path, resultSmall, True)
        aud, self.clean_json = self.censor_cursing(audio_path)
        self.clean_audio_paths.append(aud
            )
        self.clean_json_paths.append(self.clean_json)

def main():
    global transcript_paths
    transcript_paths = []
    print('loading model')
    transcriber = AudioTranscriber(model_size=MODEL_SIZE, device='cuda')
    print('finished')
    audio_path, video_ = select_audio_or_video()
    log_ = JSONLog(audio_path)
    enums = split_audio(audio_path, 'output')
    if enums:
        for counter, audio_path in enumerate(enums):
            print("wav_file_path type:", type(audio_path))
            print("wav_file_path content:", audio_path)
            print(
                f'\n\nProcessing {audio_path}...\n@@@@@@@@@@@@@@@@@@@\nindex {counter+1} of {len(enums)}\n@@@@@@@@@@@@@@@@@@@\n')
            transcriber.transcribe_and_censor(audio_path)
    else:
        print(f'Processing {audio_path}...')
        transcriber.transcribe_and_censor(audio_path)
    comb_path = combine_wav_files(transcriber.clean_audio_paths)
    transcriber.srt_combine(transcriber.srt_paths)
    transcriber.srt_combine(transcriber.srt_paths_small)
    if video_['status']:
        add_audio_to_video(video_['path'], comb_path, video_['path'].replace('.mp4', '_clean.mp4'))
    
if __name__ == '__main__':
    main()