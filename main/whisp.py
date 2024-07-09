from tkinter.colorchooser import askcolor
import stable_whisper
import json
from faster_whisper import WhisperModel
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import subprocess
import shutil
import os
from openai import OpenAI
from datetime import datetime
import moviepy.editor as mp
from process_files import *
from censorship import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def split_audio(audio_file, output_dir, segment_duration=300):
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
    audio_path = Path(audio_file)
    output_dir = Path(audio_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_pattern = str(
        output_dir / f"{audio_path.stem}_{timestamp}_%03d.wav")

    cmd = [
        'ffmpeg',
        '-i', str(audio_path),
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-c', 'copy',
        '-vn',  # Exclude video
        output_pattern
    ]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
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

def gpt(message):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "summarize with bullet points:" + message,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content


def select_audio_or_video():
    """
    Select audio or video file. This function is used to select an audio or video 
    file. The file is selected by the user and returned to the program.

    @return path to audio or video file.
    """
    root = tk.Tk()
    root.withdraw()  # Hide main window   
    root.call('wm', 'attributes', '.', '-topmost', '1')

    # File dialog to select an audio or video file
    av_path = filedialog.askopenfilename(
        title='Select A/V files',
        filetypes=[('A/V files', '*.mp3 *.wav *.mp4')]
    )
    root.destroy()
    if "mp4" in av_path or "mov" in av_path:
        av_path = convert_video_to_audio(
            av_path, av_path.replace(".mp4", ".wav"))
    if av_path:
        print(f'Audio/Video file selected: {av_path}')
        folder = Path(av_path).parent / Path(av_path).stem
        folder.mkdir(parents=True, exist_ok=True)
        av_new = str(folder / Path(av_path).name)
        shutil.copy(av_path, av_new)
        return av_new
    return None

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return Path(folder_path)

def convert_video_to_audio(video_path, audio_path):
    cmd = f'ffmpeg -y -i "{video_path}" -acodec pcm_s16le -ar 16000 "{audio_path}"'
    subprocess.run(cmd, shell=True, check=True)
    return audio_path


def transcribe_to_json(audio_file_path, model_size="large-v3", device="cuda", compute_type="float16", beam_size=5, language=None, condition_on_previous_text=True, word_timestamps=False, vad_filter=False, vad_parameters=None):
    """
    Transcribe an audio file and save the transcription results to a JSON file.
    """
    # Initialize WhisperModel
    model = WhisperModel(model_size_or_path=model_size,
                         device=device, compute_type=compute_type)

    transcribe_args = {"beam_size": beam_size}
    if language:
        transcribe_args["language"] = language
    if condition_on_previous_text is not None:
        transcribe_args["condition_on_previous_text"] = condition_on_previous_text
    if word_timestamps:
        transcribe_args["word_timestamps"] = word_timestamps
    if vad_filter:
        transcribe_args["vad_filter"] = vad_filter
        if vad_parameters:
            transcribe_args["vad_parameters"] = vad_parameters

    # Transcribe the audio file
    segments, info = model.transcribe(audio_file_path, **transcribe_args)


    transcription_data = {
        "audio_file": audio_file_path,
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "transcription": [
            {"start": segment.start,
            "end": segment.end,
            "words": [{"start": word.start, "end": word.end, "word": word.word} for word in segment.words]} if word_timestamps
            else {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ]
    }
    # Write transcription data to JSON file
    json_file_path = audio_file_path.replace(".mp3", "_transcription.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(transcription_data, json_file, indent=4)

    print(
        f"Transcription for '{audio_file_path}' has been written to '{json_file_path}'")


class AudioTranscriber:
    def __init__(self, model_size='medium.en', device='cuda'):
        """
        Initialize the transcriber with a specific model size and device.
        """
        self.model = stable_whisper.load_faster_whisper(model_size_or_path=model_size,
                                                        device=device)
        self.audio_paths = []
        self.index = len(self.audio_paths) - 1
        
    def transcribe_audio(self, audio_path, language='en', beam_size=5):
        """
        Transcribe the given audio file and return the transcription result.
        """
        self.audio_paths.append(audio_path)
        self.json_paths = []
        return self.model.transcribe_stable(
            audio_path,
            word_timestamps=True,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={'min_silence_duration_ms': 200}
        )

    def save_transcription(self, audio_path, result):
        """
        Save the transcription to .srt and .json files based on the audio file path.
        """
        # Determine file paths for the outputs
        if "wav" in audio_path:
            srt_path = audio_path.replace(".wav", ".srt")
            json_path = audio_path.replace(".wav", ".json")
            ass_path = audio_path.replace(".wav", ".ass")
        else:
            srt_path = audio_path.replace(".mp3", ".srt")
            json_path = audio_path.replace(".mp3", ".json")
            ass_path = audio_path.replace(".mp3", ".ass")

        # Write transcription to .srt file
        result.to_srt_vtt(srt_path)
        result.to_ass(ass_path)
        # Prepare transcription data for JSON export
        result.save_as_json(json_path)
        self.json_path = json_path
        
    def censor_cursing(self, audio_path):
        process_audio(audio_path, self.json_path)
        
    def process_audio(self, audio_path):
        """
        Process an audio file, transcribe it and save the results.
        """
        result = self.transcribe_audio(audio_path)
        result.split_by_length(max_chars=20)
        self.save_transcription(audio_path, result)
        self.censor_cursing(audio_path)
        
        
"""
model = stable_whisper.load_faster_whisper(model_size_or_path='base',
                                           device='cuda'
                                           )
audio_path = select_audio_or_video()
result = model.transcribe_stable(
    audio_path,
    word_timestamps=True,
    language='en',
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=200)
)

if "wav" in audio_path:
    srt_path = audio_path.replace(".wav", ".srt")
    json_path = audio_path.replace(".wav", ".json")
else:
    srt_path = audio_path.replace(".mp3", ".srt")
    json_path = audio_path.replace(".mp3", ".json")
# Write transcription to .srt file directly utilizing to_srt_vtt() method.
result.to_srt_vtt(srt_path)  # Assuming 'to_srt_vYour `file_to_fix.py` script aims to utilize functionalities from `stable_whisper`, a hypothetical API for transcript processing, to perform enhanced audio transcription. However, based on the initial documentation and the error message provided in your first query, I noted that your script includes several parameters and method names that need to be adjusted to align with the documentation details you've shared. Below is a corrected version of the `file_to_fix.py`.

# Prepare for JSON export, extracting essential data for a compact, informative JSON structure.
transcript_json = {
    'audio_file': 'audio.mp3',
    'transcription': result.segments
    # Additional metadata
}

# Write JSON output
with open(json_path, 'w') as json_file:
    json.dump(transcript_json, json_file, ensure_ascii=False, indent=4)


# Load the Whisper model with specific enhancements.
model = stable_whisper.load_faster_whisper(
    model_size_or_path='base',
    # Additional options like device and dynamic quantization (dq) were not specified in the documentation snippet provided.
)
"""

if __name__ == '__main__':
    global transcript_paths
    transcript_paths = []
    transcriber = AudioTranscriber(model_size='large-v3', device='cuda')
    audio_path = select_audio_or_video()
    enums = split_audio(audio_path, 'output')
    if enums:
        for audio_path in enums: 
            print("wav_file_path type:", type(audio_path))
            print("wav_file_path content:", audio_path)
            print(f'Processing {audio_path}...')
            transcriber.process_audio(audio_path)
    else: 
        print(f'Processing {audio_path}...')
        transcriber.process_audio(audio_path)
        
from tkinter.colorchooser import askcolor
import stable_whisper
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import subprocess
import shutil
import os
from datetime import datetime
import moviepy.editor as mp
from process_files import *
from censorship import *
import re
from datetime import datetime, timedelta

def clean_path(path_str):
    path = Path(path_str)
    clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', path.stem)
    clean_name = re.sub(r'_+', '_', clean_name)
    return path.with_stem(clean_name)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def split_audio(audio_file, output_dir, segment_duration=60):
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
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(audio_file), str(audio_path))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_pattern = str(
        output_dir / f"{audio_path.stem}_{timestamp}_%03d.wav")

    cmd = [
        'ffmpeg',
        '-i', str(audio_path),
        '-f', 'segment',
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
    root = tk.Tk()
    root.withdraw()  # Hide main window   
    root.call('wm', 'attributes', '.', '-topmost', '1')

    # File dialog to select an audio or video file
    av_path = filedialog.askopenfilename(
        title='Select A/V files',
        filetypes=[('A/V files', '*.mp3 *.wav *.mp4')]
    )
    root.destroy()
    if "mp4" in av_path or "mov" in av_path:
        av_path = convert_video_to_audio(
            av_path, av_path.replace(".mp4", ".wav"))
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
        shutil.copy(av_path, av_new)
        return av_new
    return None

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return Path(folder_path)

def convert_video_to_audio(video_path, audio_path):
    cmd = f'ffmpeg -y -i "{video_path}" -acodec pcm_s16le -ar 16000 "{audio_path}"'
    subprocess.run(cmd, shell=True, check=True)
    return audio_path


class AudioTranscriber:
    def __init__(self, model_size='large-v3', device='cuda'):
        """
        Initialize the transcriber with a specific model size and device.
        """
        self.model = stable_whisper.load_model(model_size,
                                                        device=device)
        self.audio_paths = []
        self.index = len(self.audio_paths) - 1
        self.clean_paths = []
        self.srt_paths = []

    def add_time(self, time_str, minutes=1):
        """Add minutes to SRT timestamp, adjusting for fractional seconds."""
        base_time = datetime.strptime(time_str.split(',')[0], '%H:%M:%S')
        milliseconds = int(time_str.split(',')[1]) if ',' in time_str else 0
        added_time = base_time + \
            timedelta(minutes=minutes, milliseconds=milliseconds)
        return added_time.strftime('%H:%M:%S,') + f"{milliseconds:03d}"
        
    def srt_combine(self):
        combined_content = ''
        subtitle_number = 1
        additional_minutes = 0
        for index, file_path in enumerate(self.srt_paths):
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
        name = Path(self.srt_paths[0]).stem
        output_file_prt = Path(self.srt_paths[0]).parent.parent / f'{name}.srt'
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
            #beam_size=beam_size
            # vad_filter=False,
        )

    def save_transcription(self, audio_path, result):
        """
        Save the transcription to .srt and .json files based on the audio file path.
        """
        # Determine file paths for the outputs
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
        result.to_srt_vtt(srt_path)
        # Prepare transcription data for JSON export
        result.save_as_json(json_path)
        self.json_path = json_path
        print('completed transcript files')
        self.srt_paths.append(srt_path)
        
    def censor_cursing(self, audio_path):
        return process_audio(audio_path, self.json_path)
        
    def transcribe_and_censor(self, audio_path):
        """
        Process an audio file, transcribe it and save the results.
        """
        result = self.transcribe_audio(audio_path)
        result.split_by_length(max_chars=20)
        self.save_transcription(audio_path, result)
        self.clean_paths.append(self.censor_cursing(audio_path))
        
        
if __name__ == '__main__':
    global transcript_paths
    transcript_paths = []
    print('loading model')
    transcriber = AudioTranscriber(model_size='large-v3', device='cuda')
    print('finished')
    audio_path = select_audio_or_video()
    log_ = JSONLog(audio_path)
    enums = split_audio(audio_path, 'output')
    if enums:
        for counter, audio_path in enumerate(enums): 
            print("wav_file_path type:", type(audio_path))
            print("wav_file_path content:", audio_path)
            print(
                f'\n\nProcessing {audio_path}...\n@@@@@@@@@@@@@@@@@@@\nindex {counter} of {len(enums)}\n@@@@@@@@@@@@@@@@@@@\n')
            transcriber.transcribe_and_censor(audio_path)
    else: 
        print(f'Processing {audio_path}...')
        transcriber.transcribe_and_censor(audio_path)
    combine_wav_files(transcriber.clean_paths)
    transcriber.srt_combine()
