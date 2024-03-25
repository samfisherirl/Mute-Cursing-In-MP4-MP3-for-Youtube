import tkinter as tk
from tkinter import filedialog
from moviepy.editor import VideoFileClip

def get_file_info(file_path):
    try:
        clip = VideoFileClip(file_path)
        video_codec = clip.video.reader.codec
        audio_codec = clip.audio.reader.codec
        video_bitrate = clip.video.reader.bitrate
        audio_bitrate = clip.audio.reader.bitrate
        print(f"Video Codec: {video_codec}")
        print(f"Audio Codec: {audio_codec}")
        print(f"Video Bitrate: {video_bitrate}")
        print(f"Audio Bitrate: {audio_bitrate}")
    except AttributeError as e:
        print("This file may not have an audio or video stream.")
    except Exception as e:
        print(f"An error occurred: {e}")

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()
    if file_path.endswith('.mp4'):
        get_file_info(file_path)
    else:
        print("Please select an MP4 file.")

if __name__ == "__main__":
    select_file()
