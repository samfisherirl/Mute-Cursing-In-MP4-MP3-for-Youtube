import tkinter as tk
from tkinter import messagebox
import subprocess


def download_video(url, fmt):
    # Use yt-dlp to download the best format of the video and pipe it to FFmpeg
    ytdlp_command = ['yt-dlp', '-f', 'best', '-o', '-', url]
    ffmpeg_command = ['ffmpeg', '-i', '-', 'downloaded_file.' + fmt]

    with subprocess.Popen(ytdlp_command, stdout=subprocess.PIPE) as ytdlp_proc:
        with subprocess.Popen(ffmpeg_command, stdin=ytdlp_proc.stdout) as ffmpeg_proc:
            ffmpeg_proc.communicate()


def download_window():
    def start_download_mp3():
        url = url_input.get()
        download_video(url, 'mp3')
        messagebox.showinfo("Download Complete",
                            "Video downloaded successfully as MP3!")

    def start_download_mp4():
        url = url_input.get()
        download_video(url, 'mp4')
        messagebox.showinfo("Download Complete",
                            "Video downloaded successfully as MP4!")

    root = tk.Tk()
    root.title("YouTube Downloader")

    tk.Label(root, text="Enter YouTube video URL:").grid(row=0, column=0)
    url_input = tk.Entry(root)
    url_input.grid(row=0, column=1)

    tk.Button(root, text="Download as MP3",
              command=start_download_mp3).grid(row=1, column=0)
    tk.Button(root, text="Download as MP4",
              command=start_download_mp4).grid(row=1, column=1)

    root.mainloop()


if __name__ == "__main__":
    download_window()
