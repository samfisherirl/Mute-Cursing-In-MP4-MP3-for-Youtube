# Mute curse words in videos



https://github.com/samfisherirl/Mute-Cursing-In-MP4-MP3-for-Youtube/assets/98753696/74cde7ee-774c-4517-8752-45a081d10e8f

- download: https://github.com/samfisherirl/Mute-Cursing-In-MP4-MP3-for-Youtube/archive/refs/heads/main.zip
- download and install (if you dont have python 310): https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
- During install, make sure to click "add to PATH": https://miro.medium.com/v2/resize:fit:1344/0*7nOyowsPsGI19pZT.png
- Unzip main.zip in downloads, click and run `one-click-installer`
- First run will require a download that takes a short bit.  
- Or venv + pip install -r requirements.txt
- FileSelect transcript and or mp3 file
- output to same relative path

## Requires 

- Python 3.10
- https://pytorch.org/get-started/locally/
- ffmpeg https://github.com/icedterminal/ffmpeg-installer/releases/download/6.1.1.20240201/FFmpeg_Full.msi
- run one-click-installer.bat

## Concept 

1) convert mp4/mp3 to wav
2) read mp3 transcript with openai wisper via stable-ts
3) read csv of curse words
4) if curse word found matching a word in a sentence, mute that word
5) convert back to mp3

## To add

1) Convert from mp4
   

### Concerns:

- clipping // not fading in/out of clips
- setting words to censor
- conversion time/optimization
