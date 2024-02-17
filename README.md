# mute_curse_words_for_youtube

- Call one-click-installer
- Or venv + pip install -r requirements.txt
- FileSelect transcript and or mp3 file
- output to same relative path

## Requires 

- Python 3.10
- run one-click-installer.bat

## Concept

1) intake mp4 video
2) convert to mp3
3) read mp3 transcript with wisper
4) read csv of curse words
5) if word in sentence then silence word
6) convert back to mp3
7) add mp3 to mp4 and encode
   

### Concerns:

- clipping // not fading in/out of clips
- setting words to censor
- conversion time/optimization
