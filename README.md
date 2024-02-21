# Mute curse words in videos

- Call one-click-installer
- Or venv + pip install -r requirements.txt
- FileSelect transcript and or mp3 file
- output to same relative path

## Requires 

- Python 3.10
- https://pytorch.org/get-started/locally/
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
