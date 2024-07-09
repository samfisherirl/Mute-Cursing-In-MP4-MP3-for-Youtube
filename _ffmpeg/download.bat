@echo off
setlocal
set "URL=https://github.com/samfisherirl/Mute-Cursing-In-MP4-MP3-for-Youtube/releases/download/v1/_ffmpeg.7z"
set "FILE_NAME=ffmpeg.7z"

certutil -urlcache -f %URL% "%~dp0%FILE_NAME%"
endlocal