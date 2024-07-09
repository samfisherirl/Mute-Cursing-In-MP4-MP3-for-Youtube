@echo off
setlocal
set FFMPEG_PATH=%USERPROFILE%\Documents\ffmpeg_\bin
Powershell -Command "[Environment]::SetEnvironmentVariable('Path', $Env:Path + ';%FFMPEG_PATH%', [System.EnvironmentVariableTarget]::Machine)"
cmd /c
endlocal