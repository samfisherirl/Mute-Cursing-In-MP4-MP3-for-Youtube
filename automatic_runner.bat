@echo off
set outputFile=main\output.txt

if exist "%outputFile%" goto skipSubFile1

echo Running dependency installer...
call "main\step1_dependency_installer.bat" >> "%outputFile%"

:skipSubFile1
echo Running installer check - then running...
call "main\step2-run.bat" >> "%outputFile%"

echo Completed.