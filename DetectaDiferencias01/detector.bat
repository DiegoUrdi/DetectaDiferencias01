@echo off
echo 1
start python DetectaDiferencias01.py --first a.bmp --second b.bmp
:while
(
TIMEOUT /T 1
echo 2
python DetectaDiferencias01.py --first a.bmp --second b.bmp
echo 3
python DetectaDiferencias01.py --first a.bmp --second b.bmp
goto :while
)