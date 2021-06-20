@echo off
echo 1
python DetectaDiferencias01.py --first C:\Users\D\source\repos\DiegoUrdi\KinectImage\KinectImage\bin\Release\a.bmp --second C:\Users\D\source\repos\DiegoUrdi\KinectImage\KinectImage\bin\Release\b.bmp
:while
(
TIMEOUT /T 1
echo 2
python DetectaDiferencias01.py --first C:\Users\D\source\repos\DiegoUrdi\KinectImage\KinectImage\bin\Release\a.bmp --second C:\Users\D\source\repos\DiegoUrdi\KinectImage\KinectImage\bin\Release\b.bmp
echo 3
python DetectaDiferencias01.py --first C:\Users\D\source\repos\DiegoUrdi\KinectImage\KinectImage\bin\Release\a.bmp --second C:\Users\D\source\repos\DiegoUrdi\KinectImage\KinectImage\bin\Release\b.bmp
goto :while
)