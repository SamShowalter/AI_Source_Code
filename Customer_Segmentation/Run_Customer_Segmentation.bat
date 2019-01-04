@echo off

:: Set this python variable path to your corresponding Python3 environment
set "python=C:\\Users\\sshowalter\\AppData\\Local\\Continuum\\miniconda3\\python.exe"
"%python%" Application\app.py %*
pause