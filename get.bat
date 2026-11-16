@echo off
rem Скачивает kernel.cu из репы в папку, где лежит этот файл.
rem   get            - последняя версия
rem   get d8d281a    - версия по номеру коммита
set VER=%1
if "%VER%"=="" set VER=main
set /p PASS=Proxy password: 
curl --proxy-digest --ssl-no-revoke -x http://192.168.251.1:8080 -U 35830184:%PASS% -o "%~dp0kernel.cu" "https://raw.githubusercontent.com/Ondrecho/cuda/%VER%/kernel.cu?%RANDOM%"
echo.
type "%~dp0kernel.cu"
