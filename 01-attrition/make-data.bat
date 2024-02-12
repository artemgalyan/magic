python -m pip install -e . --force-reinstall
build-features data/raw data/interim/data.csv mean median skew
echo Done
exit /b
