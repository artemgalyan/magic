python -m pip install -e . --force-reinstall
mkdir data/interim
build-features data/raw data/interim/data.csv mean median skew
echo Done
exit /b
