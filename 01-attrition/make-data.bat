python -m pip install -e . --force-reinstall
build-features data/raw data/interim/data.csv mean median skew
make-dataset data/interim/data.csv data/processed/data.csv
echo Done
exit /b
