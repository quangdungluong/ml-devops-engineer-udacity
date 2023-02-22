#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate section4
python ingestion.py
python training.py
python scoring.py
python deployment.py
python reporting.py
python app.py