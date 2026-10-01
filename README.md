# Road Damage Detection

This project is structured to mirror the project layout used in the reference violence-detection repository while adapting it to road damage inspection with exported YOLO model weights.

## Project structure

```text
road dammage model/
├── exported_models/                # Trained YOLO weights and evaluation summary
│   ├── results_summary.json
│   ├── yolov11n_best.pt
│   ├── yolov5su_best.pt
│   ├── yolov8m_best.pt
│   ├── yolov8n_best.pt
│   └── yolov8s_best.pt
├── notebooks/
│   ├── drafts/                    # Experimental notebooks
│   └── final/
│       └── road-damage-detection-final.ipynb
├── streamlit_app/
│   ├── app.py                     # Inference UI for uploaded images
│   ├── requirements.txt
│   └── models/
│       └── config.json
├── results/
│   └── results_summary.json
├── .gitignore
├── requirements.txt
└── README.md
```

## Overview

The project evaluates multiple YOLO variants for road damage detection, including:

- YOLOv8n
- YOLOv8s
- YOLOv8m
- YOLOv5s-u
- YOLOv11n

The exported model files under `exported_models/` are ready for deployment or further experimentation.

## Quick start

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app loads the trained weights from `exported_models/`.
- The results summary is stored in both `exported_models/results_summary.json` and `results/results_summary.json` for easy reporting and dashboard work.
- This project structure keeps the workflow close to the repository pattern: notebooks, app, and results are separated cleanly.
