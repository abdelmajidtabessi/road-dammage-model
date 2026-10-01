from pathlib import Path
import json

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "exported_models"
RESULTS_PATH = ROOT / "results" / "results_summary.json"
if not RESULTS_PATH.exists():
    RESULTS_PATH = EXPORT_DIR / "results_summary.json"
MODEL_FILES = sorted(EXPORT_DIR.glob("*_best.pt"))

CLASS_LABELS = {0: "Pothole", 1: "Crack", 2: "Manhole"}
CLASS_COLORS = {0: "#ff4d4d", 1: "#f7b731", 2: "#4dabf7"}


def load_model_summary():
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    try:
        with RESULTS_PATH.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not summary:
            return pd.DataFrame()
        df = pd.DataFrame(summary)
        return df.sort_values("mAP50", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_model_choices():
    if not MODEL_FILES:
        return []
    return [model.name for model in MODEL_FILES]


st.set_page_config(page_title="Road Damage Detection", page_icon="🚧", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #09111f 0%, #0d1b2a 40%, #132a3a 100%);
            color: #eaf3ff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: rgba(12, 18, 31, 0.9);
        }
        .stMetric > div {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 0.8rem;
            padding: 0.8rem 0.9rem;
        }
        .kpi-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 0.85rem;
            padding: 1rem;
            margin-bottom: 0.8rem;
        }
        .legend-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 999px;
            padding: 0.4rem 0.75rem;
            margin: 0.25rem 0.5rem 0.25rem 0;
        }
        .legend-dot {
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 50%;
            display: inline-block;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚧 Road Damage Detection")
st.caption("YOLO-powered inspection dashboard for identifying potholes, cracks, and manholes in road imagery.")

model_choices = get_model_choices()
if not model_choices:
    st.error("No exported YOLO model files were found in the exported_models folder.")
    st.stop()

summary_df = load_model_summary()
if not summary_df.empty:
    best_model_row = summary_df.sort_values("mAP50", ascending=False).iloc[0]
    best_model = best_model_row["model"]
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div class='kpi-card'><strong>Best model</strong><br><span style='font-size:1.8rem'>{best_model.upper()}</span><br>mAP50: {best_model_row['mAP50']:.3f}</div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div class='kpi-card'><strong>Precision</strong><br><span style='font-size:1.8rem'>{best_model_row['precision']:.3f}</span><br>model quality</div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<div class='kpi-card'><strong>Recall</strong><br><span style='font-size:1.8rem'>{best_model_row['recall']:.3f}</span><br>coverage</div>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("⚙️ Detection settings")
    selected_model = st.selectbox("Model", model_choices, index=0)
    confidence = st.slider("Detection confidence", min_value=0.1, max_value=0.95, value=0.25, step=0.05)
    image_size = st.slider("Image size", min_value=320, max_value=1280, value=640, step=64)

    st.markdown("### Damage classes")
    for class_id, label in CLASS_LABELS.items():
        st.markdown(
            f"<div class='legend-badge'><span class='legend-dot' style='background:{CLASS_COLORS[class_id]}'></span>{label}</div>",
            unsafe_allow_html=True,
        )

if not summary_df.empty:
    st.subheader("📊 Model leaderboard")
    leaderboard = summary_df[["model", "mAP50", "mAP5095", "precision", "recall"]].copy()
    leaderboard["mAP50"] = leaderboard["mAP50"].map(lambda x: f"{x:.3f}")
    leaderboard["mAP5095"] = leaderboard["mAP5095"].map(lambda x: f"{x:.3f}")
    leaderboard["precision"] = leaderboard["precision"].map(lambda x: f"{x:.3f}")
    leaderboard["recall"] = leaderboard["recall"].map(lambda x: f"{x:.3f}")
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

model_path = EXPORT_DIR / selected_model
model = YOLO(str(model_path))

st.subheader("🖼️ Image inspection")
uploaded_file = st.file_uploader("Choose a road image for inference", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    uploaded_image_path = ROOT / "tmp_uploaded_image.png"
    image.save(uploaded_image_path)

    left, right = st.columns(2)
    left.image(image, caption="Original image", use_container_width=True)

    with st.spinner("Running inference..."):
        results = model(str(uploaded_image_path), conf=confidence, imgsz=image_size)

    annotated_image = results[0].plot()
    right.image(annotated_image, caption=f"Detection result using {selected_model}", use_container_width=True)

    boxes = results[0].boxes
    if boxes is not None and len(boxes):
        conf_values = [float(value) for value in boxes.conf.tolist()]
        class_ids = [int(item) for item in boxes.cls.tolist()]
        class_names = [CLASS_LABELS.get(class_id, str(class_id)) for class_id in class_ids]

        detection_df = pd.DataFrame(
            {
                "class": class_names,
                "class_id": class_ids,
                "confidence": conf_values,
                "x1": [float(x) for x in boxes.xyxy[:, 0].tolist()],
                "y1": [float(y) for y in boxes.xyxy[:, 1].tolist()],
                "x2": [float(x) for x in boxes.xyxy[:, 2].tolist()],
                "y2": [float(y) for y in boxes.xyxy[:, 3].tolist()],
            }
        )

        total_detections = len(detection_df)
        mean_confidence = detection_df["confidence"].mean() if total_detections else 0.0
        class_counts = detection_df["class"].value_counts().to_dict()

        st.success(f"Detected {total_detections} objects above the confidence threshold.")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Detections", total_detections)
        stat_col2.metric("Avg confidence", f"{mean_confidence:.3f}")
        stat_col3.metric("Most frequent", max(class_counts, key=class_counts.get) if class_counts else "None")

        st.subheader("📋 Detection table")
        st.dataframe(detection_df, use_container_width=True, hide_index=True)

        st.subheader("Severity breakdown")
        breakdown_cols = st.columns(min(3, len(class_counts)))
        for idx, (label, count) in enumerate(class_counts.items()):
            breakdown_cols[idx % len(breakdown_cols)].metric(label, count)
    else:
        st.info("No road damage objects were detected above the configured confidence threshold.")
else:
    st.info("Upload an image to start the road damage detection pipeline.")
