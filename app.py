
import io
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


# =========================================================
# 1. CPU OPTIMIZATION
# =========================================================

try:
    torch.set_num_threads(8)
except RuntimeError:
    pass

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# =========================================================
# 2. PROJECT CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "xray_detector_exp_recall"
    / "weights"
    / "best.pt"
)

CLASS_NAMES = [
    "Gun",
    "Knife",
    "Pliers",
    "Scissors",
    "Wrench"
]

IMAGE_SIZE = 416
DEVICE = "cpu"


# =========================================================
# 3. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="XRAY AI | Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 4. BLACK AND VIOLET THEME
# =========================================================

st.markdown("""
<style>

:root {
    --bg: #090610;
    --panel: #130D20;
    --panel2: #1B1230;
    --border: #302044;
    --violet: #9B6BFF;
    --bright: #B98AFF;
    --text: #F5F0FF;
    --muted: #B8AFC9;
    --green: #38D6A5;
    --red: #FF5577;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(9, 6, 16, 0.92);
}

[data-testid="stSidebar"] {
    background: #100A1A;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}

.hero {
    background: linear-gradient(
        115deg,
        #1A102B 0%,
        #100A19 65%,
        #21113A 100%
    );
    border: 1px solid #3B2857;
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 40px rgba(0,0,0,0.22);
}

.hero-kicker {
    color: var(--bright);
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.17em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.hero-title {
    color: var(--text);
    font-size: clamp(1.65rem, 3vw, 2.55rem);
    line-height: 1.15;
    font-weight: 850;
    margin: 0;
}

.hero-subtitle {
    color: var(--muted);
    margin-top: 0.65rem;
    font-size: 1rem;
}

.section-title {
    color: var(--text);
    font-size: 1.08rem;
    font-weight: 750;
    margin: 0.35rem 0 0.7rem;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 14px 16px;
}

div[data-testid="stMetricLabel"] {
    color: var(--muted);
}

div[data-testid="stMetricValue"] {
    color: var(--bright);
}

div[data-testid="stFileUploader"] {
    background: var(--panel);
    border: 1px dashed #7650A7;
    border-radius: 16px;
    padding: 12px;
}

div.stButton > button,
div.stDownloadButton > button {
    background: linear-gradient(100deg, #7541C8, #9B6BFF);
    color: white;
    border: 1px solid #A982F2;
    border-radius: 11px;
    font-weight: 750;
    min-height: 2.8rem;
    transition: all 0.18s ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: linear-gradient(100deg, #8A55DF, #B18AFF);
    border-color: #D1B8FF;
    color: white;
    transform: translateY(-1px);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.status-pill {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    color: #9CF3D5;
    background: rgba(56,214,165,0.10);
    border: 1px solid rgba(56,214,165,0.30);
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

.footer {
    color: #9388A8;
    text-align: center;
    font-size: 0.78rem;
    padding-top: 0.6rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 5. LOAD TRAINED YOLO MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading YOLOv8-Nano model...")
def load_model(model_path):

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}"
        )

    return YOLO(str(path))


# =========================================================
# 6. HEADER
# =========================================================

# st.html renders the header as HTML rather than showing tags.
st.html("""
<div class="hero">

    <div class="hero-kicker">
        AI-powered security inspection
    </div>

    <h1 class="hero-title">
        Automated Luggage X-Ray Threat Detection
    </h1>

    <div class="hero-subtitle">
        Detect and review five object classes in X-ray baggage images.
    </div>

    <div style="margin-top:1rem">
        <span class="status-pill">
            ● MODEL CHECKPOINT CONFIGURED
        </span>
    </div>

</div>
""")


# =========================================================
# 7. SIDEBAR CONTROLS
# =========================================================

st.sidebar.markdown("## 🛠️ Detection Controls")

st.sidebar.caption(
    "Adjust inference thresholds before running detection."
)

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.01,
    max_value=0.90,
    value=0.25,
    step=0.01,
    help="Predictions below this confidence are filtered out."
)

iou_threshold = st.sidebar.slider(
    "NMS IoU threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
    help="Controls suppression of overlapping bounding boxes."
)

st.sidebar.divider()

st.sidebar.markdown("## Model Information")

st.sidebar.write("**Architecture:** YOLOv8-Nano")
st.sidebar.write(f"**Device:** {DEVICE.upper()}")
st.sidebar.write(f"**Inference image size:** {IMAGE_SIZE}")
st.sidebar.write(f"**Confidence:** {confidence:.2f}")
st.sidebar.write(f"**NMS IoU:** {iou_threshold:.2f}")

st.sidebar.caption(
    "Classes: Gun · Knife · Pliers · Scissors · Wrench"
)

with st.sidebar.expander("Checkpoint details"):

    st.code(str(MODEL_PATH), language="text")

    st.write(
        "Checkpoint exists:",
        MODEL_PATH.exists()
    )

    if MODEL_PATH.exists():
        st.write(
            f"Checkpoint size: {MODEL_PATH.stat().st_size:,} bytes"
        )


# =========================================================
# 8. TOP INFORMATION CARDS
# =========================================================

m1, m2, m3 = st.columns(3)

m1.metric("Detection classes", "05")
m2.metric("Inference size", f"{IMAGE_SIZE} × {IMAGE_SIZE}")
m3.metric("Execution device", "CPU")

st.divider()


# =========================================================
# 9. IMAGE UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">01 / Upload X-ray image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png", "bmp"],
    key="xray_uploader"
)


# =========================================================
# 10. RUN DETECTION
# =========================================================

if uploaded_file is None:

    st.info("Upload an X-ray image to begin inspection.")

else:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.caption(
            f"Uploaded image · {image.width} × {image.height} pixels"
        )

        left, right = st.columns(2, gap="large")

        with left:

            st.markdown(
                '<div class="section-title">Original X-ray</div>',
                unsafe_allow_html=True
            )

            st.image(
                image,
                use_container_width=True
            )

        with right:

            st.markdown(
                '<div class="section-title">Detection output</div>',
                unsafe_allow_html=True
            )

            run_detection = st.button(
                "⚡ Run Threat Detection",
                type="primary",
                use_container_width=True
            )

            if run_detection:

                try:

                    model = load_model(str(MODEL_PATH))

                    start_time = time.perf_counter()

                    with st.spinner(
                        "Analyzing X-ray image on CPU..."
                    ):

                        # Direct PIL input avoids temporary disk operations.
                        result = model.predict(
                            source=image,
                            imgsz=IMAGE_SIZE,
                            conf=confidence,
                            iou=iou_threshold,
                            device=DEVICE,
                            max_det=100,
                            agnostic_nms=False,
                            save=False,
                            verbose=False
                        )[0]

                    inference_seconds = (
                        time.perf_counter() - start_time
                    )

                    # Generate annotated image.
                    annotated_rgb = (
                        result.plot()[..., ::-1].copy()
                    )

                    st.image(
                        annotated_rgb,
                        caption="Annotated X-ray · model predictions",
                        use_container_width=True
                    )

                    # Extract detections.
                    detections = []

                    if (
                        result.boxes is not None
                        and len(result.boxes) > 0
                    ):

                        xyxy = result.boxes.xyxy.cpu().numpy()

                        classes = (
                            result.boxes.cls.cpu()
                            .numpy()
                            .astype(int)
                        )

                        scores = result.boxes.conf.cpu().numpy()

                        for coords, class_id, score in zip(
                            xyxy,
                            classes,
                            scores
                        ):

                            detections.append({

                                "Object": result.names[int(class_id)],

                                "Confidence (%)": round(
                                    float(score) * 100,
                                    2
                                ),

                                "X1": round(float(coords[0]), 2),
                                "Y1": round(float(coords[1]), 2),
                                "X2": round(float(coords[2]), 2),
                                "Y2": round(float(coords[3]), 2)

                            })

                    # Save results in session state.
                    st.session_state["xray_last_result"] = {

                        "detections": detections,

                        "annotated_rgb": annotated_rgb,

                        "inference_seconds": inference_seconds,

                        "image_size": image.size,

                        "confidence": confidence,

                        "iou": iou_threshold

                    }

                except Exception as error:

                    st.error(
                        f"Detection error: {error}"
                    )

    except Exception as error:

        st.error(
            f"Image processing error: {error}"
        )


# =========================================================
# 11. DETECTION SUMMARY
# =========================================================

saved = st.session_state.get("xray_last_result")

if saved is not None:

    st.divider()

    st.markdown(
        '<div class="section-title">02 / Detection summary</div>',
        unsafe_allow_html=True
    )

    detections = saved["detections"]

    total = len(detections)

    a, b, c = st.columns(3)

    a.metric(
        "Objects detected",
        total
    )

    b.metric(
        "Inference time",
        f"{saved['inference_seconds']:.3f} s"
    )

    c.metric(
        "Throughput",
        (
            f"{1 / saved['inference_seconds']:.2f} img/s"
            if saved["inference_seconds"] > 0
            else "—"
        )
    )

    st.caption(
        "Inference time measures the prediction call and varies "
        "with image size, CPU load and warm-up state."
    )


    # =====================================================
    # 12. DETECTED OBJECTS TABLE
    # =====================================================

    if detections:

        dataframe = pd.DataFrame(detections)

        st.markdown(
            '<div class="section-title">Detected objects</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            dataframe[["Object", "Confidence (%)"]],
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # 13. CLASS-WISE COUNTS
        # =================================================

        st.markdown(
            '<div class="section-title">Class-wise counts</div>',
            unsafe_allow_html=True
        )

        counts = Counter(
            item["Object"]
            for item in detections
        )

        chart_data = pd.DataFrame({

            "Class": CLASS_NAMES,

            "Detections": [
                counts.get(name, 0)
                for name in CLASS_NAMES
            ]

        }).set_index("Class")

        st.bar_chart(
            chart_data,
            color="#9B6BFF"
        )


        # =================================================
        # 14. DOWNLOAD CSV REPORT
        # =================================================

        csv_data = dataframe.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download detection report (CSV)",
            data=csv_data,
            file_name="xray_detection_report.csv",
            mime="text/csv"
        )


        # =================================================
        # 15. DOWNLOAD ANNOTATED IMAGE
        # =================================================

        image_bytes = io.BytesIO()

        Image.fromarray(
            saved["annotated_rgb"]
        ).save(
            image_bytes,
            format="PNG"
        )

        st.download_button(
            "⬇️ Download annotated X-ray (PNG)",
            data=image_bytes.getvalue(),
            file_name="xray_detection_result.png",
            mime="image/png"
        )

        st.success(
            f"Detection completed: {total} object(s) detected."
        )

    else:

        st.warning(
            "No objects were detected at this confidence threshold. "
            "Try a lower threshold, noting that this may increase "
            "false detections."
        )


    # =====================================================
    # 16. INFERENCE DIAGNOSTICS
    # =====================================================

    with st.expander("Inference diagnostics"):

        st.write(
            "**Checkpoint:**",
            str(MODEL_PATH)
        )

        st.write(
            "**Checkpoint exists:**",
            MODEL_PATH.exists()
        )

        st.write(
            "**Input dimensions:**",
            saved["image_size"]
        )

        st.write(
            "**Model image size:**",
            IMAGE_SIZE
        )

        st.write(
            "**Device:**",
            DEVICE
        )

        st.write(
            "**PyTorch CPU threads:**",
            torch.get_num_threads()
        )

        st.write(
            "**PyTorch inter-op threads:**",
            torch.get_num_interop_threads()
        )

        st.write(
            "**Confidence:**",
            saved["confidence"]
        )

        st.write(
            "**NMS IoU:**",
            saved["iou"]
        )

        st.write(
            "**Inference time:**",
            f"{saved['inference_seconds']:.4f} seconds"
        )


# =========================================================
# 17. FOOTER
# =========================================================

st.divider()

st.html("""
<div class="footer">
    Research prototype · YOLOv8-Nano · CPU inference<br>
    Not intended to replace certified security screening systems.
</div>
""")
