import json
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import verify_signature


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "signature_siamese.keras"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SignatureGuard — Document Examination",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOAD METRICS
# ============================================================

def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH, "r") as file:
            return json.load(file)
    except Exception:
        return {}


metrics = load_metrics()
THRESHOLD = metrics.get("optimal_threshold", 0.209426)


# ============================================================
# CUSTOM CSS
# ============================================================
# Design language: a document examiner's desk. Aged-paper card
# surfaces on a deep ink background, a serif "ledger" display
# face for the wordmark, a monospace face for measured readouts
# (the numbers a forensic report would print), and a rotated
# ink-stamp verdict in place of a generic status pill.

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

    <style>

    :root {
        --ink:        #12141a;
        --ink-panel:  #191c24;
        --paper:      #ece5d3;
        --paper-dim:  #b9b3a1;
        --rule:       rgba(236, 229, 211, 0.12);
        --text-hi:    #efece4;
        --text-lo:    #8b8d97;
        --verdict-genuine: #3f7a63;
        --verdict-forged:  #96363f;
        --mono: 'IBM Plex Mono', monospace;
        --serif: 'Fraunces', Georgia, serif;
        --sans: 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background: var(--ink);
        background-image:
            repeating-linear-gradient(
                0deg,
                rgba(236, 229, 211, 0.015) 0px,
                rgba(236, 229, 211, 0.015) 1px,
                transparent 1px,
                transparent 3px
            );
    }

    /* Faint corner ink blots — decorative, fixed, never intercept clicks */
    .ink-blot {
        position: fixed;
        pointer-events: none;
        z-index: 0;
        opacity: 0.05;
    }
    .ink-blot.top-right  { top: -60px; right: -60px; }
    .ink-blot.bottom-left { bottom: -80px; left: -80px; }

    html, body, [class*="css"] {
        font-family: var(--sans);
        color: var(--text-hi);
    }

    .main { padding-top: 0rem; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }


    /* =====================================================
       MASTHEAD
       ===================================================== */

    .masthead {
        max-width: 900px;
        margin: 0 auto;
        padding: 56px 24px 32px 24px;
        border-bottom: 1px solid var(--rule);
    }

    .masthead-kicker {
        font-family: var(--mono);
        font-size: 12px;
        color: var(--paper-dim);
        letter-spacing: 0.04em;
    }

    .masthead-title {
        font-family: var(--serif);
        font-weight: 600;
        font-size: 50px;
        line-height: 1.05;
        color: var(--paper);
        margin: 10px 0 14px 0;
    }

    .masthead-title em {
        font-style: italic;
        font-weight: 500;
        color: var(--text-hi);
    }

    .masthead-desc {
        font-size: 15px;
        line-height: 1.7;
        color: var(--text-lo);
        max-width: 640px;
    }

    /* Self-drawing signature flourish beneath the masthead copy */
    .signature-draw {
        margin-top: 26px;
        width: 260px;
        height: 60px;
        overflow: visible;
    }

    .signature-draw path {
        fill: none;
        stroke: var(--paper-dim);
        stroke-width: 1.6;
        stroke-linecap: round;
        stroke-dasharray: 620;
        stroke-dashoffset: 620;
        animation: draw-signature 2.2s 0.3s cubic-bezier(0.65, 0, 0.35, 1) forwards;
    }

    @keyframes draw-signature {
        to { stroke-dashoffset: 0; }
    }

    @media (prefers-reduced-motion: reduce) {
        .signature-draw path { animation: none; stroke-dashoffset: 0; }
    }


    /* =====================================================
       EXHIBIT LABELS + CARDS
       ===================================================== */

    .exhibit-heading {
        max-width: 900px;
        margin: 40px auto 18px auto;
        padding: 0 24px;
    }

    .exhibit-heading .eyebrow {
        font-family: var(--mono);
        font-size: 12px;
        color: var(--paper-dim);
    }

    .exhibit-heading h2 {
        font-family: var(--serif);
        font-weight: 500;
        font-size: 26px;
        color: var(--paper);
        margin: 4px 0 0 0;
    }

    .exhibit-card {
        background: var(--ink-panel);
        border: 1px solid var(--rule);
        border-radius: 4px;
        padding: 22px 22px 26px 22px;
        position: relative;
        transition: border-color 0.2s ease, transform 0.2s ease;
        animation: card-in 0.5s ease both;
    }

    .exhibit-card:hover {
        border-color: var(--paper-dim);
        transform: translateY(-2px);
    }

    div[data-testid="column"]:nth-of-type(2) .exhibit-card {
        animation-delay: 0.1s;
    }

    @keyframes card-in {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .exhibit-tag {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--paper);
        border: 1px solid var(--rule);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 2px;
        margin-bottom: 12px;
    }

    .exhibit-tag svg { flex-shrink: 0; }

    .exhibit-caption {
        font-size: 13px;
        color: var(--text-lo);
        margin-top: 14px;
        line-height: 1.5;
    }

    .preview-frame {
        background: var(--paper);
        border-radius: 3px;
        padding: 10px;
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    section[data-testid="stFileUploader"] {
        background: transparent;
        border: 1px dashed var(--rule);
        border-radius: 3px;
    }

    section[data-testid="stFileUploader"] small {
        color: var(--text-lo) !important;
    }


    /* =====================================================
       VERIFY BUTTON
       ===================================================== */

    div.stButton > button {
        width: 100%;
        height: 54px;
        border-radius: 2px;
        border: 1px solid var(--paper);
        font-family: var(--mono);
        font-size: 13px;
        letter-spacing: 0.06em;
        color: var(--ink);
        background: var(--paper);
        transition: background 0.15s ease, color 0.15s ease;
        margin-top: 8px;
    }

    div.stButton > button:hover {
        background: var(--ink);
        color: var(--paper);
    }

    div.stButton > button:focus-visible {
        outline: 2px solid var(--paper);
        outline-offset: 3px;
    }


    /* =====================================================
       VERDICT — ink stamp treatment
       ===================================================== */

    .verdict-wrap {
        max-width: 900px;
        margin: 36px auto 0 auto;
        padding: 0 24px;
        display: flex;
        justify-content: center;
    }

    .stamp {
        border: 4px solid var(--verdict-genuine);
        color: var(--verdict-genuine);
        border-radius: 6px;
        padding: 18px 44px;
        transform: rotate(-3deg);
        text-align: center;
        font-family: var(--mono);
        position: relative;
        animation: stamp-in 0.35s cubic-bezier(0.2, 0.9, 0.3, 1.2);
    }

    .stamp::before {
        content: "";
        position: absolute;
        inset: -4px;
        border-radius: 6px;
        border: 4px solid currentColor;
        opacity: 0.5;
        animation: stamp-bleed 0.6s ease-out forwards;
    }

    @keyframes stamp-bleed {
        from { transform: scale(1); opacity: 0.5; }
        to   { transform: scale(1.15); opacity: 0; }
    }

    .stamp.forged {
        border-color: var(--verdict-forged);
        color: var(--verdict-forged);
        transform: rotate(2.5deg);
    }

    .stamp-verdict {
        font-size: 30px;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    .stamp-sub {
        font-size: 11px;
        color: var(--text-lo);
        margin-top: 4px;
        letter-spacing: 0.03em;
    }

    @keyframes stamp-in {
        0%   { opacity: 0; transform: scale(1.4) rotate(-3deg); }
        100% { opacity: 1; transform: scale(1) rotate(-3deg); }
    }


    /* =====================================================
       READOUT (measurements)
       ===================================================== */

    .readout-wrap {
        max-width: 900px;
        margin: 28px auto 0 auto;
        padding: 0 24px;
    }

    .readout-row {
        display: flex;
        border-top: 1px solid var(--rule);
        padding: 14px 0;
        align-items: baseline;
        justify-content: space-between;
        opacity: 0;
        animation: row-in 0.4s ease forwards;
    }

    .readout-row:nth-child(1) { animation-delay: 0.15s; }
    .readout-row:nth-child(2) { animation-delay: 0.3s; }

    .readout-row:last-child {
        border-bottom: 1px solid var(--rule);
    }

    @keyframes row-in {
        from { opacity: 0; transform: translateX(-6px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    .readout-label {
        font-size: 13px;
        color: var(--text-lo);
    }

    .readout-value {
        font-family: var(--mono);
        font-size: 17px;
        color: var(--paper);
    }


    /* =====================================================
       EXPLANATION
       ===================================================== */

    .explain {
        max-width: 900px;
        margin: 30px auto 0 auto;
        padding: 20px 24px;
        border-left: 2px solid var(--paper-dim);
        color: var(--text-lo);
        font-size: 14px;
        line-height: 1.75;
    }

    .explain strong { color: var(--text-hi); }

    .explain-heading {
        font-family: var(--mono);
        font-size: 12px;
        color: var(--paper);
        margin-bottom: 8px;
        display: block;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        max-width: 900px;
        margin: 60px auto 0 auto;
        padding: 20px 24px 30px 24px;
        border-top: 1px solid var(--rule);
        color: var(--paper-dim);
        font-family: var(--mono);
        font-size: 11px;
        display: flex;
        justify-content: space-between;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DECORATIVE INK BLOTS (fixed, behind content, no interaction)
# ============================================================

st.markdown(
    """
    <svg class="ink-blot top-right" width="340" height="340" viewBox="0 0 340 340">
        <circle cx="170" cy="170" r="150" fill="none" stroke="#ece5d3" stroke-width="1"/>
        <circle cx="170" cy="170" r="110" fill="none" stroke="#ece5d3" stroke-width="1"/>
    </svg>
    <svg class="ink-blot bottom-left" width="300" height="300" viewBox="0 0 300 300">
        <path d="M20,220 C60,140 120,260 160,160 C190,90 230,180 270,60"
              fill="none" stroke="#ece5d3" stroke-width="1.2"/>
    </svg>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MASTHEAD
# ============================================================

st.markdown(
    """
    <div class="masthead">
        <div class="masthead-kicker">CASE FILE / OFFLINE SIGNATURE EXAMINATION</div>
        <div class="masthead-title">Signature<em>Guard</em></div>
        <div class="masthead-desc">
            A Siamese convolutional network reads two signatures into a
            shared embedding space and measures the distance between
            them. Submit a known reference and a questioned signature
            below to have the model render a finding.
        </div>
        <svg class="signature-draw" viewBox="0 0 260 60" xmlns="http://www.w3.org/2000/svg">
            <path d="M8,42 C22,10 34,10 40,32 C44,46 52,46 58,28
                     C64,12 72,12 78,34 C83,50 94,44 104,22
                     C112,6 122,10 128,26 C134,42 146,40 156,20
                     C163,6 176,12 182,30 C188,46 200,40 210,18
                     C216,6 228,14 252,16"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXHIBIT SECTION HEADING
# ============================================================

st.markdown(
    """
    <div class="exhibit-heading">
        <div class="eyebrow">STEP 01</div>
        <h2>Submit the exhibits</h2>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD SECTION
# ============================================================

col1, col2 = st.columns(2, gap="large")


with col1:
    st.markdown(
        """
        <div class="exhibit-card">
            <div class="exhibit-tag">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                    <path d="M4 20 L15 9 L19 13 L8 24 L4 24 Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/>
                    <line x1="15" y1="9" x2="19" y2="5" stroke="currentColor" stroke-width="1.4"/>
                </svg>
                EXHIBIT A — REFERENCE
            </div>
        """,
        unsafe_allow_html=True,
    )

    reference_file = st.file_uploader(
        "Reference Signature",
        type=["png", "jpg", "jpeg", "bmp"],
        key="reference",
        label_visibility="collapsed",
    )

    if reference_file:
        reference_image = Image.open(reference_file)
        st.markdown('<div class="preview-frame">', unsafe_allow_html=True)
        st.image(reference_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="exhibit-caption">Known genuine signature, on file.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="exhibit-caption">Awaiting upload of the known genuine signature.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


with col2:
    st.markdown(
        """
        <div class="exhibit-card">
            <div class="exhibit-tag">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                    <circle cx="10" cy="10" r="6.5" stroke="currentColor" stroke-width="1.6"/>
                    <line x1="15" y1="15" x2="21" y2="21" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
                </svg>
                EXHIBIT B — QUESTIONED
            </div>
        """,
        unsafe_allow_html=True,
    )

    questioned_file = st.file_uploader(
        "Questioned Signature",
        type=["png", "jpg", "jpeg", "bmp"],
        key="questioned",
        label_visibility="collapsed",
    )

    if questioned_file:
        questioned_image = Image.open(questioned_file)
        st.markdown('<div class="preview-frame">', unsafe_allow_html=True)
        st.image(questioned_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="exhibit-caption">Signature whose authorship is in question.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="exhibit-caption">Awaiting upload of the signature to be examined.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# VERIFY
# ============================================================

st.markdown(
    """
    <div class="exhibit-heading">
        <div class="eyebrow">STEP 02</div>
        <h2>Render a finding</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

verify_col = st.columns([1, 2, 1])[1]
with verify_col:
    verify = st.button("RUN COMPARISON", use_container_width=True)


# ============================================================
# VERIFICATION
# ============================================================

if verify:

    if reference_file is None:
        st.warning("Exhibit A is missing — upload the reference signature.")
        st.stop()

    if questioned_file is None:
        st.warning("Exhibit B is missing — upload the questioned signature.")
        st.stop()

    if not MODEL_PATH.exists():
        st.error("The trained model could not be found.")
        st.stop()

    reference_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    questioned_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    try:
        reference_temp.write(reference_file.getvalue())
        questioned_temp.write(questioned_file.getvalue())
        reference_temp.close()
        questioned_temp.close()

        with st.spinner("Measuring embedding distance..."):
            result = verify_signature(
                reference_temp.name, questioned_temp.name, THRESHOLD
            )

        verdict = result["verdict"]
        distance = result["distance"]
        threshold = result["threshold"]

        # --------------------------------------------------
        # STAMP
        # --------------------------------------------------

        if verdict == "GENUINE":
            st.markdown(
                """
                <div class="verdict-wrap">
                    <div class="stamp">
                        <div class="stamp-verdict">GENUINE</div>
                        <div class="stamp-sub">CONSISTENT WITH REFERENCE</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="verdict-wrap">
                    <div class="stamp forged">
                        <div class="stamp-verdict">FORGED</div>
                        <div class="stamp-sub">INCONSISTENT WITH REFERENCE</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --------------------------------------------------
        # READOUT
        # --------------------------------------------------

        st.markdown(
            f"""
            <div class="readout-wrap">
                <div class="readout-row">
                    <div class="readout-label">Euclidean distance</div>
                    <div class="readout-value">{distance:.6f}</div>
                </div>
                <div class="readout-row">
                    <div class="readout-label">Decision threshold</div>
                    <div class="readout-value">{threshold:.6f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------------------
        # EXPLANATION
        # --------------------------------------------------

        if verdict == "GENUINE":
            explanation = (
                f"The distance of <strong>{distance:.6f}</strong> falls "
                f"below the threshold of <strong>{threshold:.6f}</strong>. "
                "The closer two signatures sit in the embedding space, "
                "the more alike the model considers their stroke "
                "structure, proportion to be."
            )
        else:
            explanation = (
                f"The distance of <strong>{distance:.6f}</strong> exceeds "
                f"the threshold of <strong>{threshold:.6f}</strong>. "
                "A wider gap in the embedding space indicates the model "
                "found meaningfully different visual stroke patterns and proportions
                between the two exhibits."
            )

        st.markdown(
            f"""
            <div class="explain">
                <span class="explain-heading">HOW THIS FINDING WAS REACHED</span>
                {explanation}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "This is a model-generated estimate, not a certified "
            "forensic or legal opinion."
        )

    except Exception as error:
        st.error(f"Verification failed: {error}")

    finally:
        try:
            Path(reference_temp.name).unlink()
        except Exception:
            pass
        try:
            Path(questioned_temp.name).unlink()
        except Exception:
            pass


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <span>SIGNATUREGUARD</span>
        <span>SIAMESE CNN · CEDAR DATASET</span>
    </div>
    """,
    unsafe_allow_html=True,
)
