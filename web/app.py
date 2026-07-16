"""qml-dna web UI: paste a DNA sequence, get classical + quantum predictions, browse the
benchmark dashboard, and download the generated report -- all without opening Jupyter.

Run: streamlit run web/app.py
Requires the API (api/main.py) running separately, default at http://localhost:8000.
"""
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = os.environ.get("QMLDNA_API_URL", "http://localhost:8000")
ROOT = Path(__file__).resolve().parents[1]

MODEL_BLURBS = {
    "classical": (
        "**Classical SVM** -- a support vector machine trained on k-mer (short DNA "
        "sub-sequence) frequencies. Fast and the most accurate of the three; a good default."
    ),
    "vqc": (
        "**Variational Quantum Circuit** -- a small trainable quantum circuit (6 qubits) "
        "that re-encodes the sequence at every layer. Runs on a quantum simulator; sub-second."
    ),
    "qkernel": (
        "**Quantum Kernel (QSVM)** -- measures quantum-circuit similarity between your "
        "sequence and a set of reference sequences, then classifies with a support vector "
        "machine. Most expensive: this runs as a background job (a few seconds per window)."
    ),
}

st.set_page_config(page_title="qml-dna", page_icon="\U0001F9EC", layout="wide")
st.sidebar.title("qml-dna")
page = st.sidebar.radio("Navigate", ["Predict", "Dashboard", "Report"])
st.sidebar.caption(f"API: {API_BASE}")


def api_get(path: str, **kwargs):
    r = requests.get(f"{API_BASE}{path}", timeout=30, **kwargs)
    return r


def api_post(path: str, **kwargs):
    r = requests.post(f"{API_BASE}{path}", timeout=30, **kwargs)
    return r


def render_prediction(result: dict) -> None:
    st.metric(
        label=f"{result['model']} — positive fraction",
        value=f"{result['positive_fraction']:.0%}",
        help=f"Fraction of windows classified positive (threshold={result['threshold']:.2f})",
    )
    df = pd.DataFrame(result["windows"])
    df["probability"] = df["probability"].round(3)
    st.bar_chart(df.set_index("start")["probability"])
    st.dataframe(df, use_container_width=True, hide_index=True)


if page == "Predict":
    st.title("Classify a DNA sequence")
    st.write(
        "Paste any DNA sequence (A/C/G/T/N). It's automatically split into overlapping "
        "windows matching each model's training size, and every window gets its own prediction."
    )

    sequence = st.text_area("DNA sequence", height=150, placeholder="ACGTACGTACGT...")
    col1, col2, col3 = st.columns(3)
    run_classical = col1.checkbox("Classical SVM", value=True)
    run_vqc = col2.checkbox("VQC (quantum, fast)", value=False)
    run_qkernel = col3.checkbox("Quantum Kernel (quantum, slower)", value=False)

    if st.button("Classify", type="primary", disabled=not sequence.strip()):
        if run_classical:
            st.subheader("Classical SVM")
            st.caption(MODEL_BLURBS["classical"])
            try:
                r = api_post("/predict/classical", params={"encoding": "kmer"}, json={"sequence": sequence})
                if r.status_code == 200:
                    render_prediction(r.json())
                else:
                    st.error(r.json().get("detail", r.text))
            except requests.RequestException as e:
                st.error(f"Could not reach the API at {API_BASE}: {e}")

        if run_vqc:
            st.subheader("Variational Quantum Circuit")
            st.caption(MODEL_BLURBS["vqc"])
            try:
                r = api_post("/predict/vqc", json={"sequence": sequence})
                if r.status_code == 200:
                    render_prediction(r.json())
                else:
                    st.error(r.json().get("detail", r.text))
            except requests.RequestException as e:
                st.error(f"Could not reach the API at {API_BASE}: {e}")

        if run_qkernel:
            st.subheader("Quantum Kernel (QSVM)")
            st.caption(MODEL_BLURBS["qkernel"])
            try:
                r = api_post("/predict/qkernel", json={"sequence": sequence})
                if r.status_code != 202:
                    st.error(r.json().get("detail", r.text))
                else:
                    job_id = r.json()["job_id"]
                    with st.spinner("Running quantum kernel inference (this is the slow one)..."):
                        for _ in range(120):
                            status = api_get(f"/jobs/{job_id}").json()
                            if status["status"] == "done":
                                render_prediction(status["result"])
                                break
                            if status["status"] == "error":
                                st.error(status["error"])
                                break
                            time.sleep(1)
                        else:
                            st.warning("Still running -- check back or increase the timeout.")
            except requests.RequestException as e:
                st.error(f"Could not reach the API at {API_BASE}: {e}")

elif page == "Dashboard":
    st.title("Benchmark dashboard")
    st.write("Classical vs. quantum model performance, from the last full pipeline run.")
    try:
        r = api_get("/benchmarks")
        if r.status_code != 200:
            st.warning(r.json().get("detail", "No benchmark data available yet."))
        else:
            df = pd.DataFrame(r.json())
            test_df = df[df["split"] == "test"] if "split" in df.columns else df
            metric_cols = [c for c in ("f1", "roc_auc", "pr_auc", "balanced_acc") if c in test_df.columns]
            if metric_cols and "model" in test_df.columns:
                melted = test_df.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="value")
                fig = px.bar(melted, x="model", y="value", color="metric", barmode="group", title="Test-split metrics by model")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
    except requests.RequestException as e:
        st.error(f"Could not reach the API at {API_BASE}: {e}")

elif page == "Report":
    st.title("Generated report")
    st.write("Publication-ready DOCX/PDF report from the most recent full pipeline run.")
    try:
        r = api_get("/reports/latest")
        if r.status_code != 200:
            st.warning(r.json().get("detail", "No report generated yet."))
        else:
            info = r.json()
            st.caption(f"Generated: {info.get('generated_at', 'unknown')}")
            for kind in ("pdf", "docx"):
                rel = info.get(kind)
                if rel:
                    path = ROOT / rel
                    if path.exists():
                        st.download_button(
                            f"Download {kind.upper()}",
                            data=path.read_bytes(),
                            file_name=path.name,
                            mime="application/pdf" if kind == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
    except requests.RequestException as e:
        st.error(f"Could not reach the API at {API_BASE}: {e}")
