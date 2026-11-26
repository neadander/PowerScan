"""
PowerScan Web — Deteksi Lonjakan Listrik dengan Isolation Forest
Port dari pipeline notebook PowerScan_IsolationForest.ipynb ke Streamlit.

Jalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

RANDOM_STATE = 42
FEATURES = [
    "Global_active_power",
    "Voltage",
    "Global_intensity",
    "rolling_mean_24h",
    "rolling_std_24h",
    "gap_from_rolling_mean",
    "hour",
    "night_indicator",
]

REQUIRED_COLS = [
    "House_ID",
    "Date",
    "Time",
    "Global_active_power",
    "Voltage",
    "Global_intensity",
]
NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "Is_Anomaly",
]

st.set_page_config(page_title="PowerScan — Deteksi Lonjakan Listrik", layout="wide")


# ---------------------------------------------------------------------------
# Pipeline (sama dengan notebook)
# ---------------------------------------------------------------------------

def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    """Baca CSV dengan deteksi separator/desimal (mendukung format 'koma-id')."""
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8-sig", errors="replace")

    # Coba kombinasi separator/desimal yang umum
    candidates = [
        (";", ","), (",", "."), (",", ","), (";", "."), ("\t", ","),
    ]
    last_err = None
    for sep, dec in candidates:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, decimal=dec)
            if len(df.columns) >= 5:
                return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise ValueError(f"Tidak dapat membaca CSV. {last_err}")


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validasi, parsing timestamp, dan pembersihan — seperti sel notebook."""
    stats = {}
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    stats["baris_input"] = len(df)
    stats["timestamp_invalid"] = int(df["timestamp"].isna().sum())

    df = df.dropna(subset=["timestamp", "House_ID"] + [
        c for c in REQUIRED_COLS if c not in ("House_ID", "Date", "Time")
    ]).copy()
    df = df.sort_values(["House_ID", "timestamp"])
    stats["duplikat"] = int(df.duplicated(["House_ID", "timestamp"]).sum())
    df = df.drop_duplicates(["House_ID", "timestamp"], keep="first").reset_index(drop=True)
    stats["baris_bersih"] = len(df)
    stats["rumah"] = sorted(df["House_ID"].astype(str).unique())
    return df, stats


def add_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Feature engineering 8 fitur per rumah — sama dengan notebook."""
    def per_house(group):
        group = group.sort_values("timestamp").copy()
        group["hour"] = group["timestamp"].dt.hour
        group["night_indicator"] = (group["hour"] < 6).astype(int)
        group["rolling_mean_24h"] = (
            group["Global_active_power"].rolling(window, min_periods=1).mean()
        )
        group["rolling_std_24h"] = (
            group["Global_active_power"].rolling(window, min_periods=1).std().fillna(0)
        )
        group["gap_from_rolling_mean"] = (
            group["Global_active_power"] - group["rolling_mean_24h"]
        )
        return group

    out = df.groupby("House_ID", group_keys=False).apply(per_house)
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_pipeline(
    file_bytes: bytes,
    n_estimators: int,
    contamination: float,
    window: int,
) -> tuple[pd.DataFrame, dict, list]:
    """Jalankan seluruh pipeline. Mengembalikan (result_df, stats, events)."""
    df = read_csv_flexible(io.BytesIO(file_bytes))
    df, stats = preprocess(df)
    feature_df = add_features(df, window)

    results, models_note = [], []
    for house, part in feature_df.groupby("House_ID"):
        part = part.sort_values("timestamp").copy()
        scaler = MinMaxScaler()
        X = scaler.fit_transform(part[FEATURES])
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=RANDOM_STATE,
        )
        model.fit(X)
        part["model_prediction"] = model.predict(X)
        part["anomaly_flag"] = (part["model_prediction"] == -1).astype(int)
        part["anomaly_score"] = model.score_samples(X)
        results.append(part)
        models_note.append({"house": house, "n": len(part)})

    result_df = (
        pd.concat(results)
        .sort_values(["House_ID", "timestamp"])
        .reset_index(drop=True)
    )
    stats["total_flagged"] = int(result_df["anomaly_flag"].sum())
    stats["pct_flagged"] = float(result_df["anomaly_flag"].mean() * 100)
    stats["n_models"] = len(models_note)

    events = detect_spike_events(result_df)
    return result_df, stats, events


# ---------------------------------------------------------------------------
# Deteksi & interpretasi lonjakan
# ---------------------------------------------------------------------------

def detect_spike_events(result_df: pd.DataFrame, max_gap_hours: int = 3) -> list[dict]:
    """Kelompokkan flag anomali berurutan per rumah menjadi 'kejadian lonjakan'."""
    events = []
    for house, grp in result_df.groupby("House_ID"):
        grp = grp.sort_values("timestamp")
        flagged = grp[grp["anomaly_flag"] == 1]
        if flagged.empty:
            continue
        flagged = flagged.copy()
        flagged["delta_h"] = flagged["timestamp"].diff().dt.total_seconds() / 3600
        # mulai event baru bila jeda > max_gap_hours
        new_event = flagged["delta_h"].isna() | (flagged["delta_h"] > max_gap_hours)
        flagged["event_id"] = new_event.cumsum()

        for _, ev in flagged.groupby("event_id"):
            peak_idx = ev["gap_from_rolling_mean"].idxmax()
            peak = result_df.loc[peak_idx]
            reasons, advice = explain_event(ev)
            events.append({
                "house": house,
                "start": ev["timestamp"].min(),
                "end": ev["timestamp"].max(),
                "duration_h": int((ev["timestamp"].max() - ev["timestamp"].min()).total_seconds() // 3600) + 1,
                "n_points": len(ev),
                "peak_time": peak["timestamp"],
                "peak_power_kw": float(peak["Global_active_power"]),
                "avg_power_kw": float(ev["Global_active_power"].mean()),
                "max_gap_kw": float(ev["gap_from_rolling_mean"].max()),
                "min_voltage": float(ev["Voltage"].min()),
                "avg_voltage": float(ev["Voltage"].mean()),
                "night_share": float(ev["night_indicator"].mean()),
                "worst_score": float(ev["anomaly_score"].min()),
                "reasons": reasons,
                "advice": advice,
            })
    events.sort(key=lambda e: e["worst_score"])
    return events


def explain_event(ev: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Aturan heuristik sederhana untuk menjelaskan kemungkinan penyebab."""
    reasons, advice = [], []

    gap_max = ev["gap_from_rolling_mean"].max()
    mean_power = ev["Global_active_power"].mean()
    baseline = ev["rolling_mean_24h"].mean()
    night_share = ev["night_indicator"].mean()
    volt_mean = ev["Voltage"].mean()
    volt_min = ev["Voltage"].min()

    rel_gap = gap_max / baseline if baseline > 0 else 0

    # 1. Lonjakan daya mendadak
    if rel_gap > 0.5 or gap_max > 1.0:
        reasons.append(
            f"Lonjakan daya aktif mendadak (puncak +{gap_max:.2f} kW di atas rata-rata 24 jam)."
        )
        advice.append(
            "Periksa peralatan besar yang baru menyala (AC, pemanas air, induction, pompa). "
            "Hindari menyalakan beban besar secara bersamaan."
        )

    # 2. Aktivitas malam hari tidak biasa
    if night_share > 0.5 and mean_power > baseline * 1.3:
        reasons.append("Konsumsi tinggi terjadi pada jam malam (00.00–05.59).")
        advice.append(
            "Audit pemakaian malam hari: pastikan tidak ada peralatan standby/ceroboh "
            "atau jadwal mesin yang tidak disengaja."
        )

    # 3. Tegangan tidak stabil
    if volt_min < 200 or volt_mean < 210:
        reasons.append(
            f"Tegangan rendah/tidak stabil selama kejadian (min {volt_min:.1f} V, rata-rata {volt_mean:.1f} V)."
        )
        advice.append(
            "Periksa kondisi instalasi dan koordinasikan dengan PLN bila tegangan "
            "di bawah standar; tegangan rendah dapat memperparah lonjakan arus."
        )
    elif volt_mean > 250:
        reasons.append(f"Tegangan tinggi selama kejadian (rata-rata {volt_mean:.1f} V).")
        advice.append("Verifikasi tegangan dengan PLN; tegangan tinggi berisiko merusak peralatan.")

    # 4. Ketidakstabilan pola konsumsi
    if ev["rolling_std_24h"].mean() > ev["Global_active_power"].std() * 1.5:
        reasons.append("Pola konsumsi sangat tidak stabil dibanding baseline rumah.")
        advice.append("Tinjau ulang jadwal pemakaian; fluktuasi ekstrem memperberat kerja instalasi.")

    # 5. Sub-metering dominan (indikasi sumber beban)
    for sm, label in [("Sub_metering_1", "dapur"), ("Sub_metering_2", "cuci/pengering"),
                      ("Sub_metering_3", "pemanas air/AC")]:
        if sm in ev.columns and ev[sm].mean() > 0:
            total_sm = ev[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].mean().sum()
            if total_sm > 0 and ev[sm].mean() / total_sm > 0.6:
                reasons.append(f"Beban didominasi sub-metering {sm} (kemungkinan area {label}).")
                advice.append(f"Fokus inspeksi area {label}: periksa peralatan yang biasanya besar dayanya.")
                break

    if not reasons:
        reasons.append("Pola konsumsi menyimpang dari perilaku normal rumah (skor anomali Isolation Forest rendah).")
        advice.append(
            "Lakukan pengecekan visual instalasi, pastikan MCB/nominal daya memadai, "
            "dan catat kejadian untuk pemantauan lanjutan."
        )
    return reasons, advice


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("⚡ PowerScan — Deteksi Lonjakan Listrik")
st.caption(
    "Port Streamlit dari pipeline Isolation Forest per rumah di notebook PowerScan. "
    "Model: Isolation Forest (150 trees, contamination 0.10) dengan 8 fitur rolling 24 jam."
)

with st.sidebar:
    st.header("Parameter Model")
    n_estimators = st.number_input("Jumlah estimator", 50, 500, 150, step=25)
    contamination = st.number_input(
        "Contamination", 0.01, 0.5, 0.10, step=0.01,
        help="Perkiraan proporsi outlier yang diasumsikan model.",
    )
    window = st.number_input("Rolling window (jam)", 6, 168, 24, step=6)

    st.header("Unggah CSV")
    uploaded = st.file_uploader(
        "Dataset per-jam (mis. 40.000+ baris)",
        type=["csv"],
    )
    st.markdown(
        "Kolom wajib: `House_ID`, `Date`, `Time`, `Global_active_power`, "
        "`Voltage`, `Global_intensity`. Kolom lain opsional."
    )

if uploaded is None:
    st.info("Silakan unggah file CSV di sidebar untuk memulai analisis.")
    st.stop()

file_bytes = uploaded.getvalue()
with st.spinner("Memproses data & melatih Isolation Forest per rumah..."):
    try:
        result_df, stats, events = run_pipeline(
            file_bytes, int(n_estimators), float(contamination), int(window)
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal memproses file: {exc}")
        st.stop()

# ---------------------------------------------------------------------------
# Tab 1: Ringkasan
# ---------------------------------------------------------------------------
tab_summary, tab_chart, tab_cause, tab_table = st.tabs(
    ["📊 Ringkasan", "📈 Grafik Anomali", "🔍 Penyebab & Saran", "📋 Tabel Hasil"]
)

with tab_summary:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Baris input", f"{stats['baris_input']:,}")
    c2.metric("Baris bersih", f"{stats['baris_bersih']:,}")
    c3.metric("Jumlah rumah", stats["n_models"])
    c4.metric("Total flag anomali", f"{stats['total_flagged']:,}",
              help="Jumlah rekaman per-jam yang ditandai anomali.")
    c5.metric("Lonjakan terdeteksi", len(events),
              help="Kelompok kejadian anomali berurutan per rumah.")

    st.markdown("**Statistik pembersihan data**")
    st.json({
        "Timestamp invalid": stats["timestamp_invalid"],
        "Duplikat House_ID-timestamp": stats["duplikat"],
        "Persentase anomali": f"{stats['pct_flagged']:.2f}%",
    })

    per_house = (
        result_df.groupby("House_ID")
        .agg(
            total=("timestamp", "size"),
            anomali=("anomaly_flag", "sum"),
            skor_min=("anomaly_score", "min"),
        )
        .reset_index()
    )
    per_house["pct"] = (per_house["anomali"] / per_house["total"] * 100).round(2)
    st.markdown("**Flag anomali per rumah**")
    st.dataframe(per_house, use_container_width=True, hide_index=True)

    st.markdown("**Jumlah lonjakan per rumah**")
    ev_counts = pd.Series([e["house"] for e in events]).value_counts().reset_index()
    ev_counts.columns = ["House_ID", "Jumlah lonjakan"]
    st.bar_chart(ev_counts.set_index("House_ID"))

# ---------------------------------------------------------------------------
# Tab 2: Grafik
# ---------------------------------------------------------------------------
with tab_chart:
    houses = sorted(result_df["House_ID"].astype(str).unique())
    sel_house = st.selectbox("Pilih rumah", houses)
    view = result_df[result_df["House_ID"] == sel_house].sort_values("timestamp")
    flagged = view[view["anomaly_flag"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["timestamp"], y=view["Global_active_power"],
        mode="lines", name="Global_active_power", line=dict(width=1),
    ))
    fig.add_trace(go.Scatter(
        x=flagged["timestamp"], y=flagged["Global_active_power"],
        mode="markers", name="Anomali",
        marker=dict(color="red", size=6),
    ))
    fig.update_layout(
        title=f"Daya aktif per jam — {sel_house}",
        xaxis_title="Waktu", yaxis_title="kW", height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Distribusi anomaly_score**")
    fig2 = px.histogram(view, x="anomaly_score", nbins=60,
                        title=f"Skor anomali — {sel_house} (skor rendah = lebih abnormal)")
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Penyebab & saran
# ---------------------------------------------------------------------------
with tab_cause:
    st.markdown(f"### {len(events)} kejadian lonjakan terdeteksi")
    st.caption(
        "Interpretasi berbasis heuristik dari fitur model — bukan diagnosis pasti. "
        "Selalu verifikasi kondisi fisik instalasi."
    )
    if not events:
        st.success("Tidak ada lonjakan yang terdeteksi pada parameter ini.")
    else:
        for i, ev in enumerate(events[:20], 1):
            with st.expander(
                f"#{i} · {ev['house']} · {ev['start']:%Y-%m-%d %H:%M} → "
                f"{ev['end']:%Y-%m-%d %H:%M} · puncak {ev['peak_power_kw']:.2f} kW "
                f"({ev['duration_h']} jam, {ev['n_points']} titik)",
                expanded=(i <= 5),
            ):
                m = st.columns(4)
                m[0].metric("Puncak gap vs baseline", f"+{ev['max_gap_kw']:.2f} kW")
                m[1].metric("Tegangan min", f"{ev['min_voltage']:.1f} V")
                m[2].metric("Porsi malam", f"{ev['night_share']*100:.0f}%")
                m[3].metric("Skor terburuk", f"{ev['worst_score']:.3f}")
                st.markdown("**Kemungkinan penyebab:**")
                for r in ev["reasons"]:
                    st.write(f"- {r}")
                st.markdown("**Saran tindakan:**")
                for a in ev["advice"]:
                    st.write(f"- {a}")
        if len(events) > 20:
            st.info(f"Menampilkan 20 teratas dari {len(events)} kejadian.")

# ---------------------------------------------------------------------------
# Tab 4: Tabel & download
# ---------------------------------------------------------------------------
with tab_table:
    st.markdown(f"**{len(result_df):,} baris hasil**")
    st.dataframe(
        result_df.sort_values("anomaly_score").head(500),
        use_container_width=True,
        hide_index=True,
    )
    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download hasil CSV",
        data=csv_bytes,
        file_name="PowerScan_hasil_analisis.csv",
        mime="text/csv",
    )
