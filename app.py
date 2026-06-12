import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="US Bankruptcy Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    # 이 함수 안의 코드는 앱이 켜질 때 딱 한 번만 실행됩니다.
    m  = joblib.load("us_bankruptcy_model.joblib")
    s  = joblib.load("us_scaler.joblib")
    f  = joblib.load("us_features.joblib")
    t  = joblib.load("us_threshold.joblib")
    fi = joblib.load("us_feature_importance.joblib")
    return m, s, f, t, fi

try:
    model, scaler, FEATURES, THRESHOLD, FI = load_models()
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다: {e}")
    st.info("VS Code 터미널에서 `python train.py`를 먼저 실행하세요.")
    st.stop()

eps = 1e-9

def safe_div(a, b):
    return a / (b + eps)

def compute_features(x1, x4, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x17):
    """원본 변수 → 12개 파생 비율"""
    feats = {
        "WC_TA":     safe_div(x1 - x14, x10),
        "RE_TA":     safe_div(x15, x10),
        "EBIT_TA":   safe_div(x12, x10),
        "MV_TL":     safe_div(x8,  x17),
        "S_TA":      safe_div(x9,  x10),
        "NI_S":      safe_div(x6,  x9),
        "GP_S":      safe_div(x13, x9),
        "EBITDA_TA": safe_div(x4,  x10),
        "TL_TA":     safe_div(x17, x10),
        "CA_CL":     safe_div(x1,  x14),
        "LTD_TA":    safe_div(x11, x10),
        "REC_S":     safe_div(x7,  x9),
    }
    return feats

def altman_z(wc_ta, re_ta, ebit_ta, mv_tl, s_ta):
    """Altman Z-Score 근사 계산 (상장기업 원공식)"""
    return 1.2*wc_ta + 1.4*re_ta + 3.3*ebit_ta + 0.6*mv_tl + 1.0*s_ta

st.sidebar.header("재무제표 데이터 입력 (단위: 달러)")
st.sidebar.markdown("---")

st.sidebar.subheader("자산 항목")
x1  = st.sidebar.number_input("유동자산 (Current Assets)",          min_value=0.0, value=500_000.0,   step=10_000.0, format="%.0f")
x7  = st.sidebar.number_input("매출채권 (Total Receivables)",        min_value=0.0, value=80_000.0,    step=5_000.0,  format="%.0f")
x8  = st.sidebar.number_input("시가총액 (Market Value)",             min_value=0.0, value=1_000_000.0, step=50_000.0, format="%.0f")
x10 = st.sidebar.number_input("총자산 (Total Assets)",              min_value=1.0, value=1_500_000.0, step=10_000.0, format="%.0f")

st.sidebar.subheader("부채 항목")
x11 = st.sidebar.number_input("장기부채 (Long-term Debt)",          min_value=0.0, value=300_000.0,   step=10_000.0, format="%.0f")
x14 = st.sidebar.number_input("유동부채 (Current Liabilities)",     min_value=0.0, value=300_000.0,   step=10_000.0, format="%.0f")
x17 = st.sidebar.number_input("총부채 (Total Liabilities)",         min_value=0.0, value=800_000.0,   step=10_000.0, format="%.0f")
x15 = st.sidebar.number_input("이익잉여금 (Retained Earnings)",     value=200_000.0, step=10_000.0,   format="%.0f")

st.sidebar.subheader("손익 항목")
x2  = st.sidebar.number_input("매출원가 (COGS)",                     min_value=0.0, value=300_000.0,   step=10_000.0, format="%.0f")
x3  = st.sidebar.number_input("감가상각 (Depreciation & Amort.)",    min_value=0.0, value=50_000.0,    step=1_000.0,  format="%.0f")
x9  = st.sidebar.number_input("순매출 (Net Sales)",                  min_value=0.0, value=900_000.0,   step=10_000.0, format="%.0f")
x18 = st.sidebar.number_input("영업비용 (Operating Expenses)",      min_value=0.0, value=850_000.0,   step=10_000.0, format="%.0f")

x13 = x9 - x2                        
x12 = x9 - x18                       
x4  = x12 + x3                       
x6  = x12 - x10 * 0.03              

st.title("US Public Company Bankruptcy Risk Predictor")
st.markdown("**부도 위험도 실시간 진단**")
st.markdown("---")

if st.button("부도 위험 진단 실행", type="primary", use_container_width=True):

    if x10 <= 0:
        st.error("총자산은 0보다 커야 합니다.")
        st.stop()

    feat_dict = compute_features(x1, x4, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x17)
    feat_vec  = np.array([feat_dict[f] for f in FEATURES]).reshape(1, -1)

    feat_scaled    = scaler.transform(feat_vec)
    risk_proba     = float(model.predict_proba(feat_scaled)[0][1])
    risk_pct       = risk_proba * 100
    is_bankrupt    = risk_proba >= THRESHOLD

    z_score = altman_z(
        feat_dict["WC_TA"], feat_dict["RE_TA"],
        feat_dict["EBIT_TA"], feat_dict["MV_TL"], feat_dict["S_TA"]
    )

    t_pct = THRESHOLD * 100

    if risk_pct < t_pct * 0.5:     # 임계값의 절반 미만 (완전 안전)
        grade, color, icon = "LOW RISK",    "#2ecc71", "🟢"
        grade_kr = "저위험 — 안전 구간"
    elif risk_pct < t_pct * 0.8:   # 임계값의 80% 미만
        grade, color, icon = "WATCH",       "#f39c12", "🟡"
        grade_kr = "관찰 필요 — 경미한 위험 신호"
    elif risk_pct < t_pct:         # 임계값 도달 직전
        grade, color, icon = "CAUTION",     "#e67e22", "🟠"
        grade_kr = "주의 구간 — 재무 구조 점검 권고"
    else:                          # 임계값 초과 (15.2%는 무조건 여기 걸림!)
        grade, color, icon = "HIGH RISK",   "#e74c3c", "🔴"
        grade_kr = "고위험 — 부도 가능성 임박"

    col_gauge, col_metrics = st.columns([1, 1])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            title={"text": "부도 위험 확률 (%)", "font": {"size": 18}},
            number={"suffix": "%", "font": {"size": 40}},
            delta={"reference": t_pct, "increasing": {"color": "#e74c3c"}, "decreasing": {"color": "#2ecc71"}},
            gauge={
                "axis": {"range": [0, max(100, t_pct * 3)], "tickwidth": 1}, # 스케일도 동적 조정
                "bar":  {"color": color, "thickness": 0.25},
                "steps": [
                    {"range": [0,  t_pct * 0.5],  "color": "#d5f5e3"},
                    {"range": [t_pct * 0.5, t_pct * 0.8],  "color": "#fef9e7"},
                    {"range": [t_pct * 0.8, t_pct], "color": "#fdebd0"},
                    {"range": [t_pct, max(100, t_pct * 3)], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "#c0392b", "width": 3},
                    "thickness": 0.8,
                    "value": t_pct
                },
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f"<div style='text-align:center; padding:12px; background:{color}22; "
            f"border:2px solid {color}; border-radius:10px;'>"
            f"<span style='font-size:28px'>{icon}</span>&nbsp;"
            f"<strong style='font-size:20px; color:{color}'>{grade}</strong><br>"
            f"<span style='color:#666'>{grade_kr}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col_metrics:
        st.subheader("핵심 재무 비율")

        metrics_data = {
            "유동비율 (CA/CL)":    (feat_dict["CA_CL"],   1.5,   "배",  True),
            "부채비율 (TL/TA)":    (feat_dict["TL_TA"],   0.5,   "",    False),
            "ROA (EBIT/TA)":       (feat_dict["EBIT_TA"], 0.05,  "",    True),
            "순이익률 (NI/S)":     (feat_dict["NI_S"],    0.05,  "",    True),
            "자산회전율 (S/TA)":   (feat_dict["S_TA"],    0.6,   "배",  True),
            "Altman Z-Score":      (z_score,              1.81,  "",    True),
        }

        for label, (val, benchmark, unit, higher_better) in metrics_data.items():
            safe_val = val if not (np.isnan(val) or np.isinf(val)) else 0.0
            good = (safe_val >= benchmark) if higher_better else (safe_val <= benchmark)
            status_icon = "✅" if good else "⚠️"
            delta_color = "normal" if good else "inverse"
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.metric(
                    label=f"{status_icon} {label}",
                    value=f"{safe_val:.3f}{unit}",
                    delta=f"벤치마크: {benchmark}{unit}",
                    delta_color=delta_color
                )

    st.markdown("---")

    st.subheader("🕸️ 재무 건전성 레이더")

    radar_labels  = ["유동성\n(CA/CL)", "수익성\n(ROA)", "레버리지\n(1-TL/TA)",
                     "수익률\n(NI/S)", "효율성\n(S/TA)", "성장성\n(RE/TA)"]
    radar_raw     = [
        min(feat_dict["CA_CL"]  / 3.0,  1),
        min(max(feat_dict["EBIT_TA"] / 0.15 + 0.5, 0), 1),
        min(max(1 - feat_dict["TL_TA"], 0), 1),
        min(max(feat_dict["NI_S"]    / 0.2 + 0.5, 0), 1),
        min(feat_dict["S_TA"]   / 1.5,  1),
        min(max(feat_dict["RE_TA"]   / 0.3 + 0.5, 0), 1),
    ]
    radar_bench   = [0.5, 0.5, 0.7, 0.5, 0.4, 0.5]

    rgba_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_raw + [radar_raw[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        fillcolor=rgba_color,  
        line=dict(color=color, width=2),
        name="Analyzed Company"
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_bench + [radar_bench[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        fillcolor="rgba(100,100,200,0.1)",
        line=dict(color="rgba(100,100,200,0.6)", width=1.5, dash="dot"),
        name="업계 평균"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400,
        margin=dict(t=30, b=30, l=60, r=60)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("🔍 부도 예측 기여도 (변수 중요도)")
    fi_df = pd.DataFrame.from_dict(FI, orient="index", columns=["importance"])
    fi_df = fi_df.sort_values("importance", ascending=True).tail(12)
    fig_fi = px.bar(
        fi_df, x="importance", y=fi_df.index,
        orientation="h",
        color="importance",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        labels={"importance": "중요도", "index": "변수"},
    )
    fig_fi.update_layout(height=350, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.subheader("📄 리스크 관리 리포트")

    if z_score > 2.99:
        z_msg = f"✅ **안전 구간** (Z={z_score:.2f} > 2.99) — 단기 부도 가능성 낮음"
    elif z_score > 1.81:
        z_msg = f"⚠️ **회색 지대** (Z={z_score:.2f}, 1.81~2.99) — 모니터링 필요"
    else:
        z_msg = f"🔴 **위험 구간** (Z={z_score:.2f} < 1.81) — SEC 부도 신청 위험 높음"

    report_lines = [
        f"**Altman Z-Score:** {z_msg}",
        f"**AI 부도 확률:** `{risk_pct:.2f}%` (임계값: `{THRESHOLD*100:.1f}%`)",
        "",
        "**📌 세부 분석:**",
    ]

    if feat_dict["CA_CL"] < 1.0:
        report_lines.append("- ⚠️ 유동비율 < 1.0 — 단기 채무 불이행 위험이 높습니다.")
    if feat_dict["TL_TA"] > 0.8:
        report_lines.append("- 🔴 총부채비율 > 80% — 자본잠식 위험 구간입니다.")
    if feat_dict["EBIT_TA"] < 0:
        report_lines.append("- 🔴 EBIT < 0 — 영업 손실 발생 중입니다.")
    if feat_dict["NI_S"] < 0:
        report_lines.append("- ⚠️ 순이익률 음수 — 매출에서 손실이 발생하고 있습니다.")
    if feat_dict["RE_TA"] < 0:
        report_lines.append("- ⚠️ 이익잉여금 음수 — 누적 적자 상태입니다.")
    if feat_dict["MV_TL"] < 0.1:
        report_lines.append("- 🔴 시가총액/총부채 비율 매우 낮음 — 시장이 부도 위험을 반영 중입니다.")

    if is_bankrupt:
        report_lines += [
            "",
            "**종합 의견:**",
            f"AI 모델이 이 기업의 부도 가능성을 **{risk_pct:.1f}%**로 평가합니다. "
            "SEC 기준 **Chapter 11(회생)** 또는 **Chapter 7(청산)** 신청 가능성에 주의가 필요합니다. "
            "투자자산 포지션을 재검토하고, 경영진의 구조조정 계획 및 신용등급 변화를 즉시 모니터링하십시오."
        ]
    else:
        report_lines += [
            "",
            "**종합 의견:**",
            f"AI 모델 부도 확률 **{risk_pct:.1f}%** — 현재 단기 부도 위험은 낮은 수준입니다. "
            "다만 유동성 지표와 부채 구조를 정기적으로 점검하여 재무 건전성을 유지하시기 바랍니다."
        ]

    for line in report_lines:
        st.markdown(line)

else:
    st.info("👈 왼쪽 사이드바에 재무 수치를 입력한 후 **[부도 위험 진단 실행]** 버튼을 누르세요.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("학습 데이터", "78,682건", "1999~2018")
    with col2:
        st.metric("대상 기업", "8,262개사", "NYSE + NASDAQ")
    with col3:
        st.metric("파생 변수", "12개 재무 비율", "Altman Z 포함")

    st.markdown("---")
    st.markdown("""
    ### 사용 방법
    1. 사이드바에서 분석 대상 기업의 재무제표 수치를 입력합니다.
    2. [부도 위험 진단 실행] 버튼을 클릭합니다.
    3. AI가 실시간으로 부도 확률, 재무 비율, Altman Z-Score, 리스크 리포트를 출력합니다.

    ### 입력 변수 안내
    | 변수 | 설명 | 부도 관련성 |
    |------|------|------------|
    | 유동자산 | 1년 내 현금화 가능 자산 | 상 |
    | 총자산 | 기업이 보유한 모든 자산 | 상 |
    | 유동부채 | 1년 내 상환해야 할 부채 | 상 |
    | 총부채 | 전체 부채 합계 | 상 |
    | 이익잉여금 | 누적 순이익 | 상 |
    | 순매출 | 반품·할인 차감 후 매출 | 중 |
    | 시가총액 | 주식 시장가 × 발행주식수 | 중 |
    | 장기부채 | 1년 초과 부채 | 중 |
    """)
