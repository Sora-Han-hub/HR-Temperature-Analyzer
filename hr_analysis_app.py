import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from datetime import time as dt_time

# ─────────────────────────────────────────────
# 변경 요약 (수정사항 1~5 반영):
# 1. UI 순서 변경: 업로드 -> 데이터 필터 -> 요약통계 -> 시각화 -> 데이터 분석표 -> 다운로드
# 2. name 필터 추가 및 미참여 제외 전처리, 시간 범위 필터 직접 입력(밤샘 처리 포함)으로 개선
# 3. 그래프 제목 동적 생성 및 소제목 입력 기능 추가
# 4. 시각화 배경 라인을 기관별 평균에서 개인별 raw 데이터(name 기준)로 변경
# 5. 데이터 분석표는 필터가 적용되지 않은 업로드 원본 전체 데이터 기준으로 생성되도록 수정
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HR & Temperature Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# 헬퍼: 컬럼 자동 탐지
# ─────────────────────────────────────────────
def _find_col(df: pd.DataFrame, candidates: list) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

# ─────────────────────────────────────────────
# 전처리 함수
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # 컬럼 자동 탐지 (필수 및 선택 보강)
    col_device = _find_col(df, ["device.id", "분석대상(device.id 등)"])
    col_sojok = _find_col(df, ["소속기관", "소속"])
    col_name = _find_col(df, ["name", "이름"])
    col_time = _find_col(df, ["time", "Time", "TIME"])
    col_datetime = _find_col(df, ["datetime"])
    col_hr = _find_col(df, ["HR"])
    
    col_hrdiff = _find_col(df, ["HR_편차", "HR편차", "HR_하위10P"])
    col_aws_t = _find_col(df, ["AWS 기온"])
    col_aws_h = _find_col(df, ["AWS 습도"])
    col_field_t = _find_col(df, ["현장 기온", "기온(°C)", "기온"])
    col_field_h = _find_col(df, ["현장 습도", "습도(%)", "습도"])
    col_watch_amb = _find_col(df, ["워치 ambient_temp"])
    col_watch_obj = _find_col(df, ["워치 object_temp"])

    missing = [n for n, c in [("HR", col_hr), ("time", col_time), ("device.id", col_device), ("소속기관", col_sojok)] if c is None]
    if missing:
        raise KeyError(f"필수 컬럼 없음: {missing}")

    # 1. 숫자 변환
    num_cols = [col_hr, col_hrdiff, col_aws_t, col_aws_h, col_field_t, col_field_h, col_watch_amb, col_watch_obj]
    for c in num_cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2. 필터링: HR > 0
    mask = df[col_hr] > 0
    df = df[mask].copy()
    
    # "미참여" 제외 전처리 (수정사항 2A)
    if col_name and col_name in df.columns:
        df = df[df[col_name].astype(str).str.strip() != "미참여"].copy()

    # 3. 시간 변수 파싱
    time_str = df[col_time].astype(str).str.extract(r'(\d{1,2}:\d{2})')[0].fillna("00:00")
    df["virtual_time"] = pd.to_datetime(time_str, format="%H:%M", errors="coerce").dt.time
    df["virtual_datetime"] = pd.to_datetime("2026-01-01 " + time_str, errors="coerce")

    meta = {
        "col_device": col_device,
        "col_sojok": col_sojok,
        "col_name": col_name,
        "col_time": col_time,
        "col_datetime": col_datetime,
        "col_hr": col_hr,
        "col_hrdiff": col_hrdiff,
        "col_aws_t": col_aws_t,
        "col_aws_h": col_aws_h,
        "col_field_t": col_field_t,
        "col_field_h": col_field_h,
        "col_watch_amb": col_watch_amb,
        "col_watch_obj": col_watch_obj,
    }
    return df, meta

# ─────────────────────────────────────────────
# 테이블 생성 헬퍼 (시간별 / 날짜별)
# ─────────────────────────────────────────────
def make_time_table(df, meta):
    df_temp = df.copy()
    if df_temp.empty:
        return pd.DataFrame()
        
    df_temp["time_str"] = df_temp["virtual_time"].apply(lambda x: x.strftime("%H:%M") if pd.notnull(x) else None)
    
    agg_dict = {}
    rename_cols = {}
    
    col_mappings = [
        (meta.get("col_hr"), "평균 심박수"),
        (meta.get("col_hrdiff"), "평균 심박수 편차"),
        (meta.get("col_aws_t"), "평균 AWS 기온"),
        (meta.get("col_aws_h"), "평균 AWS 습도"),
        (meta.get("col_field_t"), "평균 현장 기온"),
        (meta.get("col_field_h"), "평균 현장 습도"),
    ]
    
    for c, new_name in col_mappings:
        if c and c in df_temp.columns:
            agg_dict[c] = "mean"
            rename_cols[c] = new_name
            
    if not agg_dict:
        return pd.DataFrame()
        
    res = df_temp.groupby("time_str").agg(agg_dict).reset_index()
    res.rename(columns={"time_str": "시간"}, inplace=True)
    res.rename(columns=rename_cols, inplace=True)
    
    num_cols = [c for c in res.columns if c != "시간"]
    res[num_cols] = res[num_cols].round(1)
    
    res.sort_values("시간", inplace=True)
    return res

def make_date_table(df, meta):
    if df.empty or not meta.get("col_datetime") or meta["col_datetime"] not in df.columns:
        return pd.DataFrame()
        
    agg_dict = {}
    rename_cols = {}
    
    col_mappings = [
        (meta.get("col_hr"), "평균 심박수"),
        (meta.get("col_hrdiff"), "평균 심박수 편차"),
        (meta.get("col_aws_t"), "평균 AWS 기온"),
        (meta.get("col_aws_h"), "평균 AWS 습도"),
        (meta.get("col_field_t"), "평균 현장 기온"),
        (meta.get("col_field_h"), "평균 현장 습도"),
    ]
    
    for c, new_name in col_mappings:
        if c and c in df.columns:
            agg_dict[c] = "mean"
            rename_cols[c] = new_name
            
    if not agg_dict:
        return pd.DataFrame()
        
    res = df.groupby(meta["col_datetime"]).agg(agg_dict).reset_index()
    res.rename(columns={meta["col_datetime"]: "날짜 및 시간"}, inplace=True)
    res.rename(columns=rename_cols, inplace=True)
    
    try:
        res["_sort_key"] = pd.to_datetime(res["날짜 및 시간"], errors="coerce")
        res.sort_values("_sort_key", inplace=True)
        res.drop(columns=["_sort_key"], inplace=True)
    except:
        res.sort_values("날짜 및 시간", inplace=True)
        
    num_cols = [c for c in res.columns if c != "날짜 및 시간"]
    res[num_cols] = res[num_cols].round(1)
    
    return res

# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────
def build_figure(filtered_df: pd.DataFrame, df_bytime: pd.DataFrame, df_grand: pd.DataFrame,
                 meta: dict, title_text: str, show_raw_lines: bool = True) -> go.Figure:

    fig        = go.Figure()
    has_temp   = meta.get("has_temp", False) and "Total_Temp" in df_grand.columns
    temp_name  = meta.get("temp_name", "기온")

    # 배경: 개인별 raw 데이터 (수정사항 4)
    legend_shown = False
    col_id = meta["col_name"] if meta.get("col_name") else meta.get("col_device")
    
    if col_id and col_id in filtered_df.columns:
        ids = filtered_df[col_id].dropna().unique()
        
        # 데이터가 많을 경우 투명도 조절
        opacity = 0.35
        if len(ids) > 50: opacity = 0.15
        if len(ids) > 100: opacity = 0.05
            
        for uid in ids:
            sub = filtered_df[filtered_df[col_id] == uid]
            if sub.empty: continue
            
            if show_raw_lines:
                # 필터 적용 시: 진짜 raw 데이터 연결
                sub = sub.sort_values("virtual_datetime")
                x_data = sub["virtual_datetime"]
                y_data = sub[meta["col_hr"]]
                trace_name = "개인별 데이터"
            else:
                # 필터 미적용 시: 개인별 평균 선 (시간대별 평균)
                sub_agg = sub.groupby("virtual_datetime")[meta["col_hr"]].mean().reset_index()
                sub_agg = sub_agg.sort_values("virtual_datetime")
                x_data = sub_agg["virtual_datetime"]
                y_data = sub_agg[meta["col_hr"]]
                trace_name = "개인별 평균 선"

            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines",
                line=dict(color=f"rgba(180,180,190,{opacity})", width=1),
                name=trace_name,
                legendgroup="individual",
                showlegend=not legend_shown,
            ))
            legend_shown = True

    if not df_grand.empty:
        # 전체 평균 HR 에러 밴드 (표준편차)
        fig.add_trace(go.Scatter(
            x=df_grand["virtual_datetime"].tolist() + df_grand["virtual_datetime"].tolist()[::-1],
            y=(df_grand["Total_HR"] + df_grand["Total_HR_std"]).tolist() + (df_grand["Total_HR"] - df_grand["Total_HR_std"]).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(65, 105, 225, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="평균 심박수 ± 1 std",
            yaxis="y1",
        ))

        # 전체 평균 HR (좌축, royalblue)
        fig.add_trace(go.Scatter(
            x=df_grand["virtual_datetime"],
            y=df_grand["Total_HR"],
            mode="lines",
            line=dict(color="#4169e1", width=3),
            name="평균 심박수",
            yaxis="y1",
        ))

        # 선택된 평균 기온 (우축, firebrick 점선)
        if has_temp:
            fig.add_trace(go.Scatter(
                x=df_grand["virtual_datetime"],
                y=df_grand["Total_Temp"],
                mode="lines",
                line=dict(color="#b22222", width=2.5, dash="dot"),
                name=f"평균 {temp_name}",
                yaxis="y2",
            ))

    # X축 눈금 (동적 1시간 간격)
    if not df_grand.empty and pd.notna(df_grand["virtual_datetime"].min()):
        min_dt = df_grand["virtual_datetime"].min()
        max_dt = df_grand["virtual_datetime"].max()
        tickvals = pd.date_range(min_dt.floor('h'), max_dt.ceil('h'), freq="1h")
    else:
        tickvals = pd.date_range("2026-01-01 06:00", "2026-01-01 14:00", freq="1h")
    ticktext = [t.strftime("%H:%M") for t in tickvals]

    layout_kw = dict(
        paper_bgcolor="white",
        plot_bgcolor="#f9f9fc",
        font=dict(family="DM Sans, sans-serif", size=13, color="#333"),
        title=dict(
            text=f"<b>{title_text}</b>",
            font=dict(family="DM Serif Display, serif", size=18, color="#1a1a2e"),
            x=0.03,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e0e0e8",
            borderwidth=1,
        ),
        xaxis=dict(
            title="시간 (Time)",
            tickvals=tickvals,
            ticktext=ticktext,
            gridcolor="#e8e8f0",
            linecolor="#ccc",
        ),
        yaxis=dict(
            title=dict(text="심박수 (bpm)", font=dict(color="#4169e1")), 
            range=[50, 130],
            gridcolor="#e8e8f0",
            linecolor="#ccc",
            tickfont=dict(color="#4169e1"),
        ),
        margin=dict(l=65, r=75, t=90, b=60),
        hovermode="x unified",
    )

    if has_temp and not df_grand.empty:
        t_min = df_grand["Total_Temp"].min()
        t_max = df_grand["Total_Temp"].max()
        pad   = max((t_max - t_min) * 0.25, 1) if pd.notna(t_min) and pd.notna(t_max) else 1
        range_min = t_min - pad if pd.notna(t_min) else 0
        range_max = t_max + pad if pd.notna(t_max) else 40
        
        layout_kw["yaxis2"] = dict(
            title=dict(text=f"{temp_name} (°C)", font=dict(color="#b22222")),
            range=[range_min, range_max],
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(color="#b22222")
        )

    fig.update_layout(**layout_kw)
    return fig

# ─────────────────────────────────────────────
# Excel 변환
# ─────────────────────────────────────────────
def to_excel_bytes(df: pd.DataFrame, sheet_name="Sheet1") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()

# ─────────────────────────────────────────────
# ── UI ──
# ─────────────────────────────────────────────
st.title("심박수 & 기온 분석 대시보드")
st.markdown("#####  과제명: 시민참여형 적응 최적화를 위한 시민 〮사회 〮경제 〮환경 데이터 분석을 통한 정책평가기술 개발")

st.markdown("#### 📂 데이터 업로드")
uploaded_file = st.file_uploader(
    "xlsx 또는 csv 파일을 업로드하세요",
    type=["xlsx", "csv"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("👆 `.xlsx` 또는 `.csv` 파일을 업로드하면 자동으로 분석이 시작됩니다.")
    st.markdown("""
**필수 컬럼 안내**

| 컬럼명 | 설명 |
|---|---|
| `device.id` | 기기 식별자 |
| `소속기관` | 기관명 (그룹 기준) |
| `name` | 참여자 이름 |
| `time` | 측정 시각 (`HH:MM`) |
| `HR` | 심박수 |
| `HR_편차` | 심박수 편차 |
| `AWS 기온`, `현장 기온` | 기온 관련 (선택) |
| `AWS 습도`, `현장 습도` | 습도 관련 (선택) |
""")
    st.stop()

# ── 파일 읽기
try:
    raw_df = load_data(uploaded_file)
except Exception as e:
    st.error(f"파일 읽기 오류: {e}")
    st.stop()

# ── 전처리
try:
    df, meta = preprocess(raw_df)
except KeyError as e:
    st.error(f"필수 컬럼이 없습니다: {e}\n감지된 컬럼: {list(raw_df.columns)}")
    st.stop()
except Exception as e:
    st.error(f"전처리 오류: {e}")
    st.stop()

st.divider()

# ── 🔍 데이터 필터 (수정사항 1, 2)
st.markdown("#### 🔍 데이터 필터")

# (추가 수정) 시간 필터 배경색 통일성 부여
st.markdown("""
<style>
div[data-testid="column"]:has(div[data-testid="stTimeInput"]) {
    background-color: rgba(240, 242, 246, 0.8);
    padding: 10px 15px 0px 15px;
    border-radius: 10px;
    border: 1px solid #e0e0e8;
}
</style>
""", unsafe_allow_html=True)

f1, f2, f3, f4, f5 = st.columns(5)

with f1:
    start_time = st.time_input("시작 시각", value=dt_time(6, 0))
    end_time = st.time_input("종료 시각", value=dt_time(14, 0))

with f2:
    all_devices = sorted(df[meta["col_device"]].dropna().unique().tolist()) if meta["col_device"] else []
    sel_devices = st.multiselect("기기 ID", options=all_devices, default=[])

with f3:
    all_names = sorted(df[meta["col_name"]].dropna().unique().tolist()) if meta["col_name"] else []
    sel_names = st.multiselect("참여자", options=all_names, default=[])

with f4:
    all_orgs = sorted(df[meta["col_sojok"]].dropna().unique().tolist()) if meta["col_sojok"] else []
    sel_orgs = st.multiselect("소속기관", options=all_orgs, default=[])

with f5:
    temp_candidates = ["AWS 기온", "현장 기온", "워치 ambient_temp", "워치 object_temp"]
    avail_temps = []
    for c in temp_candidates:
        if c == "AWS 기온" and meta.get("col_aws_t"): avail_temps.append(meta["col_aws_t"])
        elif c == "현장 기온" and meta.get("col_field_t"): avail_temps.append(meta["col_field_t"])
        elif c == "워치 ambient_temp" and meta.get("col_watch_amb"): avail_temps.append(meta["col_watch_amb"])
        elif c == "워치 object_temp" and meta.get("col_watch_obj"): avail_temps.append(meta["col_watch_obj"])
    
    if avail_temps:
        sel_temp = st.selectbox("기온 변수 선택", options=avail_temps)
    else:
        sel_temp = None
        st.selectbox("기온 변수 선택", options=["데이터 없음"], disabled=True)

# ── 데이터 필터 적용 (수정사항 2B)
filtered_df = df.copy()

if start_time and end_time:
    if start_time <= end_time:
        filtered_df = filtered_df[(filtered_df["virtual_time"] >= start_time) & (filtered_df["virtual_time"] <= end_time)]
    else:
        # 밤샘 로직 처리
        filtered_df = filtered_df[(filtered_df["virtual_time"] >= start_time) | (filtered_df["virtual_time"] <= end_time)]
        # 자정을 넘긴 시간대는 하루 뒤로 미루기 (그래프 X축 정렬용)
        mask_next_day = filtered_df["virtual_time"] <= end_time
        filtered_df.loc[mask_next_day, "virtual_datetime"] += pd.Timedelta(days=1)

if sel_devices:
    filtered_df = filtered_df[filtered_df[meta["col_device"]].isin(sel_devices)]

if sel_names:
    filtered_df = filtered_df[filtered_df[meta["col_name"]].isin(sel_names)]

if sel_orgs:
    filtered_df = filtered_df[filtered_df[meta["col_sojok"]].isin(sel_orgs)]

st.divider()

# ── 📊 요약 통계 (수정사항 1)
st.markdown("#### 📊 요약 통계")
c1, c2, c3, c4 = st.columns(4)

with c1:
    if meta["col_name"] and meta["col_name"] in filtered_df.columns:
        names = filtered_df[meta["col_name"]].dropna().unique()
        valid_names = [n for n in names if str(n).strip() != "미참여"]
        st.metric("전체 실험 참여자 수", f"{len(valid_names)} 명")
    else:
        st.metric("전체 실험 참여자 수", "데이터 없음")
        
with c2:
    if meta["col_hr"] and meta["col_hr"] in filtered_df.columns:
        mean_hr = filtered_df[meta["col_hr"]].mean()
        val = f"{mean_hr:.1f} bpm" if pd.notna(mean_hr) else "데이터 없음"
        st.metric("전체 인원 평균 심박", val)
    else:
        st.metric("전체 인원 평균 심박", "데이터 없음")
        
with c3:
    if meta["col_aws_t"] and meta["col_aws_t"] in filtered_df.columns:
        aws_t = filtered_df[meta["col_aws_t"]].mean()
        val = f"{aws_t:.1f} °C" if pd.notna(aws_t) else "데이터 없음"
        st.metric("AWS 평균 기온", val)
    else:
        st.metric("AWS 평균 기온", "데이터 없음")
        
with c4:
    if meta["col_field_t"] and meta["col_field_t"] in filtered_df.columns:
        field_t = filtered_df[meta["col_field_t"]].mean()
        val = f"{field_t:.1f} °C" if pd.notna(field_t) else "데이터 없음"
        st.metric("현장 평균 기온", val)
    else:
        st.metric("현장 평균 기온", "데이터 없음")

st.divider()

# ── 📈 시각화 (수정사항 1, 3, 4)
st.markdown("#### 📈 시각화")

# 소제목 입력 (수정사항 3B)
subtitle = st.text_input("그래프 소제목 (선택사항)")

# 그래프 제목 동적 생성 (수정사항 3A)
base_title = "시간대 별 심박수 및 실외기온 변화 추이"
selected_items = []
if sel_devices:
    selected_items.extend(sel_devices)
if sel_names:
    selected_items.extend(sel_names)

if selected_items:
    title_text = f"{base_title}: {', '.join(map(str, selected_items))}"
else:
    title_text = base_title

if subtitle:
    title_text += f"<br><sub>{subtitle}</sub>"

# 그래프용 데이터 생성
grp_cols = [meta["col_sojok"], "virtual_datetime"]
agg_src = {meta["col_hr"]: "mean"}

if sel_temp and sel_temp in filtered_df.columns:
    agg_src[sel_temp] = "mean"

if filtered_df.empty:
    df_bytime = pd.DataFrame(columns=grp_cols + list(agg_src.keys()))
    df_grand = pd.DataFrame(columns=["virtual_datetime", "Total_HR", "Total_HR_std", "Total_Temp"])
else:
    df_bytime = filtered_df.groupby(grp_cols).agg(agg_src).reset_index()

    rename_map = {meta["col_hr"]: "Mean_HR"}
    if sel_temp:
        rename_map[sel_temp] = "Temperature"
    df_bytime.rename(columns=rename_map, inplace=True)

    # 리본(오차범위)을 정확하고 넓게 그리기 위해 개인 원본 데이터에서 직접 평균/표준편차 계산
    agg_kws_grand = dict(
        Total_HR=(meta["col_hr"], "mean"),
        Total_HR_std=(meta["col_hr"], "std")
    )
    if sel_temp and sel_temp in filtered_df.columns:
        agg_kws_grand["Total_Temp"] = (sel_temp, "mean")
        
    df_grand = filtered_df.groupby("virtual_datetime").agg(**agg_kws_grand).reset_index()
    df_grand["Total_HR_std"] = df_grand["Total_HR_std"].fillna(0)


meta_fig = meta.copy()
meta_fig["has_temp"] = "Temperature" in df_bytime.columns
meta_fig["temp_name"] = sel_temp if sel_temp else "기온"

# 필터 적용 여부 확인 (시간 필터 제외, 개체 필터 기준)
filters_applied = bool(sel_devices or sel_names or sel_orgs)

fig = build_figure(filtered_df, df_bytime, df_grand, meta_fig, title_text, show_raw_lines=filters_applied)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── 📋 데이터 분석표 (수정사항 1, 5)
st.markdown("#### 📋 데이터 분석표")
st.caption("※ 이 표는 업로드된 전체 데이터 기준입니다 (필터 미적용)")

# 필터가 적용되지 않은 원본 df 전체 사용 (수정사항 5)
df_by_time = make_time_table(df, meta)
df_by_date = make_date_table(df, meta)

tab1, tab2 = st.tabs(["📋 시간별 분석", "📋 날짜별 분석"])
with tab1:
    if df_by_time.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
    else:
        st.dataframe(df_by_time, use_container_width=True)
with tab2:
    if df_by_date.empty:
        st.info("데이터에 `datetime` 컬럼이 없거나 조건에 맞는 데이터가 없습니다.")
    else:
        st.dataframe(df_by_date, use_container_width=True)

st.divider()

# ── ⬇️ 다운로드
st.markdown("#### ⬇️ 다운로드")
dl1, dl2, dl3 = st.columns(3)

with dl1:
    if not df_by_time.empty:
        st.download_button(
            label="📥 시간별 분석표 (Excel)",
            data=to_excel_bytes(df_by_time, "시간별_분석"),
            file_name="HeartRate_시간별분석.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.download_button("📥 시간별 분석표 (Excel)", b"", disabled=True)
        
with dl2:
    if not df_by_date.empty:
        st.download_button(
            label="📥 날짜별 분석표 (Excel)",
            data=to_excel_bytes(df_by_date, "날짜별_분석"),
            file_name="HeartRate_날짜별분석.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.download_button("📥 날짜별 분석표 (Excel)", b"", disabled=True)
        
with dl3:
    try:
        png_bytes = fig.to_image(format="png", width=1400, height=700, scale=2)
        st.download_button(
            label="🖼️ 현재 그래프 다운로드 (PNG)",
            data=png_bytes,
            file_name="hr_temperature_filtered_chart.png",
            mime="image/png",
        )
    except Exception:
        st.info("PNG 저장: `pip install kaleido` 설치 후 사용 가능합니다.")
