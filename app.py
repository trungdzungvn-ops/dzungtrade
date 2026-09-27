import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
from vnstock import Market
import yfinance as yf

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Quant Trading & DCA Assistant",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Quant Trading & DCA Assistant")
st.caption("VN Stock • Forex / XAUUSD • Technical Analysis • DCA")

# ============================================================
# VN STOCK DATA
# ============================================================
@st.cache_data(ttl=300)
def load_vn_stock(symbol, days=500):
    market = Market()
    equity = market.equity(symbol.upper())
    df = equity.ohlcv(count=days, interval="1D")

    if df is None or df.empty:
        raise ValueError("Không lấy được dữ liệu VN Stock.")

    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    date_candidates = ["time", "date", "datetime", "trading_date"]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["date"] = pd.to_datetime(df.index, errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            raise ValueError(f"Thiếu cột {c}.")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    return df.sort_values("date").reset_index(drop=True)


# ============================================================
# FOREX / GOLD DATA
# ============================================================
@st.cache_data(ttl=300)
def load_forex(symbol):
    """
    Yahoo Finance market-data proxy.
    XAUUSD uses GC=F (gold futures), not broker-specific spot.
    """
    mapping = {
        "XAUUSD": "GC=F",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "CAD=X",
        "USDCHF": "CHF=X",
        "NZDUSD": "NZDUSD=X",
    }

    ticker_symbol = mapping.get(symbol.upper(), symbol.upper())
    ticker = yf.Ticker(ticker_symbol)

    df = ticker.history(
        period="2y",
        interval="1d",
        auto_adjust=False,
    )

    if df is None or df.empty:
        raise ValueError(f"Không lấy được dữ liệu {symbol}.")

    df = df.reset_index()
    df.columns = [
        str(c).lower().strip().replace(" ", "_")
        for c in df.columns
    ]

    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.to_datetime(df.index)

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    return df.sort_values("date").reset_index(drop=True)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def add_indicators(df):
    d = df.copy()

    d["MA20"] = d["close"].rolling(20).mean()
    d["MA50"] = d["close"].rolling(50).mean()
    d["MA200"] = d["close"].rolling(200).mean()

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()

    d["MACD"] = ema12 - ema26
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIGNAL"]

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI14"] = 100 - 100 / (1 + rs)

    d["BB_MID"] = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    d["BB_UPPER"] = d["BB_MID"] + 2 * std
    d["BB_LOWER"] = d["BB_MID"] - 2 * std

    prev_close = d["close"].shift(1)
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    d["ATR14"] = tr.rolling(14).mean()

    d["VOL_MA20"] = d["volume"].rolling(20).mean()
    d["VOL_RATIO"] = d["volume"] / d["VOL_MA20"].replace(0, np.nan)

    d["LOW20"] = d["low"].rolling(20).min()
    d["LOW50"] = d["low"].rolling(50).min()
    d["LOW100"] = d["low"].rolling(100).min()

    d["HIGH20"] = d["high"].rolling(20).max()
    d["HIGH50"] = d["high"].rolling(50).max()

    d["PEAK60"] = d["close"].rolling(60).max()
    d["DD60"] = (d["close"] / d["PEAK60"] - 1) * 100

    d["RET20"] = d["close"].pct_change(20) * 100
    d["RET60"] = d["close"].pct_change(60) * 100

    return d


# ============================================================
# BUY / DCA ENGINE
# ============================================================
def build_buy_levels(df, price):
    last = df.iloc[-1]

    atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else price * 0.03
    ma20 = float(last["MA20"]) if pd.notna(last["MA20"]) else price
    ma50 = float(last["MA50"]) if pd.notna(last["MA50"]) else price
    ma200 = float(last["MA200"]) if pd.notna(last["MA200"]) else price

    candidates = [
        ("Mua thăm dò", ma20, "MA20"),
        ("DCA 1", ma50, "MA50"),
        ("DCA 2", float(last["BB_LOWER"]), "Bollinger Lower"),
        ("DCA 3", ma200, "MA200"),
        ("Hỗ trợ 20D", float(last["LOW20"]), "Low 20D"),
        ("Hỗ trợ 50D", float(last["LOW50"]), "Low 50D"),
        ("Hỗ trợ 100D", float(last["LOW100"]), "Low 100D"),
    ]

    # Add ATR levels so a plan still exists if moving averages are above price.
    candidates += [
        ("ATR 1", price - 0.75 * atr, "0.75 ATR"),
        ("ATR 2", price - 1.50 * atr, "1.50 ATR"),
        ("ATR 3", price - 2.50 * atr, "2.50 ATR"),
    ]

    valid = []
    for name, value, basis in candidates:
        if pd.notna(value) and value > 0 and value < price * 1.005:
            valid.append(
                {
                    "Mức": name,
                    "Giá": float(value),
                    "Giảm (%)": (1 - float(value) / price) * 100,
                    "Cơ sở": basis,
                }
            )

    valid.sort(key=lambda x: x["Giá"], reverse=True)

    clean = []
    for item in valid:
        if not any(
            abs(item["Giá"] - x["Giá"]) / price < 0.012
            for x in clean
        ):
            clean.append(item)

    return clean[:5]


def make_dca_levels(levels, capital, first_lot=0.01, lot_step=0.01,
                    max_lot=0.10, contract_size=100):
    if not levels:
        return pd.DataFrame()

    weights = np.array([10, 20, 30, 40], dtype=float)
    n = min(len(levels), len(weights))
    weights = weights[:n]
    weights = weights / weights.sum()

    rows = []

    for i in range(n):
        level = levels[i]
        lot = min(max_lot, max(first_lot, first_lot * (i + 1)))
        lot = round(lot / lot_step) * lot_step

        price = level["Giá"]
        notional = lot * contract_size * price
        budget = capital * weights[i]

        rows.append(
            {
                "Mốc": level["Mức"],
                "Giá": price,
                "Giảm (%)": level["Giảm (%)"],
                "Tỷ trọng vốn (%)": weights[i] * 100,
                "Lot": lot,
                "Số CP": int(budget // price) if capital > 0 else 0,
                "Notional": notional,
                "Cơ sở": level["Cơ sở"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# AI
# ============================================================
def deepseek_analysis(api_key, title, snapshot):
    if not api_key:
        return None

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    prompt = f"""
Bạn là trợ lý phân tích định lượng.

Tài sản: {title}

Dữ liệu:
{snapshot}

Hãy trả lời bằng tiếng Việt:

1. Xu hướng hiện tại.
2. Vùng mua thăm dò.
3. DCA 1 / DCA 2 / DCA 3.
4. Điều kiện xác nhận trước khi tăng vị thế.
5. Khi nào nên dừng DCA.
6. Kịch bản tăng / đi ngang / giảm.
7. Checklist quản trị vốn.

Chỉ sử dụng dữ liệu được cung cấp.
Không tự bịa mức giá.
Không cam kết lợi nhuận.
Phân biệt dữ liệu kỹ thuật và kịch bản.
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý phân tích định lượng. "
                    "Không bịa dữ liệu và không cam kết kết quả."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )

    return response.choices[0].message.content


# ============================================================
# UI HELPERS
# ============================================================
def render_chart(df, buy_df, current_price, title):
    chart = df.tail(250)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart["date"],
            open=chart["open"],
            high=chart["high"],
            low=chart["low"],
            close=chart["close"],
            name=title,
        )
    )

    for col in ["MA20", "MA50", "MA200", "BB_UPPER", "BB_LOWER"]:
        fig.add_trace(
            go.Scatter(
                x=chart["date"],
                y=chart[col],
                mode="lines",
                name=col,
            )
        )

    if buy_df is not None and not buy_df.empty:
        for _, row in buy_df.iterrows():
            label = row["Mức"] if "Mức" in buy_df.columns else row["Mốc"]
            fig.add_hline(
                y=row["Giá"],
                line_dash="dot",
                annotation_text=(
                    f"{label} {row['Giá']:,.2f}"
                ),
                annotation_position="right",
            )

    fig.add_hline(
        y=current_price,
        line_dash="dash",
        annotation_text=f"CURRENT {current_price:,.2f}",
        annotation_position="left",
    )

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_common_analysis(df, asset_name, capital,
                           first_lot=0.01, lot_step=0.01,
                           max_lot=0.10, contract_size=100,
                           api_key=""):
    df = add_indicators(df)
    last = df.iloc[-1]
    price = float(last["close"])

    buy_levels = build_buy_levels(df, price)

    buy_df = pd.DataFrame(buy_levels)

    if not buy_df.empty:
        dca_df = make_dca_levels(
            buy_levels,
            capital,
            first_lot,
            lot_step,
            max_lot,
            contract_size,
        )
    else:
        dca_df = pd.DataFrame()

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Giá hiện tại", f"{price:,.2f}")
    c2.metric("RSI14", f"{last['RSI14']:.1f}")
    c3.metric("MA20", f"{last['MA20']:,.2f}")
    c4.metric("MA50", f"{last['MA50']:,.2f}")
    c5.metric("MA200", f"{last['MA200']:,.2f}")
    c6.metric("ATR14", f"{last['ATR14']:,.2f}")

    # Trend diagnostic
    score = 0
    if price > last["MA20"]:
        score += 1
    if last["MA20"] > last["MA50"]:
        score += 1
    if last["MA50"] > last["MA200"]:
        score += 1
    if 40 <= last["RSI14"] <= 60:
        score += 1
    elif last["RSI14"] < 40:
        score += 2

    if score >= 4:
        regime = "Xu hướng tích cực — ưu tiên chờ nhịp điều chỉnh"
    elif score >= 2:
        regime = "Trung tính — chia nhỏ vốn"
    else:
        regime = "Yếu — thận trọng với DCA"

    st.subheader("🎯 Trạng thái thị trường")
    st.info(f"Technical score: **{score}/5** — {regime}")

    st.subheader("🟢 Vùng mua / DCA")

    if not buy_df.empty:
        display = buy_df.copy()
        display["Giá"] = display["Giá"].map(lambda x: f"{x:,.2f}")
        display["Giảm (%)"] = display["Giảm (%)"].map(
            lambda x: f"{x:.2f}%"
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    if not dca_df.empty:
        st.subheader("💰 Kế hoạch DCA")

        display = dca_df.copy()
        display["Giá"] = display["Giá"].map(lambda x: f"{x:,.2f}")
        display["Giảm (%)"] = display["Giảm (%)"].map(
            lambda x: f"{x:.2f}%"
        )
        display["Tỷ trọng vốn (%)"] = display[
            "Tỷ trọng vốn (%)"
        ].map(lambda x: f"{x:.1f}%")
        display["Lot"] = display["Lot"].map(lambda x: f"{x:.2f}")
        display["Notional"] = display["Notional"].map(
            lambda x: f"${x:,.2f}"
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    # Risk levels
    atr = float(last["ATR14"])
    st.subheader("🛡️ Quản trị rủi ro tham khảo")

    r1, r2, r3 = st.columns(3)
    r1.metric("ATR Stop", f"{price - 2.5 * atr:,.2f}")
    r2.metric("+2 ATR", f"{price + 2 * atr:,.2f}")
    r3.metric("+4 ATR", f"{price + 4 * atr:,.2f}")

    render_chart(df, buy_df, price, asset_name)

    if api_key:
        snapshot = f"""
Asset: {asset_name}
Current price: {price:.4f}
RSI14: {last['RSI14']:.2f}
MA20: {last['MA20']:.4f}
MA50: {last['MA50']:.4f}
MA200: {last['MA200']:.4f}
MACD: {last['MACD']:.4f}
MACD Signal: {last['MACD_SIGNAL']:.4f}
BB Upper: {last['BB_UPPER']:.4f}
BB Lower: {last['BB_LOWER']:.4f}
ATR14: {last['ATR14']:.4f}
20D Return: {last['RET20']:.2f}%
60D Return: {last['RET60']:.2f}%
60D Drawdown: {last['DD60']:.2f}%

BUY LEVELS:
{buy_df.to_string(index=False)}

DCA:
{dca_df.to_string(index=False)}
"""
        st.subheader("🤖 DeepSeek")
        try:
            st.markdown(
                deepseek_analysis(
                    api_key,
                    asset_name,
                    snapshot,
                )
            )
        except Exception as e:
            st.error(f"DeepSeek API lỗi: {e}")

    st.subheader("📥 Xuất dữ liệu")
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{asset_name.replace('/', '_')}_analysis.csv",
        mime="text/csv",
    )


# ============================================================
# TWO TABS
# ============================================================
tab_vn, tab_fx = st.tabs(
    ["🇻🇳 VN STOCK — VNSTOCK", "🌎 FOREX / XAUUSD"]
)

# ============================================================
# TAB 1 — VN STOCK
# ============================================================
with tab_vn:
    st.header("🇻🇳 Phân tích cổ phiếu Việt Nam")

    col1, col2, col3 = st.columns(3)

    with col1:
        vn_symbol = st.text_input(
            "Mã cổ phiếu",
            value="ACB",
            key="vn_symbol",
        ).upper().strip()

    with col2:
        vn_capital = st.number_input(
            "Vốn DCA (VNĐ)",
            min_value=0.0,
            value=50_000_000.0,
            step=5_000_000.0,
            key="vn_capital",
        )

    with col3:
        vn_api = st.text_input(
            "DeepSeek API Key",
            type="password",
            key="vn_api",
        )

    vn_run = st.button(
        "🔎 PHÂN TÍCH VN STOCK",
        type="primary",
        key="vn_run",
    )

    if vn_run:
        try:
            with st.spinner(f"Đang lấy dữ liệu {vn_symbol}..."):
                vn_df = load_vn_stock(vn_symbol)

            st.success(
                f"{vn_symbol}: đã tải {len(vn_df):,} phiên dữ liệu."
            )

            render_common_analysis(
                vn_df,
                vn_symbol,
                vn_capital,
                first_lot=1,
                lot_step=1,
                max_lot=100000,
                contract_size=1,
                api_key=vn_api,
            )

        except Exception as e:
            st.error(f"❌ Lỗi VN Stock: {e}")

# ============================================================
# TAB 2 — FOREX / XAUUSD
# ============================================================
with tab_fx:
    st.header("🌎 Forex / XAUUSD — Phân tích điểm mua & DCA")

    f1, f2, f3 = st.columns(3)

    with f1:
        fx_symbol = st.selectbox(
            "Symbol",
            [
                "XAUUSD",
                "EURUSD",
                "GBPUSD",
                "USDJPY",
                "AUDUSD",
                "USDCAD",
                "USDCHF",
                "NZDUSD",
            ],
            key="fx_symbol",
        )

    with f2:
        fx_capital = st.number_input(
            "Vốn tham chiếu (USD)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            key="fx_capital",
        )

    with f3:
        fx_api = st.text_input(
            "DeepSeek API Key",
            type="password",
            key="fx_api",
        )

    st.markdown("### ⚙️ Lot / Contract")

    l1, l2, l3, l4 = st.columns(4)

    with l1:
        fx_start_lot = st.number_input(
            "Lot bắt đầu",
            min_value=0.01,
            value=0.01,
            step=0.01,
            format="%.2f",
            key="fx_start_lot",
        )

    with l2:
        fx_lot_step = st.number_input(
            "Lot step",
            min_value=0.01,
            value=0.01,
            step=0.01,
            format="%.2f",
            key="fx_lot_step",
        )

    with l3:
        fx_max_lot = st.number_input(
            "Lot tối đa",
            min_value=0.01,
            value=0.10,
            step=0.01,
            format="%.2f",
            key="fx_max_lot",
        )

    with l4:
        fx_contract = st.number_input(
            "Contract size",
            min_value=1.0,
            value=100.0,
            step=1.0,
            key="fx_contract",
        )

    fx_run = st.button(
        "🔎 PHÂN TÍCH FOREX / XAUUSD",
        type="primary",
        key="fx_run",
    )

    if fx_run:
        try:
            with st.spinner(f"Đang lấy dữ liệu {fx_symbol}..."):
                fx_df = load_forex(fx_symbol)

            st.success(
                f"{fx_symbol}: đã tải {len(fx_df):,} phiên dữ liệu."
            )

            render_common_analysis(
                fx_df,
                fx_symbol,
                fx_capital,
                first_lot=fx_start_lot,
                lot_step=fx_lot_step,
                max_lot=fx_max_lot,
                contract_size=fx_contract,
                api_key=fx_api,
            )

            if fx_symbol == "XAUUSD":
                st.warning(
                    "XAUUSD đang dùng GC=F làm dữ liệu tham chiếu. "
                    "Giá/contract size/spread của broker MT4/MT5 có thể khác."
                )

        except Exception as e:
            st.error(f"❌ Lỗi Forex/XAUUSD: {e}")

st.markdown("---")
st.caption(
    "Công cụ hỗ trợ phân tích và quản trị vốn; không đảm bảo lợi nhuận. "
    "Kiểm tra dữ liệu và thông số broker trước giao dịch thật."
)
