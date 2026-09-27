import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
from vnstock import Market

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Quant DCA Assistant - VN Stock",
    page_icon="📈",
    layout="wide",
)

st.title("🤖 Quant Trading & DCA Assistant")
st.caption("VN Stock • Technical Analysis • DCA Calculator • DeepSeek")

# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=3600)
def get_vietnam_stock_data(symbol, start_date, end_date):
    try:
        # Vnstock v4+ Unified API.
        # This avoids relying on the older source-specific Quote API.
        market = Market()
        # Vnstock 4.x: equity() is a factory/method that receives the symbol,
        # then ohlcv() is called on the returned equity object.
        equity = market.equity(symbol.upper())
        df = equity.ohlcv(
            start=start_date,
            end=end_date,
            interval="1D",
        )

        if df is None or df.empty:
            return None, "Không có dữ liệu."

        df = df.copy()

        # Chuẩn hóa tên cột
        df.columns = [str(c).lower().strip() for c in df.columns]

        # Xác định cột ngày
        date_candidates = [
            "time", "date", "datetime", "trading_date",
            "timestamp", "tradingdate"
        ]
        date_col = next((c for c in date_candidates if c in df.columns), None)

        if date_col:
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            df["date"] = pd.to_datetime(df.index, errors="coerce")

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"Thiếu cột dữ liệu: {missing}"

        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

        if len(df) < 60:
            return None, "Dữ liệu lịch sử chưa đủ để tính các chỉ báo."

        return df, None

    except Exception as e:
        return None, str(e)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def calculate_indicators(df):
    d = df.copy()

    # Moving averages
    d["MA20"] = d["close"].rolling(20).mean()
    d["MA50"] = d["close"].rolling(50).mean()
    d["MA200"] = d["close"].rolling(200).mean()

    # EMA
    d["EMA12"] = d["close"].ewm(span=12, adjust=False).mean()
    d["EMA26"] = d["close"].ewm(span=26, adjust=False).mean()

    # MACD
    d["MACD"] = d["EMA12"] - d["EMA26"]
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    # RSI 14
    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    d["BB_MID"] = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["BB_UPPER"] = d["BB_MID"] + 2 * bb_std
    d["BB_LOWER"] = d["BB_MID"] - 2 * bb_std

    # ATR 14
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

    # Volume MA
    d["VOL_MA20"] = d["volume"].rolling(20).mean()
    d["VOL_RATIO"] = d["volume"] / d["VOL_MA20"]

    # Returns and drawdown
    d["RETURN_20D"] = d["close"].pct_change(20) * 100
    d["RETURN_60D"] = d["close"].pct_change(60) * 100
    d["PEAK_60D"] = d["close"].rolling(60).max()
    d["DD_60D"] = (d["close"] / d["PEAK_60D"] - 1) * 100

    return d


def safe_num(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{digits}f}"


# ============================================================
# DCA ENGINE
# ============================================================
def build_dca_plan(
    current_price,
    current_qty,
    current_avg,
    total_capital,
    levels_pct,
    allocations,
    lot_size=100,
):
    rows = []

    current_cost = current_qty * current_avg
    remaining_capital = total_capital

    qty = int(current_qty)
    cost = current_cost

    for i, (drop_pct, allocation_pct) in enumerate(
        zip(levels_pct, allocations), start=1
    ):
        target_price = current_price * (1 - drop_pct / 100)
        budget = total_capital * allocation_pct / 100

        budget = min(budget, remaining_capital)
        raw_qty = int(budget // target_price)

        # Làm tròn theo lô 100 cổ phiếu, nếu đủ
        if lot_size and raw_qty >= lot_size:
            dca_qty = (raw_qty // lot_size) * lot_size
        else:
            dca_qty = raw_qty

        actual_cost = dca_qty * target_price

        qty += dca_qty
        cost += actual_cost
        avg_after = cost / qty if qty > 0 else np.nan

        remaining_capital -= actual_cost

        rows.append(
            {
                "Mốc": f"DCA {i}",
                "Giảm từ giá hiện tại (%)": drop_pct,
                "Giá DCA": target_price,
                "Vốn dự kiến": budget,
                "SL cổ phiếu": dca_qty,
                "Vốn thực tế": actual_cost,
                "Tổng CP sau DCA": qty,
                "Giá vốn sau DCA": avg_after,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Cấu hình")

api_key = st.sidebar.text_input(
    "DeepSeek API Key",
    type="password",
    help="API key được dùng trực tiếp trong phiên Streamlit.",
)

symbol = st.sidebar.text_input(
    "Mã cổ phiếu",
    value="ACB",
).upper().strip()

capital = st.sidebar.number_input(
    "Vốn dành cho DCA (VNĐ)",
    min_value=0.0,
    value=50_000_000.0,
    step=5_000_000.0,
)

st.sidebar.subheader("📦 Vị thế hiện tại")

current_qty = st.sidebar.number_input(
    "Số cổ phiếu đang giữ",
    min_value=0,
    value=5000,
    step=100,
)

current_avg = st.sidebar.number_input(
    "Giá vốn bình quân hiện tại",
    min_value=0.0,
    value=27_500.0,
    step=100.0,
)

st.sidebar.subheader("🧮 DCA")

lot_size = st.sidebar.number_input(
    "Lô cổ phiếu",
    min_value=1,
    value=100,
    step=100,
)

dca1 = st.sidebar.number_input("DCA 1: giảm (%)", value=3.0, step=0.5)
dca2 = st.sidebar.number_input("DCA 2: giảm (%)", value=7.0, step=0.5)
dca3 = st.sidebar.number_input("DCA 3: giảm (%)", value=12.0, step=0.5)
dca4 = st.sidebar.number_input("DCA 4: giảm (%)", value=18.0, step=0.5)

alloc1 = st.sidebar.number_input("Vốn DCA 1 (%)", value=15.0, step=5.0)
alloc2 = st.sidebar.number_input("Vốn DCA 2 (%)", value=20.0, step=5.0)
alloc3 = st.sidebar.number_input("Vốn DCA 3 (%)", value=25.0, step=5.0)
alloc4 = st.sidebar.number_input("Vốn DCA 4 (%)", value=40.0, step=5.0)

allocation_total = alloc1 + alloc2 + alloc3 + alloc4

user_position = st.sidebar.text_area(
    "Ghi chú vị thế",
    value="Đã mua gom, cần đánh giá khả năng DCA khi điều chỉnh.",
    height=80,
)

run = st.sidebar.button(
    "🚀 PHÂN TÍCH",
    type="primary",
    use_container_width=True,
)

# ============================================================
# MAIN
# ============================================================
if run:
    if not symbol:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if allocation_total > 100:
        st.error(f"Tổng tỷ trọng DCA đang là {allocation_total:.1f}%, phải ≤ 100%.")
        st.stop()

    with st.spinner(f"Đang lấy dữ liệu {symbol} từ vnstock..."):
        df, error = get_vietnam_stock_data(
            symbol,
            "2023-01-01",
            "2026-12-31",
        )

    if error:
        st.error(f"Không lấy được dữ liệu: {error}")
        st.stop()

    df = calculate_indicators(df)
    latest = df.iloc[-1]

    current_price = float(latest["close"])

    # ========================================================
    # KPI
    # ========================================================
    st.success(f"Đã tải {len(df):,} phiên dữ liệu cho {symbol}.")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Giá hiện tại", f"{current_price:,.0f}")
    c2.metric("RSI 14", safe_num(latest["RSI14"], 1))
    c3.metric("MA20", safe_num(latest["MA20"], 0))
    c4.metric("MA50", safe_num(latest["MA50"], 0))
    c5.metric("MA200", safe_num(latest["MA200"], 0))
    c6.metric("ATR14", safe_num(latest["ATR14"], 0))

    # ========================================================
    # POSITION
    # ========================================================
    st.subheader("💰 Vị thế hiện tại")

    invested = current_qty * current_avg
    market_value = current_qty * current_price
    pnl = market_value - invested
    pnl_pct = (pnl / invested * 100) if invested else 0

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Số lượng", f"{current_qty:,}")
    p2.metric("Vốn đã đầu tư", f"{invested:,.0f} VNĐ")
    p3.metric("Giá trị hiện tại", f"{market_value:,.0f} VNĐ")
    p4.metric("Lãi/Lỗ", f"{pnl:,.0f} VNĐ", f"{pnl_pct:+.2f}%")

    # ========================================================
    # DCA
    # ========================================================
    st.subheader("🧮 Kế hoạch DCA")

    dca_df = build_dca_plan(
        current_price=current_price,
        current_qty=current_qty,
        current_avg=current_avg,
        total_capital=capital,
        levels_pct=[dca1, dca2, dca3, dca4],
        allocations=[alloc1, alloc2, alloc3, alloc4],
        lot_size=lot_size,
    )

    display_dca = dca_df.copy()

    for col in ["Giá DCA", "Vốn dự kiến", "Vốn thực tế", "Giá vốn sau DCA"]:
        display_dca[col] = display_dca[col].map(lambda x: f"{x:,.0f}")

    for col in ["SL cổ phiếu", "Tổng CP sau DCA"]:
        display_dca[col] = display_dca[col].map(lambda x: f"{x:,}")

    display_dca["Giảm từ giá hiện tại (%)"] = display_dca[
        "Giảm từ giá hiện tại (%)"
    ].map(lambda x: f"{x:.1f}%")

    st.dataframe(
        display_dca,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # TECHNICAL CHART
    # ========================================================
    st.subheader("📈 Biểu đồ kỹ thuật")

    chart_df = df.tail(250).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="Price",
        )
    )

    for col, name in [
        ("MA20", "MA20"),
        ("MA50", "MA50"),
        ("MA200", "MA200"),
        ("BB_UPPER", "BB Upper"),
        ("BB_LOWER", "BB Lower"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=chart_df[col],
                mode="lines",
                name=name,
            )
        )

    # DCA lines
    for i, row in dca_df.iterrows():
        fig.add_hline(
            y=row["Giá DCA"],
            line_dash="dot",
            annotation_text=f"DCA {i+1}: {row['Giá DCA']:,.0f}",
            annotation_position="right",
        )

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # INDICATOR TABLE
    # ========================================================
    st.subheader("📊 Chỉ báo định lượng")

    indicator_data = {
        "Chỉ báo": [
            "RSI 14",
            "MACD",
            "MACD Signal",
            "MACD Histogram",
            "MA20",
            "MA50",
            "MA200",
            "Bollinger Upper",
            "Bollinger Lower",
            "ATR14",
            "Volume / MA20 Volume",
            "Return 20D",
            "Return 60D",
            "Drawdown 60D",
        ],
        "Giá trị": [
            latest["RSI14"],
            latest["MACD"],
            latest["MACD_signal"],
            latest["MACD_hist"],
            latest["MA20"],
            latest["MA50"],
            latest["MA200"],
            latest["BB_UPPER"],
            latest["BB_LOWER"],
            latest["ATR14"],
            latest["VOL_RATIO"],
            latest["RETURN_20D"],
            latest["RETURN_60D"],
            latest["DD_60D"],
        ],
    }

    indicators_df = pd.DataFrame(indicator_data)
    indicators_df["Giá trị"] = indicators_df["Giá trị"].map(
        lambda x: "N/A" if pd.isna(x) else f"{x:,.2f}"
    )

    st.dataframe(
        indicators_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # MARKET SNAPSHOT FOR AI
    # ========================================================
    market_snapshot = f"""
Mã: {symbol}
Ngày dữ liệu cuối: {latest['date'].date()}
Giá hiện tại: {current_price:,.0f}
Open: {latest['open']:,.0f}
High: {latest['high']:,.0f}
Low: {latest['low']:,.0f}
Volume: {latest['volume']:,.0f}

RSI14: {latest['RSI14']:.2f}
MACD: {latest['MACD']:.4f}
MACD Signal: {latest['MACD_signal']:.4f}
MACD Histogram: {latest['MACD_hist']:.4f}

MA20: {latest['MA20']:.2f}
MA50: {latest['MA50']:.2f}
MA200: {latest['MA200']:.2f}

Bollinger Upper: {latest['BB_UPPER']:.2f}
Bollinger Mid: {latest['BB_MID']:.2f}
Bollinger Lower: {latest['BB_LOWER']:.2f}

ATR14: {latest['ATR14']:.2f}
Volume / Volume MA20: {latest['VOL_RATIO']:.2f}
Return 20D: {latest['RETURN_20D']:.2f}%
Return 60D: {latest['RETURN_60D']:.2f}%
Drawdown 60D: {latest['DD_60D']:.2f}%

Vị thế:
Số CP: {current_qty:,}
Giá vốn: {current_avg:,.0f}
Vốn đã đầu tư: {invested:,.0f}
Lãi/Lỗ hiện tại: {pnl:,.0f} ({pnl_pct:+.2f}%)

Vốn dành cho DCA: {capital:,.0f}

DCA PLAN:
{dca_df.to_string(index=False)}
"""

    # ========================================================
    # DEEPSEEK
    # ========================================================
    st.subheader("🤖 Phân tích DeepSeek")

    if not api_key:
        st.info("Nhập DeepSeek API Key ở sidebar để chạy phần AI.")
    else:
        with st.spinner("DeepSeek đang phân tích dữ liệu..."):
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )

                prompt = f"""
Bạn là trợ lý phân tích định lượng cho nhà đầu tư chứng khoán Việt Nam.

Hãy phân tích dữ liệu ĐÃ ĐƯỢC TÍNH TOÁN bằng Python dưới đây.
Không tự bịa giá hoặc chỉ báo không có trong dữ liệu.

{market_snapshot}

Ghi chú của nhà đầu tư:
{user_position}

Yêu cầu trả lời bằng tiếng Việt:

1. TÓM TẮT TRẠNG THÁI THỊ TRƯỜNG
- Xu hướng giá dựa trên MA20/MA50/MA200.
- RSI.
- MACD.
- Volume.
- Bollinger Bands.
- ATR và mức biến động.

2. ĐÁNH GIÁ VỊ THẾ
- Giá hiện tại so với giá vốn.
- Lãi/lỗ hiện tại.
- Rủi ro nếu tiếp tục DCA.
- Không khẳng định chắc chắn thị trường sẽ tăng hoặc giảm.

3. PHÂN TÍCH KẾ HOẠCH DCA
- Nhận xét từng tầng DCA.
- Giá vốn thay đổi như thế nào.
- Tầng nào cần thận trọng hơn.
- Không tự thay đổi số liệu trong bảng DCA.

4. QUẢN TRỊ RỦI RO
Đưa ra các kịch bản:
- Kịch bản tích cực.
- Kịch bản đi ngang.
- Kịch bản tiêu cực.

Nêu các mức giá cần theo dõi dựa trên dữ liệu kỹ thuật.
Không đưa ra cam kết lợi nhuận.

5. KẾT LUẬN
Tóm tắt thành checklist hành động theo từng kịch bản.
Đây là phân tích hỗ trợ quyết định, không phải khuyến nghị đầu tư chắc chắn.
"""

                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Bạn là chuyên gia phân tích định lượng. "
                                "Luôn phân biệt dữ liệu thực tế, giả định và kịch bản. "
                                "Không bịa số liệu."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    stream=False,
                )

                result = response.choices[0].message.content
                st.markdown(result)

            except Exception as e:
                st.error(f"Lỗi DeepSeek API: {e}")

    # ========================================================
    # DOWNLOAD
    # ========================================================
    st.subheader("📥 Xuất dữ liệu")

    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "⬇️ Download dữ liệu kỹ thuật CSV",
        data=csv,
        file_name=f"{symbol}_technical_data.csv",
        mime="text/csv",
    )

else:
    st.info(
        "Nhập mã cổ phiếu và thông tin vị thế ở sidebar, sau đó bấm "
        "**🚀 PHÂN TÍCH**."
    )
