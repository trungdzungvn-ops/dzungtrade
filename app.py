from openai import OpenAI
import streamlit as st

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Quant Trading & DCA Assistant (Webhook)",
    page_icon="📈",
    layout="wide",
)

st.title("🤖 Trợ lý Phân tích Định lượng & DCA (Tích hợp TradingView Webhook)")
st.markdown(
    "Hệ thống tự động nhận dữ liệu từ **TradingView Alert** và gọi **DeepSeek"
    " API** để tính toán chiến lược."
)

# --- THANH CÔNG CỤ (SIDEBAR) ---
st.sidebar.header("⚙️ Cấu hình hệ thống")
api_key_input = st.sidebar.text_input("DeepSeek API Key:", type="password")

# --- NHẬN DỮ LIỆU TỰ ĐỘNG TỪ TRADINGVIEW (WEBHOOK QUERY PARAMS) ---
# Khi TradingView bắn alert kèm đường dẫn dạng: ?symbol=XAUUSD&price=2625&status=...
query_params = st.query_params

auto_symbol = query_params.get("symbol", "XAUUSD")
auto_price = query_params.get("price", "2625")
auto_status = query_params.get(
    "status", "Giá chạm vùng hỗ trợ kỹ thuật, cần tính toán DCA."
)

st.sidebar.divider()
st.sidebar.subheader("📊 Thông tin Thị trường & Lệnh")

market_category = st.sidebar.selectbox(
    "Chọn thị trường giao dịch",
    ["Forex (XAUUSD)", "Crypto (BTCUSD)", "Cổ phiếu Việt Nam"],
)

symbol = st.sidebar.text_input("Mã tài sản", value=auto_symbol)
capital = st.sidebar.number_input(
    "Tổng vốn hiện tại (USD hoặc VNĐ)", value=5000.0, step=1000.0
)

current_status = st.sidebar.text_area(
    "Trạng thái vị thế / Giá từ TradingView",
    value=(
        f"Giá hiện tại từ TradingView: {auto_price}. Tình trạng:"
        f" {auto_status}"
    ),
    height=100,
)

tech_indicators = st.sidebar.text_area(
    "Dữ liệu kỹ thuật / Khung thời gian / Hỗ trợ",
    value=(
        "Tự động kích hoạt từ cảnh báo kỹ thuật trên biểu đồ TradingView."
    ),
    height=100,
)

# --- XỬ LÝ KHI BẤM NÚT HOẶC NHẬN TÍN HIỆU ---
if st.sidebar.button("🚀 Chạy Phân Tích Định Lượng & DCA") or query_params:
  if not api_key_input:
    st.error(
        "⚠️ Vui lòng nhập DeepSeek API Key ở thanh bên trái để hệ thống xử lý"
        " tự động!"
    )
  else:
    try:
      client = OpenAI(api_key=api_key_input, base_url="https://api.deepseek.com")

      prompt = f"""
            Bạn là một chuyên gia phân tích định lượng tài chính và quản trị rủi ro chuyên nghiệp (Quant Trader). 
            Hãy lập một kế hoạch giao dịch và chiến lược DCA (Dollar-Cost Averaging) chi tiết dựa trên dữ liệu thời gian thực sau:
            - Thị trường: {market_category} - Mã: {symbol}
            - Tổng vốn phân bổ: {capital:,.0f}
            - Dữ liệu giá & Vị thế thực tế: {current_status}
            - Dữ liệu kỹ thuật / Hỗ trợ: {tech_indicators}

            Yêu cầu cấu trúc đầu ra bắt buộc bằng tiếng Việt:
            1. **NHẬN ĐỊNH XU HƯỚNG:** Đánh giá nhanh ngắn hạn và rủi ro hiện tại dựa trên giá real-time.
            2. **KẾ HOẠCH QUẢN TRỊ VỊ THẾ HIỆN TẠI:** Đánh giá trạng thái và mức độ rủi ro vốn.
            3. **CHIẾN LƯỢC RẢI VỐN DCA (Nếu thị trường tiếp tục đi ngược hướng):** 
               - Mốc DCA 1: Giá [...] - Tỷ trọng vốn/Khối lượng đề xuất: [...] - Cơ sở kỹ thuật: [...]
               - Mốc DCA 2: Giá [...] - Tỷ trọng vốn/Khối lượng đề xuất: [...] - Cơ sở kỹ thuật: [...]
            4. **QUẢN TRỊ RỦI RO & THOÁT LỆNH:** Điểm Stop Loss tổng (SL) cho toàn bộ chiến lược và Mục tiêu chốt lời (TP).
            """

      with st.spinner(
          "🔄 Đang nhận tín hiệu từ TradingView và gọi DeepSeek phân"
          " tích..."
      ):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý phân tích định lượng tài chính xuất sắc,"
                        " đưa ra các con số tính toán rủi ro và DCA cực kỳ sắc"
                        " bén."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        result = response.choices[0].message.content

        st.success(
            "✅ Đã tiếp nhận tín hiệu và phân tích thành công từ dữ liệu"
            " thực tế!"
        )
        st.markdown("---")
        st.markdown(
            f"### 📈 Báo cáo Chiến lược Định lượng thời gian thực cho:"
            f" **{symbol}**"
        )
        st.markdown(result)

    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi khi kết nối DeepSeek API: {e}")