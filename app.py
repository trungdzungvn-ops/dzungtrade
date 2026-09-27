from openai import OpenAI
import streamlit as st

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Quant Trading & DCA Assistant", page_icon="📈", layout="wide"
)

st.title("🤖 Trợ lý Phân tích Định lượng & Chiến lược DCA")
st.markdown(
    "Hệ thống ứng dụng **DeepSeek API** hỗ trợ phân tích điểm vào lệnh, quản"
    " trị rủi ro và chiến lược rải vốn DCA cho **XAUUSD, BTCUSD & Cổ phiếu"
    " Việt Nam**."
)

# --- THANH CÔNG CỤ (SIDEBAR) ---
st.sidebar.header("⚙️ Cấu hình hệ thống")

# Lấy API Key an toàn từ Streamlit Secrets (nếu đã cấu hình) hoặc nhập trực tiếp
api_key_input = st.sidebar.text_input(
    "DeepSeek API Key:",
    type="password",
    value=(
        st.secrets.get("DEEPSEEK_API_KEY", "")
        if "DEEPSEEK_API_KEY" in st.secrets
        else ""
    ),
)

st.sidebar.divider()
st.sidebar.subheader("📊 Thông tin Thị trường & Lệnh")

# Chọn loại thị trường
market_category = st.sidebar.selectbox(
    "Chọn thị trường giao dịch",
    ["Forex (XAUUSD)", "Crypto (BTCUSD)", "Cổ phiếu Việt Nam"],
)

# Tự động gợi ý mã và dữ liệu mẫu tương ứng với thị trường
if market_category == "Forex (XAUUSD)":
  default_symbol = "XAUUSD"
  default_capital = 5000.0
  default_status = (
      "Đang kẹt lệnh Buy ở giá 2,650 (Khối lượng: 0.1 lot). Giá thị trường hiện"
      " tại: 2,625."
  )
  default_tech = (
      "Khung H4 chạm hỗ trợ mạnh 2,610 - 2,615. RSI H4 quá bán (chạm vùng 28)."
  )
elif market_category == "Crypto (BTCUSD)":
  default_symbol = "BTCUSD"
  default_capital = 2000.0
  default_status = (
      "Đang giữ vị thế Long ở giá 64,500$. Giá thị trường hiện tại: 62,800$."
  )
  default_tech = (
      "Khung D1 đang test đường biên dưới Bollinger Bands, volume giảm dần khi"
      " giảm giá."
  )
else:
  default_symbol = "HPG (hoặc VNM, ACB...)"
  default_capital = 50000000.0  # VNĐ
  default_status = (
      "Đã mua gom giá trung bình 27.5, hiện giá thị trường về 25.8. Tỷ trọng"
      " đang chiếm 30% tổng danh mục."
  )
  default_tech = (
      "Đang về vùng hỗ trợ MA200 khung ngày, khối lượng bán tháo cạn kiệt."
  )

symbol = st.sidebar.text_input("Mã tài sản", value=default_symbol)
capital = st.sidebar.number_input(
    "Tổng vốn hiện tại (USD hoặc VNĐ)", value=default_capital, step=1000.0
)

current_status = st.sidebar.text_area(
    "Trạng thái vị thế / Kẹt lệnh hiện tại", value=default_status, height=100
)

tech_indicators = st.sidebar.text_area(
    "Dữ liệu kỹ thuật / Khung thời gian / Hỗ trợ",
    value=default_tech,
    height=100,
)

# --- XỬ LÝ KHI BẤM NÚT PHÂN TÍCH ---
if st.sidebar.button("🚀 Chạy Phân Tích Định Lượng & DCA"):
  if not api_key_input:
    st.error(
        "⚠️ Vui lòng nhập DeepSeek API Key ở ô bên trên hoặc cấu hình trong"
        " Secrets!"
    )
  else:
    # Khởi tạo DeepSeek Client (Sử dụng base_url chính thức của DeepSeek)
    client = OpenAI(api_key=api_key_input, base_url="https://api.deepseek.com")

    # Xây dựng Prompt chuyên sâu tùy theo thị trường
    prompt = f"""
        Bạn là một chuyên gia phân tích định lượng tài chính và quản trị rủi ro chuyên nghiệp (Quant Trader). 
        Hãy lập một kế hoạch giao dịch và chiến lược DCA (Dollar-Cost Averaging) chi tiết dựa trên dữ liệu sau:
        - Thị trường: {market_category} - Mã: {symbol}
        - Tổng vốn phân bổ: {capital:,.0f}
        - Tình trạng vị thế hiện tại: {current_status}
        - Dữ liệu kỹ thuật / Hỗ trợ: {tech_indicators}

        Yêu cầu cấu trúc đầu ra bắt buộc bằng tiếng Việt:
        1. **NHẬN ĐỊNH XU HƯỚNG:** Đánh giá nhanh ngắn hạn và rủi ro hiện tại của mã tài sản này.
        2. **KẾ HOẠCH QUẢN TRỊ VỊ THẾ HIỆN TẠI:** Đánh giá trạng thái đang âm/lãi và mức độ rủi ro vốn.
        3. **CHIẾN LƯỢC RẢI VỐN DCA (Nếu thị trường tiếp tục đi ngược hướng):** 
           - Mốc DCA 1: Giá [...] - Tỷ trọng vốn/Khối lượng đề xuất: [...] - Cơ sở kỹ thuật: [...]
           - Mốc DCA 2: Giá [...] - Tỷ trọng vốn/Khối lượng đề xuất: [...] - Cơ sở kỹ thuật: [...]
        4. **QUẢN TRỊ RỦI RO & THOÁT LỆNH:** Điểm Stop Loss tổng (SL) cho toàn bộ chiến lược và Mục tiêu chốt lời (TP).
        """

    with st.spinner(
        "🔄 DeepSeek đang tính toán xác suất và lập kế hoạch chiến lược..."
    ):
      try:
        # Sử dụng model deepseek-chat (hoặc deepseek-reasoner nếu cần tư duy suy luận cực sâu)
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

        # Hiển thị kết quả ra màn hình chính
        st.success("✅ Phân tích hoàn tất thành công!")
        st.markdown("---")
        st.markdown(f"### 📈 Báo cáo Chiến lược Định lượng cho: **{symbol}**")
        st.markdown(result)

      except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi khi kết nối DeepSeek API: {e}")