from openai import OpenAI
import streamlit as st
from vnstock import Quote  # Cú pháp mới của vnstock

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Quant Trading & DCA Assistant (VN Stock + DeepSeek)",
    page_icon="📈",
    layout="wide",
)

st.title("🤖 Trợ lý Phân tích Định lượng & DCA (Cổ phiếu Việt Nam & DeepSeek)")
st.markdown(
    "Hệ thống tự động kết nối **vnstock** để lấy dữ liệu giá thực tế và sử dụng"
    " **DeepSeek API** để lập chiến lược rải vốn DCA."
)

# --- THANH CÔNG CỤ (SIDEBAR) ---
st.sidebar.header("⚙️ Cấu hình hệ thống")
api_key_input = st.sidebar.text_input("DeepSeek API Key:", type="password")

st.sidebar.divider()
st.sidebar.subheader("📊 Thông tin Cổ phiếu Việt Nam")

# Nhập mã cổ phiếu (VD: HPG, ACB, TCB, VNM, MBB...)
symbol_input = st.sidebar.text_input(
    "Mã cổ phiếu (Viết hoa)", value="ACB"
).upper()
capital = st.sidebar.number_input(
    "Tổng vốn phân bổ (VNĐ)", value=50000000.0, step=5000000.0
)

user_position = st.sidebar.text_area(
    "Trạng thái vị thế hiện tại của bạn",
    value="Đã mua gom giá trung bình 27.5, hiện tỷ trọng chiếm 30% danh mục.",
    height=80,
)


# --- HÀM LẤY DỮ LIỆU TỰ ĐỘNG TỪ VNSTOCK ---
@st.cache_data(ttl=3600)  # Cache dữ liệu trong 1 giờ để tối ưu tốc độ
def get_vietnam_stock_data(symbol):
    try:
        # Khởi tạo đối tượng Quote với mã và nguồn dữ liệu (VCI)
        quote = Quote(symbol=symbol, source="VCI")

        # Lấy dữ liệu lịch sử với cú pháp mới
        df = quote.history(
            start="2025-01-01",
            end="2026-12-31",
            interval="1D",
        )

        if df is not None and not df.empty:
            latest = df.iloc[-1]  # Phiên giao dịch gần nhất
            prev = df.iloc[-2]    # Phiên liền trước
            change_pct = ((latest["close"] - prev["close"]) / prev["close"]) * 100

            market_info = (
                f"Mã: {symbol} | Ngày giao dịch gần nhất: {latest.get('time', 'N/A')}\n"
                f"- Giá đóng cửa gần nhất: {latest['close']:,.0f} VNĐ\n"
                f"- Thay đổi so với phiên trước: {change_pct:+.2f}%\n"
                f"- Khối lượng giao dịch: {latest['volume']:,} cổ phiếu\n"
                f"- Giá cao nhất phiên: {latest['high']:,.0f} VNĐ | Giá thấp nhất"
                f" phiên: {latest['low']:,.0f} VNĐ"
            )
            return market_info, df
        else:
            return None, None
    except Exception as e:
        return None, str(e)


# --- XỬ LÝ KHI BẤM NÚT PHÂN TÍCH ---
if st.sidebar.button("🚀 Lấy Dữ Liệu & Phân Tích Định Lượng"):
    if not api_key_input:
        st.error("⚠️ Vui lòng nhập DeepSeek API Key ở thanh bên trái!")
    elif not symbol_input:
        st.error("⚠️ Vui lòng nhập mã cổ phiếu!")
    else:
        with st.spinner(
            f"🔄 Đang kết nối lấy dữ liệu giá thực tế của mã {symbol_input}..."
        ):
            stock_data_text, raw_df = get_vietnam_stock_data(symbol_input)

        if stock_data_text is None:
            st.error(
                f"❌ Không thể lấy dữ liệu cho mã '{symbol_input}'. Lỗi:"
                f" {raw_df}. Vui lòng kiểm tra lại mã chứng khoán."
            )
        else:
            st.success("✅ Đã lấy dữ liệu giá thị trường thành công từ vnstock!")

            # Hiển thị thông tin giá nhanh ra giao diện
            st.info(f"📊 **Dữ liệu thị trường mới nhất của {symbol_input}:**")
            st.text(stock_data_text)

            # --- GỌI DEEPSEEK API PHÂN TÍCH CHIẾN LƯỢC ---
            with st.spinner("🤖 DeepSeek đang tính toán kỹ thuật và chiến lược DCA..."):
                try:
                    client = OpenAI(
                        api_key=api_key_input, base_url="https://api.deepseek.com"
                    )

                    prompt = f"""
                    Bạn là một chuyên gia phân tích định lượng tài chính và quản trị rủi ro chuyên nghiệp (Quant Trader trên thị trường chứng khoán Việt Nam). 
                    Hãy lập một kế hoạch giao dịch và chiến lược DCA (Dollar-Cost Averaging) chi tiết dựa trên dữ liệu thực tế sau:
                    - Mã cổ phiếu: {symbol_input}
                    - Dữ liệu giá thị trường thực tế mới nhất:
                    {stock_data_text}
                    - Tổng vốn phân bổ cho mã này: {capital:,.0f} VNĐ
                    - Trạng thái vị thế hiện tại của nhà đầu tư: {user_position}

                    Yêu cầu cấu trúc đầu ra bắt buộc bằng tiếng Việt:
                    1. **NHẬN ĐỊNH XU HƯỚNG KỸ THUẬT:** Đánh giá xu hướng ngắn hạn dựa trên mức giá và khối lượng thực tế vừa cung cấp.
                    2. **ĐÁNH GIÁ VỊ THẾ HIỆN TẠI:** Phân tích trạng thái lỗ/lãi và mức độ rủi ro với số vốn hiện tại.
                    3. **CHIẾN LƯỢC RẢI VỐN DCA (Nếu thị trường tiếp tục điều chỉnh):** 
                       - Mốc DCA 1: Vùng giá [...] - Tỷ trọng vốn/Số lượng cổ phiếu đề xuất: [...] - Cơ sở kỹ thuật: [...]
                       - Mốc DCA 2: Vùng giá [...] - Tỷ trọng vốn/Số lượng cổ phiếu đề xuất: [...] - Cơ sở kỹ thuật: [...]
                    4. **QUẢN TRỊ RỦI RO & THOÁT LỆNH:** Điểm cắt lỗ tổng (Stop Loss) và Mục tiêu chốt lời (Take Profit).
                    """

                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Bạn là trợ lý phân tích định lượng tài chính xuất sắc,"
                                    " chuyên gia phân tích thị trường chứng khoán Việt"
                                    " Nam."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        stream=False,
                    )

                    result = response.choices[0].message.content

                    st.markdown("---")
                    st.markdown(
                        f"### 📈 Báo cáo Chiến lược Định lượng cho: **{symbol_input}**"
                    )
                    st.markdown(result)

                except Exception as e:
                    st.error(f"❌ Đã xảy ra lỗi khi kết nối DeepSeek API: {e}")
