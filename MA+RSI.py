import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG & CSS TÙY CHỈNH (Mô phỏng Tailwind UI cũ) ---
st.set_page_config(page_title="Hệ Thống Đánh Giá Cổ Phiếu AI", layout="wide", page_icon="📈")

# CSS để tạo giao diện Card (Thẻ) và làm đẹp giống bản HTML cũ
st.markdown("""
<style>
    /* Tổng thể nền */
    .stApp {
        background-color: #f3f4f6;
    }
    
    /* Style cho các Card (Hộp nội dung) */
    .css-1r6slb0, .css-12oz5g7, .stMarkdown, .stDataFrame, .stPlotlyChart {
        
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    
    /* Tạo khung trắng bo góc (Card) cho các container */
    .custom-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Tiêu đề chính */
    .main-header {
        color: #1e40af; /* Blue-800 */
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #4b5563; /* Gray-600 */
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Input to và đẹp hơn */
    .stTextInput input {
        font-size: 20px;
        font-weight: bold;
        text-transform: uppercase;
        padding: 10px;
    }
    
    /* Nút bấm lớn */
    .stButton button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    
    /* Màu sắc khuyến nghị */
    .rec-box-buy {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 5px;
        color: #065f46;
    }
    .rec-box-sell {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 5px;
        color: #991b1b;
    }
    .rec-box-hold {
        background-color: #f9fafb;
        border-left: 5px solid #9ca3af;
        padding: 15px;
        border-radius: 5px;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. LOGIC TÍNH TOÁN (Giữ nguyên từ bản trước) ---

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest_strategy(prices, ma_series, rsi_series):
    cash = 100_000_000 
    shares = 0
    initial_capital = cash
    trades = 0
    wins = 0
    
    price_val = prices.values
    ma_val = ma_series.values
    rsi_val = rsi_series.values
    
    trade_history = []
    
    for i in range(1, len(prices)):
        if np.isnan(ma_val[i]) or np.isnan(rsi_val[i]): continue
            
        current_price = price_val[i]
        current_ma = ma_val[i]
        current_rsi = rsi_val[i]
        
        # MUA: Giá < MA và RSI < 30
        if shares == 0 and current_price < current_ma and current_rsi < 30:
            shares = cash / current_price
            cash = 0
            trade_history.append({'type': 'BUY', 'price': current_price})
            
        # BÁN: Giá > MA và RSI > 70
        elif shares > 0 and current_price > current_ma and current_rsi > 70:
            sell_value = shares * current_price
            last_buy = trade_history[-1]['price']
            if current_price > last_buy: wins += 1
            cash = sell_value
            shares = 0
            trades += 1
            trade_history.append({'type': 'SELL', 'price': current_price})
            
    final_value = cash + (shares * price_val[-1])
    roi = ((final_value - initial_capital) / initial_capital) * 100
    
    return {'roi': roi, 'trades': trades, 'wins': wins}

# --- 2. GIAO DIỆN CHÍNH (LAYOUT MỚI) ---

# Header
st.markdown('<h1 class="main-header"><i class="fas fa-chart-line"></i> Hệ Thống Đánh Giá Cổ Phiếu AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Tối ưu hóa chiến lược kết hợp MA & RSI (Backtest tự động với dữ liệu Yahoo Finance)</p>', unsafe_allow_html=True)

# Input Section (Giữa màn hình giống bản HTML)
col_spacer1, col_input, col_btn, col_spacer2 = st.columns([1, 3, 1, 1])

with col_input:
    ticker_input = st.text_input("", placeholder="Nhập mã (VD: HPG)", label_visibility="collapsed").upper().strip()

with col_btn:
    st.write("") # Spacer để căn chỉnh nút bấm thẳng hàng với input
    run_btn = st.button("Phân Tích Ngay")

# Xử lý khi bấm nút
if run_btn and ticker_input:
    # Xử lý mã CK Việt Nam
    ticker_symbol = f"{ticker_input}.VN" if not ticker_input.endswith(".VN") else ticker_input
    
    with st.spinner(f'Đang lấy dữ liệu và chạy thuật toán cho {ticker_input}...'):
        try:
            # 1. Lấy dữ liệu
            df = yf.download(ticker_symbol, period="max", progress=False)
            
            # Xử lý định dạng cột của yfinance mới
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs('Close', level=0, axis=1)
                df.columns = ['Close']
            elif 'Close' in df.columns:
                df = df[['Close']]
            elif 'Adj Close' in df.columns:
                 df = df[['Adj Close']].rename(columns={'Adj Close': 'Close'})
            
            if df.empty:
                st.error(f"Không tìm thấy dữ liệu cho mã {ticker_input}. Vui lòng thử mã khác.")
                st.stop()

            # 2. Tính toán
            df['RSI'] = calculate_rsi(df['Close'], 14)
            results = []
            ma_range = range(5, 206, 10)
            
            # Progress bar ẩn
            progress_bar = st.empty()
            
            for idx, ma_period in enumerate(ma_range):
                ma_series = df['Close'].rolling(window=ma_period).mean()
                perf = backtest_strategy(df['Close'], ma_series, df['RSI'])
                results.append({
                    'MA': ma_period,
                    'Lợi Nhuận': perf['roi'],
                    'Số Lệnh': perf['trades'],
                    'Số Thắng': perf['wins']
                })
            
            # 3. Kết quả tốt nhất
            results_df = pd.DataFrame(results)
            best_row = results_df.loc[results_df['Lợi Nhuận'].idxmax()]
            best_ma = int(best_row['MA'])
            
            # Chuẩn bị dữ liệu hiển thị
            df['BestMA'] = df['Close'].rolling(window=best_ma).mean()
            curr_price = df['Close'].iloc[-1]
            curr_rsi = df['RSI'].iloc[-1]
            curr_ma = df['BestMA'].iloc[-1]
            
            # Logic Khuyến Nghị
            rec_html = ""
            status_text = ""
            reason_text = ""
            
            if curr_price < curr_ma and curr_rsi < 30:
                status_text = "MUA NGAY"
                reason_text = f"Giá ({curr_price:,.0f}) < MA{best_ma} và RSI vùng Quá Bán ({curr_rsi:.1f} < 30)."
                rec_html = f"""
                <div class="rec-box-buy">
                    <h3 style="margin:0">KHUYẾN NGHỊ: {status_text}</h3>
                    <p style="margin:5px 0 0 0">{reason_text}</p>
                </div>
                """
            elif curr_price > curr_ma and curr_rsi > 70:
                status_text = "BÁN NGAY"
                reason_text = f"Giá ({curr_price:,.0f}) > MA{best_ma} và RSI vùng Quá Mua ({curr_rsi:.1f} > 70)."
                rec_html = f"""
                <div class="rec-box-sell">
                    <h3 style="margin:0">KHUYẾN NGHỊ: {status_text}</h3>
                    <p style="margin:5px 0 0 0">{reason_text}</p>
                </div>
                """
            else:
                status_text = "NẮM GIỮ / QUAN SÁT"
                if curr_price > curr_ma:
                    reason_text = f"Giá đang trên MA{best_ma} (Xu hướng tăng), chờ RSI > 70 để chốt lời."
                else:
                    reason_text = f"Giá đang dưới MA{best_ma} (Xu hướng giảm), chờ RSI < 30 để bắt đáy."
                rec_html = f"""
                <div class="rec-box-hold">
                    <h3 style="margin:0">KHUYẾN NGHỊ: {status_text}</h3>
                    <p style="margin:5px 0 0 0">{reason_text}</p>
                </div>
                """

            # --- HIỂN THỊ KẾT QUẢ ---
            
            # 1. Hàng Card thông tin (4 cột)
            st.markdown("###") # Spacer
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="custom-card" style="border-left: 5px solid #3b82f6;">', unsafe_allow_html=True)
                st.metric("Giá Hiện Tại", f"{curr_price:,.0f} đ")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="custom-card" style="border-left: 5px solid #a855f7;">', unsafe_allow_html=True)
                status_rsi = "Quá Mua" if curr_rsi > 70 else ("Quá Bán" if curr_rsi < 30 else "Trung Tính")
                st.metric("RSI (14)", f"{curr_rsi:.2f}", status_rsi)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="custom-card" style="border-left: 5px solid #eab308;">', unsafe_allow_html=True)
                st.metric("MA Tối Ưu", f"MA {best_ma}", f"Lãi: {best_row['Lợi Nhuận']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                # Custom HTML card cho Recommendation để nổi bật
                st.markdown(rec_html, unsafe_allow_html=True)

            # 2. Biểu đồ
            with st.container():
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                plot_df = df.tail(200).copy()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], mode='lines', name='Giá', line=dict(color='#2563eb', width=2)))
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BestMA'], mode='lines', name=f'MA {best_ma}', line=dict(color='#fbbf24', width=2, dash='dash')))
                fig.update_layout(title="Biểu Đồ Giá & Đường MA Tối Ưu (200 phiên gần nhất)", height=450, xaxis_title="", yaxis_title="", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 3. Hai cột: Bảng tối ưu & Logic
            c_left, c_right = st.columns([1, 1])
            
            with c_left:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("##### 📊 Top 5 Đường MA Hiệu Quả Nhất")
                top_5 = results_df.sort_values(by='Lợi Nhuận', ascending=False).head(5)
                st.dataframe(top_5, hide_index=True, use_container_width=True, column_config={"Lợi Nhuận": st.column_config.NumberColumn(format="%.2f%%")})
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c_right:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("##### 💡 Logic Thuật Toán")
                st.markdown("""
                - **Dữ liệu:** Lấy trực tiếp từ Yahoo Finance (lịch sử tối đa).
                - **Quét MA:** Chạy thử nghiệm các đường MA từ 5 đến 205 (bước nhảy 10).
                - **Mua:** Khi Giá < MA và RSI < 30.
                - **Bán:** Khi Giá > MA và RSI > 70.
                - **Kết luận:** Hệ thống chọn đường MA có *Lợi nhuận cao nhất* trong quá khứ để đưa ra khuyến nghị hiện tại.
                """)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Lỗi: {e}")

# Footer
st.markdown("<div style='text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 2rem;'>AI Stock Analyzer - Powered by Streamlit & Yahoo Finance</div>", unsafe_allow_html=True)
