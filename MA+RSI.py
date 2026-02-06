import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. CẤU HÌNH TRANG & CSS DARK MODE ---
st.set_page_config(page_title="Stock Advisor Pro", layout="wide", page_icon="📈")

# CSS Tùy chỉnh để giống hệt ảnh bạn gửi
st.markdown("""
<style>
    /* Nền tổng thể màu tối */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* 1. Header Neon */
    .main-header {
        font-family: 'Arial Black', sans-serif;
        font-size: 3.5rem;
        text-align: center;
        background: -webkit-linear-gradient(#00ff88, #00b8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        margin-bottom: -10px;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #a0a0a0;
        margin-bottom: 30px;
        font-weight: 300;
    }
    
    /* 2. Disclaimer Box */
    .disclaimer-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .warning-icon { font-size: 1.2rem; color: #eab308; }
    .warning-title { color: #ef4444; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;}
    .warning-text { font-size: 0.9rem; color: #8b949e; margin-top: 5px; }
    .warning-highlight { color: #e0e0e0; font-weight: bold; text-decoration: underline; }
    
    /* 3. Input Container */
    .input-container {
        background-color: #161b22;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #30363d;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        max-width: 800px;
        margin: 0 auto 40px auto;
    }
    
    /* Tùy chỉnh Input Field */
    .stTextInput input, .stNumberInput input {
        background-color: #0d1117 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 15px !important;
        font-size: 1.1rem !important;
    }
    
    /* Tùy chỉnh Nút Bấm */
    .stButton button {
        width: 100%;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        font-weight: bold;
        padding: 15px 20px;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s;
        margin-top: 10px;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Metric Cards cho Dark Mode */
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        color: white;
    }
    .metric-label { font-size: 0.9rem; color: #9ca3af; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; font-weight: bold; margin-top: 5px; }
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {
        background-color: #161b22;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOGIC TÍNH TOÁN (Cập nhật thêm Cắt lỗ) ---

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest_strategy(prices, ma_series, rsi_series, stop_loss_pct):
    cash = 100_000_000 
    shares = 0
    initial_capital = cash
    trades = 0
    wins = 0
    
    trade_history = []
    
    # Chuyển sang array để loop nhanh
    price_val = prices.values
    ma_val = ma_series.values
    rsi_val = rsi_series.values
    dates = prices.index
    
    last_buy_price = 0
    
    for i in range(1, len(prices)):
        if np.isnan(ma_val[i]) or np.isnan(rsi_val[i]): continue
            
        current_price = price_val[i]
        current_ma = ma_val[i]
        current_rsi = rsi_val[i]
        
        # LOGIC MUA: Giá < MA và RSI < 30
        if shares == 0:
            if current_price < current_ma and current_rsi < 30:
                shares = cash / current_price
                last_buy_price = current_price
                cash = 0
                trade_history.append({'date': dates[i], 'type': 'BUY', 'price': current_price})
        
        # LOGIC BÁN
        elif shares > 0:
            # 1. Bán Cắt Lỗ (Nếu được kích hoạt)
            is_stop_loss = False
            if stop_loss_pct > 0:
                stop_price = last_buy_price * (1 - stop_loss_pct/100)
                if current_price <= stop_price:
                    is_stop_loss = True
            
            # 2. Bán Chốt Lời/Chiến thuật (Giá > MA và RSI > 70)
            is_take_profit = (current_price > current_ma and current_rsi > 70)
            
            if is_stop_loss or is_take_profit:
                sell_value = shares * current_price
                if current_price > last_buy_price: wins += 1
                
                cash = sell_value
                shares = 0
                trades += 1
                type_str = 'STOP LOSS' if is_stop_loss else 'TAKE PROFIT'
                trade_history.append({'date': dates[i], 'type': type_str, 'price': current_price})
            
    final_value = cash + (shares * price_val[-1])
    roi = ((final_value - initial_capital) / initial_capital) * 100
    
    return {'roi': roi, 'trades': trades, 'wins': wins, 'history': trade_history}

# --- 3. GIAO DIỆN CHÍNH ---

# Header Section
st.markdown('<h1 class="main-header">STOCK ADVISOR PRO</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hệ thống Hỗ trợ Phân tích & Quản trị Rủi ro Đầu tư</p>', unsafe_allow_html=True)

# Disclaimer Box
st.markdown("""
<div class="disclaimer-box">
    <div class="warning-title"><span class="warning-icon">⚠️</span> TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM</div>
    <div class="warning-text">
        Công cụ sử dụng thuật toán kỹ thuật (MA, RSI) để hỗ trợ tham khảo.<br>
        <span class="warning-highlight">KHÔNG phải lời khuyên đầu tư tài chính chính thức.</span><br>
        Người dùng tự chịu trách nhiệm. Dữ liệu Yahoo Finance.
    </div>
</div>
""", unsafe_allow_html=True)

# Input Section (Giống ảnh)
st.markdown('<div class="input-container">', unsafe_allow_html=True)
col_in1, col_in2 = st.columns([2, 1])
with col_in1:
    ticker_input = st.text_input("Mã cổ phiếu:", value="MBB", help="Ví dụ: VNM, HPG, FPT...").upper().strip()
with col_in2:
    stop_loss_input = st.number_input("Cắt lỗ % (0 = Tắt):", min_value=0.0, max_value=20.0, value=7.0, step=0.5)

run_btn = st.button("🚀 PHÂN TÍCH & BACKTEST")
st.markdown('</div>', unsafe_allow_html=True)

# --- 4. XỬ LÝ PHÂN TÍCH ---
if run_btn and ticker_input:
    # Xử lý mã VN
    ticker_symbol = f"{ticker_input}.VN" if not ticker_input.endswith(".VN") else ticker_input
    
    with st.spinner(f'Đang tải dữ liệu và chạy mô phỏng cho {ticker_input}...'):
        try:
            # 1. Lấy dữ liệu
            df = yf.download(ticker_symbol, period="max", progress=False)
            
            # Xử lý format cột yfinance mới
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs('Close', level=0, axis=1)
                df.columns = ['Close']
            elif 'Close' in df.columns:
                df = df[['Close']]
            elif 'Adj Close' in df.columns:
                 df = df[['Adj Close']].rename(columns={'Adj Close': 'Close'})

            if df.empty:
                st.error(f"Không tìm thấy dữ liệu cho mã {ticker_input}.")
                st.stop()

            # 2. Tính RSI
            df['RSI'] = calculate_rsi(df['Close'], 14)
            
            # 3. Backtest Loop
            results = []
            ma_range = range(5, 206, 10)
            
            for ma_period in ma_range:
                ma_series = df['Close'].rolling(window=ma_period).mean()
                perf = backtest_strategy(df['Close'], ma_series, df['RSI'], stop_loss_input)
                results.append({
                    'MA': ma_period,
                    'ROI': perf['roi'],
                    'Trades': perf['trades'],
                    'Wins': perf['wins'],
                    'History': perf['history'] # Lưu lịch sử để vẽ điểm mua bán sau này nếu cần
                })
            
            # 4. Tìm Best MA
            results_df = pd.DataFrame(results)
            best_row = results_df.loc[results_df['ROI'].idxmax()]
            best_ma = int(best_row['MA'])
            
            # Lấy data hiện tại
            df['BestMA'] = df['Close'].rolling(window=best_ma).mean()
            curr_price = df['Close'].iloc[-1]
            curr_rsi = df['RSI'].iloc[-1]
            curr_ma = df['BestMA'].iloc[-1]
            
            # Logic Khuyến Nghị
            rec_status = "QUAN SÁT"
            rec_color = "#9ca3af" # Gray
            rec_reason = "Chờ tín hiệu..."
            
            if curr_price < curr_ma and curr_rsi < 30:
                rec_status = "MUA MẠNH"
                rec_color = "#00ff88" # Neon Green
                rec_reason = f"Giá < MA{best_ma} & RSI Quá Bán ({curr_rsi:.1f})"
            elif curr_price > curr_ma and curr_rsi > 70:
                rec_status = "BÁN CHỐT LỜI"
                rec_color = "#ff4d4d" # Neon Red
                rec_reason = f"Giá > MA{best_ma} & RSI Quá Mua ({curr_rsi:.1f})"
            else:
                if curr_price > curr_ma:
                    rec_status = "NẮM GIỮ"
                    rec_color = "#3b82f6" # Blue
                    rec_reason = f"Xu hướng tăng trên MA{best_ma}"
                else:
                    rec_status = "CHỜ MUA"
                    rec_color = "#eab308" # Yellow
                    rec_reason = f"Xu hướng giảm dưới MA{best_ma}"

            # --- HIỂN THỊ KẾT QUẢ (DARK MODE UI) ---
            st.markdown("---")
            
            # 1. Kết quả tổng quan
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {rec_color};">
                    <div class="metric-label">KHUYẾN NGHỊ</div>
                    <div class="metric-value" style="color: {rec_color}; font-size: 1.5rem;">{rec_status}</div>
                    <div style="font-size: 0.8rem; color: #888;">{rec_reason}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #3b82f6;">
                    <div class="metric-label">GIÁ HIỆN TẠI</div>
                    <div class="metric-value">{curr_price:,.0f}</div>
                    <div style="font-size: 0.8rem;">VND</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                color_rsi = "#ff4d4d" if curr_rsi > 70 else ("#00ff88" if curr_rsi < 30 else "#e0e0e0")
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {color_rsi};">
                    <div class="metric-label">RSI (14)</div>
                    <div class="metric-value" style="color: {color_rsi}">{curr_rsi:.1f}</div>
                    <div style="font-size: 0.8rem;">Sức mạnh giá</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                roi_color = "#00ff88" if best_row['ROI'] > 0 else "#ff4d4d"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #eab308;">
                    <div class="metric-label">CHIẾN LƯỢC TỐI ƯU</div>
                    <div class="metric-value">MA {best_ma}</div>
                    <div style="font-size: 0.8rem; color: {roi_color}">Backtest ROI: {best_row['ROI']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # 2. Biểu đồ (Dark Theme Plotly)
            st.markdown("### 📉 Biểu Đồ Phân Tích")
            plot_df = df.tail(250).copy()
            
            fig = go.Figure()
            # Giá
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], mode='lines', name='Giá Đóng Cửa', line=dict(color='#00b8ff', width=2)))
            # Best MA
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BestMA'], mode='lines', name=f'MA {best_ma}', line=dict(color='#eab308', width=1, dash='dash')))
            
            # Thêm điểm mua bán (Nếu muốn chi tiết hơn)
            # (Phần này nâng cao, có thể thêm sau)

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", y=1, x=0, bgcolor='rgba(0,0,0,0)'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#333')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. Bảng Top Hiệu Quả
            st.markdown("### 🏆 Top 5 Đường MA Hiệu Quả Nhất")
            top_5 = results_df.sort_values(by='ROI', ascending=False).head(5)
            st.dataframe(
                top_5[['MA', 'ROI', 'Trades', 'Wins']],
                use_container_width=True,
                column_config={
                    "ROI": st.column_config.NumberColumn("Lợi Nhuận (%)", format="%.2f %%"),
                    "Trades": "Tổng Lệnh",
                    "Wins": "Lệnh Thắng"
                },
                hide_index=True
            )

        except Exception as e:
            st.error(f"Lỗi: {e}")

# Footer
st.markdown("<div style='text-align: center; color: #555; margin-top: 50px;'>© 2024 Stock Advisor Pro. Powered by Streamlit</div>", unsafe_allow_html=True)
