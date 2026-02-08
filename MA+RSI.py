import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import random
import textwrap
from datetime import datetime, timedelta

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(layout="wide", page_title="Stock Advisor", page_icon="📈")

# --- CSS TÙY CHỈNH (Giữ nguyên giao diện đẹp) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', 'Segoe UI', sans-serif; }
    
    /* HEADER */
    .main-title {
        text-align: center; font-weight: 900;
        background: -webkit-linear-gradient(45deg, #00E676, #69F0AE); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3.5rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 2px;
        text-shadow: 0px 0px 20px rgba(0, 230, 118, 0.3);
    }
    .sub-title {
        text-align: center; color: #E0E0E0 !important; font-size: 1.2rem;
        font-weight: 400; margin-bottom: 20px; letter-spacing: 0.5px;
    }

    /* DISCLAIMER */
    .disclaimer-box {
        background-color: #1E1E1E; border: 1px solid #444; border-radius: 8px;
        padding: 20px; margin: 0 auto 30px auto; text-align: center; max-width: 800px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .disclaimer-title { color: #FF5252; font-weight: bold; font-size: 1rem; text-transform: uppercase; margin-bottom: 12px; letter-spacing: 1px; }
    .d-line-1 { color: #AAA; font-size: 0.95rem; margin-bottom: 5px; }
    .d-line-2 { color: #E0E0E0; font-size: 1rem; font-weight: bold; margin-bottom: 5px; text-decoration: underline; text-decoration-color: #555; }
    .d-line-3 { color: #888; font-size: 0.85rem; font-style: italic; }

    /* RESULT CARD */
    .result-card {
        padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .bg-green { background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); }
    .bg-red { background: linear-gradient(135deg, #b71c1c 0%, #c62828 100%); }
    .bg-blue { background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%); }
    .result-title { font-size: 2.2rem; font-weight: 800; color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
    .result-reason { font-size: 1.1rem; color: #EEE; margin-top: 10px; font-style: italic; }

    /* REPORT BOX */
    .report-box { background-color: #1E1E1E; border: 1px solid #444; border-radius: 12px; padding: 25px; margin-top: 10px; }
    .report-header { color: #00E676; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px; border-bottom: 1px solid #444; padding-bottom: 10px; text-transform: uppercase; }
    .report-item { margin-bottom: 12px; font-size: 1rem; color: #FAFAFA; display: flex; align-items: center; }
    .icon-dot { margin-right: 12px; font-size: 1.2rem; }

    /* METRIC CARDS */
    .metric-container {
        background-color: #262730; border: 1px solid #41424C; border-radius: 12px;
        padding: 15px 10px; text-align: center; height: 160px;
        display: flex; flex-direction: column; justify-content: flex-start; align-items: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-label { font-size: 0.9rem; color: #FFF; font-weight: 700; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; height: 20px; display: flex; align-items: center; }
    .metric-value-box { flex-grow: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    .metric-value { font-size: 2.2rem; font-weight: 900; color: #FFF; line-height: 1; }
    .trend-badge { padding: 10px 30px; border-radius: 30px; font-size: 1.3rem; font-weight: 900; color: white; display: inline-block; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
    
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; height: 50px; font-size: 1.1rem; }
    
    /* BACKTEST RESULT BOX */
    .backtest-box {
        background: linear-gradient(135deg, #263238 0%, #37474F 100%);
        border-radius: 10px; padding: 20px; margin-top: 20px; text-align: center;
        border: 1px solid #546E7A;
    }
    .backtest-label { color: #CFD8DC; font-size: 1rem; margin-bottom: 5px; }
    .backtest-val { color: #00E676; font-size: 2rem; font-weight: 900; }
    .backtest-sub { color: #AAA; font-size: 0.8rem; }
    
    /* TABLE CUSTOM STYLE */
    .ma-table { width: 100%; border-collapse: collapse; font-size: 1.1rem; background-color: #1E1E1E; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px; }
    .ma-table th { background-color: #263238; color: #00E676; padding: 15px; text-align: center; font-weight: bold; border-bottom: 2px solid #444; text-transform: uppercase; font-size: 0.9rem; }
    .ma-table td { padding: 15px; text-align: center; border-bottom: 1px solid #333; color: #E0E0E0; }
    .ma-table tr:last-child td { border-bottom: none; }
    .ma-table tr:hover { background-color: rgba(255, 255, 255, 0.05); }
    .highlight-val { font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# --- HÀM TÍNH TOÁN CƠ BẢN ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- HÀM BACKTEST (CORE) ---
def run_backtest_for_ma(prices_series, ma_series, rsi_series, stop_loss_pct):
    # Chuyển sang numpy array để loop nhanh tối đa
    p_arr = prices_series.values
    ma_arr = ma_series.values
    rsi_arr = rsi_series.values
    
    cash = 100_000_000
    initial_capital = cash
    shares = 0
    trade_count = 0
    wins = 0
    
    last_buy_price = 0
    use_sl = stop_loss_pct > 0
    
    start_idx = 205 
    if len(p_arr) <= start_idx: return -999, -999, 0, 0

    for i in range(start_idx, len(p_arr)):
        price = p_arr[i]
        ma = ma_arr[i]
        rsi = rsi_arr[i]
        
        if np.isnan(ma) or np.isnan(rsi): continue
        
        # MUA
        if shares == 0:
            if price < ma and rsi < 30:
                shares = cash / price
                cash = 0
                last_buy_price = price
        
        # BÁN
        elif shares > 0:
            is_sell = False
            # Cắt lỗ
            if use_sl:
                pct_loss = (price - last_buy_price) / last_buy_price * 100
                if pct_loss <= -stop_loss_pct:
                    is_sell = True
            
            # Chốt lời chiến thuật
            if not is_sell:
                if price > ma and rsi > 70:
                    is_sell = True
            
            if is_sell:
                sell_val = shares * price
                if sell_val > shares * last_buy_price: wins += 1
                cash = sell_val
                shares = 0
                trade_count += 1
                
    final_val = cash + (shares * p_arr[-1])
    total_roi = ((final_val - initial_capital) / initial_capital) * 100
    
    start_date = prices_series.index[start_idx]
    end_date = prices_series.index[-1]
    days = (end_date - start_date).days
    years = days / 365.25
    avg_annual_roi = total_roi / years if years > 0 else 0
    
    return total_roi, avg_annual_roi, trade_count, wins

# --- HÀM TỐI ƯU HÓA KÉP (MA + SL) ---
def optimize_ma_strategy_dual(df, user_sl_pct):
    prices = df['Close']
    rsi = df['RSI']
    results = []
    
    # 1. Quét MA (5 -> 205)
    ma_ranges = range(5, 206, 10)
    
    # 2. Quét SL (0% -> 10%, bước 0.5%) 
    # Tạo mảng: 0, 0.5, 1.0, 1.5 ... 10.0
    sl_ranges = [i * 0.5 for i in range(0, 21)] 
    
    # Progress bar để user đỡ sốt ruột
    progress_text = "Đang chạy siêu tối ưu hóa (MA & Stoploss)..."
    my_bar = st.progress(0, text=progress_text)
    total_steps = len(ma_ranges)
    
    for idx, ma_period in enumerate(ma_ranges):
        # Tính MA một lần cho mỗi chu kỳ MA (Tối ưu hiệu năng)
        ma_series = prices.rolling(window=ma_period).mean()
        
        # --- A. Tìm SL tốt nhất cho đường MA này (Vòng lặp con) ---
        best_sl_for_this_ma = 0
        best_roi_for_this_ma = -99999
        best_stats_for_this_ma = None
        
        for sl_opt in sl_ranges:
            total_roi, annual_roi, trades, wins = run_backtest_for_ma(prices, ma_series, rsi, sl_opt)
            if annual_roi > best_roi_for_this_ma:
                best_roi_for_this_ma = annual_roi
                best_sl_for_this_ma = sl_opt
                best_stats_for_this_ma = (total_roi, annual_roi, trades, wins)
        
        # --- B. Tính hiệu quả với SL của User (Để so sánh) ---
        u_total, u_annual, u_trades, u_wins = run_backtest_for_ma(prices, ma_series, rsi, user_sl_pct)
        
        # Lưu kết quả
        if best_stats_for_this_ma:
            results.append({
                'MA': ma_period,
                # Thông số tối ưu nhất (AI tìm ra)
                'Opt SL': best_sl_for_this_ma,
                'Opt Annual ROI': best_stats_for_this_ma[1],
                'Opt Trades': best_stats_for_this_ma[2],
                'Opt Wins': best_stats_for_this_ma[3],
                # Thông số theo User nhập (Để so sánh)
                'User SL': user_sl_pct,
                'User Annual ROI': u_annual,
                'User Trades': u_trades
            })
            
        # Update progress
        my_bar.progress((idx + 1) / total_steps, text=progress_text)
        
    my_bar.empty() # Xóa thanh loading khi xong
        
    results_df = pd.DataFrame(results)
    if results_df.empty: return None, None
    
    # Sắp xếp theo kết quả Tối ưu nhất của AI
    best_res = results_df.loc[results_df['Opt Annual ROI'].idxmax()]
    return best_res, results_df

# --- HELPER UI ---
def render_metric_card(label, value, delta=None, color=None):
    delta_html = ""
    if delta is not None:
        delta_color = "#00E676" if delta > 0 else ("#FF5252" if delta < 0 else "#888")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "")
        delta_val = f"{abs(delta):.1f}"
        delta_html = f"<div style='font-size:0.9rem; margin-top:5px; color:{delta_color}'>{arrow} {delta_val} vs phiên trước</div>"
    
    if color:
        value_html = f"<div class='trend-badge' style='background-color:{color}'>{value}</div>"
    else:
        value_html = f"<div class='metric-value'>{value}</div>"

    card_html = f"<div class='metric-container'><div class='metric-label'>{label}</div><div class='metric-value-box'>{value_html}{delta_html}</div></div>"
    st.markdown(card_html, unsafe_allow_html=True)

# --- MAIN APP ---
st.markdown("<h1 class='main-title'>STOCK ADVISOR</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Hệ thống Tối ưu hóa Kép (MA & Stoploss)</p>", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer-box'>
    <div class='disclaimer-title'>⚠️ TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM</div>
    <div class='d-line-1'>Công cụ tự động tối ưu hóa tham số quá khứ.</div>
    <div class='d-line-2'>KHÔNG phải lời khuyên đầu tư tài chính chính thức.</div>
    <div class='d-line-3'>Người dùng tự chịu trách nhiệm. Dữ liệu Yahoo Finance.</div>
</div>
""", unsafe_allow_html=True)

# === CALLBACK XỬ LÝ SỰ KIỆN ===
def trigger_analysis():
    st.session_state['run_analysis'] = True
    if 'ticker_input_key' in st.session_state:
        st.session_state['confirmed_ticker'] = st.session_state['ticker_input_key'].strip().upper()

# === PHẦN NHẬP LIỆU ===
col1, col2, col3 = st.columns([1, 2, 1]) 
with col2:
    c_ticker, c_sl = st.columns([2, 1])
    with c_ticker:
        st.text_input(
            "Mã cổ phiếu:", 
            value=st.session_state.get('confirmed_ticker', ''), 
            placeholder="VD: HPG, VNM...",
            key="ticker_input_key",
            on_change=trigger_analysis
        )
    with c_sl:
        stop_loss_input = st.number_input("SL mong muốn (%):", min_value=0.0, max_value=20.0, value=7.0, step=0.5, help="Mức cắt lỗ bạn muốn áp dụng để so sánh với AI")

    run_btn = st.button('🚀 PHÂN TÍCH & SIÊU TỐI ƯU', use_container_width=True, on_click=trigger_analysis)

# === LOGIC XỬ LÝ ===

if st.session_state.get('run_analysis', False) and st.session_state.get('confirmed_ticker'):
    
    # Hack ẩn bàn phím mobile
    js_hack = f"""<script>function forceBlur(){{const activeElement=window.parent.document.activeElement;if(activeElement){{activeElement.blur();}}window.parent.document.body.focus();}}forceBlur();setTimeout(forceBlur,200);</script><div style="display:none;">{random.random()}</div>"""
    components.html(js_hack, height=0)

    ticker = st.session_state['confirmed_ticker']
    current_user_sl = stop_loss_input 

    if not ticker:
        st.warning("⚠️ Vui lòng nhập mã cổ phiếu!")
    else:
        symbol = ticker if ".VN" in ticker else f"{ticker}.VN"
        
        # --- BƯỚC 1: TẢI DỮ LIỆU ---
        if 'data' not in st.session_state or st.session_state.get('current_symbol') != symbol:
            with st.spinner(f'Đang tải dữ liệu {ticker}...'):
                try:
                    df_full = yf.download(symbol, period="max", interval="1d", progress=False)
                    if df_full.empty:
                        st.error(f"❌ Không tìm thấy mã **{ticker}**!")
                        st.stop()
                    
                    if isinstance(df_full.columns, pd.MultiIndex): df_full.columns = df_full.columns.get_level_values(0)
                    df_full['RSI'] = calculate_rsi(df_full['Close'], 14)
                    
                    st.session_state['data'] = df_full
                    st.session_state['current_symbol'] = symbol
                    
                    # Intraday
                    df_intra = yf.download(symbol, period="1d", interval="5m", progress=False)
                    if isinstance(df_intra.columns, pd.MultiIndex): df_intra.columns = df_intra.columns.get_level_values(0)
                    if not df_intra.empty:
                        if df_intra.index.tzinfo is None:
                            df_intra.index = df_intra.index + timedelta(hours=7)
                        else:
                            df_intra.index = df_intra.index.tz_convert('Asia/Ho_Chi_Minh')
                    st.session_state['data_intra'] = df_intra
                    
                except Exception as e:
                    st.error(f"Lỗi tải dữ liệu: {e}")
                    st.stop()
        
        # --- BƯỚC 2: TÍNH TOÁN CHIẾN THUẬT (QUÉT KÉP) ---
        if 'data' in st.session_state:
            df_calc = st.session_state['data']
            # Gọi hàm tối ưu kép
            best_res, results_df = optimize_ma_strategy_dual(df_calc, current_user_sl)
            
            if best_res is not None:
                st.session_state['best_ma'] = int(best_res['MA'])
                st.session_state['best_opt_sl'] = best_res['Opt SL'] # SL tốt nhất AI tìm ra
                st.session_state['best_opt_roi'] = best_res['Opt Annual ROI']
                
                # Thông tin so sánh (Của user)
                st.session_state['user_roi'] = best_res['User Annual ROI']
                
                st.session_state['top_mas'] = results_df.sort_values(by='Opt Annual ROI', ascending=False).head(5)
            else:
                st.error("Không đủ dữ liệu tính toán.")
                st.stop()

        # --- BƯỚC 3: HIỂN THỊ GIAO DIỆN ---
        try:
            df = st.session_state['data']
            df_intra = st.session_state.get('data_intra', pd.DataFrame())
            
            best_ma_val = st.session_state['best_ma']
            best_opt_sl_val = st.session_state['best_opt_sl']
            best_opt_roi_val = st.session_state['best_opt_roi']
            user_roi_val = st.session_state['user_roi']
            top_mas_df = st.session_state['top_mas']
            
            # Tính đường Best MA
            df['BestSMA'] = df['Close'].rolling(window=best_ma_val).mean()
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            curr_price = curr['Close']
            curr_ma = curr['BestSMA']
            curr_rsi = curr['RSI']
            
            # Logic Recommendation (Dựa trên Best MA + Best SL)
            rec = "QUAN SÁT (WAIT)"
            reason = "Chưa có tín hiệu."
            bg_class = "bg-blue"
            
            if curr_price < curr_ma and curr_rsi < 30:
                rec = "MUA NGAY"
                reason = f"Giá ({curr_price:,.0f}) < MA{best_ma_val} và RSI ({curr_rsi:.1f}) < 30."
                bg_class = "bg-green"
            elif curr_price > curr_ma and curr_rsi > 70:
                rec = "BÁN NGAY"
                reason = f"Giá ({curr_price:,.0f}) > MA{best_ma_val} và RSI ({curr_rsi:.1f}) > 70."
                bg_class = "bg-red"
            else:
                if curr_price > curr_ma: reason = f"Giá trên MA{best_ma_val} (Xu hướng tăng), chờ RSI > 70."
                else: reason = f"Giá dưới MA{best_ma_val} (Xu hướng giảm), chờ RSI < 30."

            st.markdown(f"<div class='result-card {bg_class}'><div class='result-title'>{rec}</div><div class='result-reason'>💡 {reason}</div></div>", unsafe_allow_html=True)
            
            # --- HIỂN THỊ SO SÁNH (AI vs USER) ---
            ai_color = "#00E676" if best_opt_roi_val > 0 else "#FF5252"
            user_color = "#00E676" if user_roi_val > 0 else "#FF5252"
            
            st.markdown(f"""
            <div class='backtest-box'>
                <div style='display:flex; justify-content:space-around; align-items:center;'>
                    <div style='text-align:center;'>
                         <div class='backtest-label'>🤖 AI TỐI ƯU (MA {best_ma_val})</div>
                        <div class='backtest-val' style='color:{ai_color}'>{best_opt_roi_val:+.1f}%<span style='font-size:1rem'>/năm</span></div>
                        <div class='backtest-sub'>SL Tối ưu: <b style='color:#FFF'>{best_opt_sl_val:.1f}%</b></div>
                    </div>
                    <div style='border-left:1px solid #546E7A; height:60px;'></div>
                    <div style='text-align:center;'>
                        <div class='backtest-label'>👤 CÀI ĐẶT CỦA BẠN</div>
                        <div class='backtest-val' style='color:{user_color}'>{user_roi_val:+.1f}%<span style='font-size:1rem'>/năm</span></div>
                        <div class='backtest-sub'>SL Bạn chọn: <b style='color:#FFF'>{current_user_sl}%</b></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # REPORT
            report = f"""
            <div class='report-box'>
                <div class='report-header'>📝 KẾT QUẢ TỐI ƯU HÓA KÉP</div>
                <div class='report-item'><span class='icon-dot'>🧠</span> <span>Hệ thống đã chạy thử nghiệm kết hợp các đường MA và mức Stoploss (0-10%, bước 0.5%).</span></div>
                <div class='report-item'><span class='icon-dot'>🏆</span> <span>Chiến lược tốt nhất: <b>MA {best_ma_val}</b> đi kèm mức cắt lỗ <b>{best_opt_sl_val:.1f}%</b>.</span></div>
                <div class='report-item'><span class='icon-dot'>⚖️</span> <span>So sánh: Nếu dùng SL {current_user_sl}% của bạn trên cùng đường MA này, hiệu quả là <b>{user_roi_val:.1f}%/năm</b>.</span></div>
            </div>
            """
            st.markdown(report, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # INTRADAY
            if not df_intra.empty:
                st.divider()
                latest_date = df_intra.index[0].strftime('%d/%m/%Y')
                st.markdown(f"### ⏱️ Diễn biến trong ngày ({latest_date})")
                ref_price = df['Close'].iloc[-2]
                current_price = df_intra['Close'].iloc[-1]
                line_color = '#00E676' if current_price >= ref_price else '#FF5252'
                fig_intra = go.Figure()
                fig_intra.add_trace(go.Scatter(x=df_intra.index, y=df_intra['Close'], mode='lines', line=dict(color=line_color, width=2), name='Intraday'))
                fig_intra.update_layout(height=350, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', tickformat="%H:%M"), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', autorange=True))
                st.plotly_chart(fig_intra, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

            # METRICS
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1: render_metric_card("GIÁ ĐÓNG CỬA", f"{curr['Close']:,.0f}", curr['Close'] - prev['Close'])
            with col_m2: render_metric_card("RSI (14)", f"{curr['RSI']:.1f}", curr['RSI'] - prev['RSI'])
            with col_m3: render_metric_card("MA TỐI ƯU", f"MA {best_ma_val}", curr['Close'] - curr['BestSMA'])
            with col_m4:
                status = "UPTREND" if curr_price > curr_ma else "DOWNTREND"
                color_st = "#00E676" if status == "UPTREND" else "#FF5252"
                render_metric_card("XU HƯỚNG", status, None, color=color_st)

            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()
            
            # --- CHART ---
            st.markdown(f"### 📊 Biểu đồ Kỹ Thuật & Top Hiệu Quả")
            time_tabs = st.radio("Khung thời gian:", ["1 Tháng", "3 Tháng", "6 Tháng", "1 Năm", "3 Năm", "Tất cả"], horizontal=True, index=3)
            
            df_chart = df.copy()
            if time_tabs == "1 Tháng": df_chart = df.iloc[-22:]
            elif time_tabs == "3 Tháng": df_chart = df.iloc[-66:]
            elif time_tabs == "6 Tháng": df_chart = df.iloc[-132:]
            elif time_tabs == "1 Năm": df_chart = df.iloc[-252:]
            elif time_tabs == "3 Năm": df_chart = df.iloc[-756:]

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BestSMA'], line=dict(color='#FF914D', width=2), name=f"MA {best_ma_val} (Tối ưu)"))
            fig1.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Giá"))
            fig1.update_layout(height=500, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=40), legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5), xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333'), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', autorange=True))
            st.plotly_chart(fig1, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("### 🚀 Chỉ số RSI")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#E040FB', width=2), name="RSI"))
                fig2.add_hline(y=70, line_dash="dot", line_color="#FF5252")
                fig2.add_hline(y=30, line_dash="dot", line_color="#00E676")
                fig2.update_layout(height=350, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=40), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333'), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', autorange=True))
                st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': False})

            with col_c2:
                st.markdown("### 🏆 Top 5 Combo Tốt Nhất")
                
                table_html = """<table class="ma-table"><thead><tr><th>Đường MA</th><th>SL Tối Ưu</th><th>Lãi AI/Năm</th><th>Lãi Của Bạn/Năm</th></tr></thead><tbody>"""
                
                for _, row in top_mas_df.iterrows():
                    ai_roi = row['Opt Annual ROI']
                    user_roi = row['User Annual ROI']
                    
                    c_ai = "#00E676" if ai_roi > 0 else "#FF5252"
                    c_user = "#00E676" if user_roi > 0 else "#FF5252"
                    
                    # Hiển thị SL với 1 số lẻ thập phân
                    row_html = f"""<tr>
                        <td class="highlight-val">MA {int(row['MA'])}</td>
                        <td style="color:#FFB74D; font-weight:bold">{row['Opt SL']:.1f}%</td>
                        <td style="color:{c_ai}; font-weight:bold">{ai_roi:.2f}%</td>
                        <td style="color:{c_user}">{user_roi:.2f}%</td>
                    </tr>"""
                    table_html += row_html
                
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Đã xảy ra lỗi hiển thị: {e}")
