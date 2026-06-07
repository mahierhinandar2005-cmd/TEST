import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Battery Assistant",
    page_icon="🤖",
    layout="wide"
)

# ==================== CUSTOM CSS (CHAT STYLE) ====================
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #0D0D0D;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Message bubbles */
    .bot-message {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 20px;
        animation: fadeIn 0.3s ease;
    }
    
    .bot-avatar {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #10B981, #06B6D4);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    
    .bot-bubble {
        background: #1E1E1E;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        color: #E0E0E0;
        font-size: 0.9rem;
        line-height: 1.5;
        max-width: 80%;
        border: 1px solid #2E2E2E;
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
        animation: fadeIn 0.3s ease;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #10B981, #06B6D4);
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        color: white;
        font-size: 0.9rem;
        max-width: 70%;
    }
    
    /* Input area */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #0D0D0D;
        padding: 1rem;
        border-top: 1px solid #2E2E2E;
    }
    
    .input-wrapper {
        max-width: 800px;
        margin: 0 auto;
        display: flex;
        gap: 10px;
    }
    
    .input-wrapper input {
        flex: 1;
        background: #1E1E1E;
        border: 1px solid #2E2E2E;
        border-radius: 24px;
        padding: 12px 20px;
        color: white;
        font-size: 0.9rem;
    }
    
    .input-wrapper input:focus {
        outline: none;
        border-color: #10B981;
    }
    
    .input-wrapper button {
        background: linear-gradient(135deg, #10B981, #06B6D4);
        border: none;
        border-radius: 24px;
        padding: 12px 24px;
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .input-wrapper button:hover {
        opacity: 0.8;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #1E1E1E, #2E2E2E);
        border-radius: 20px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid #3E3E3E;
    }
    
    .soh-value {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10B981, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
    }
    
    .healthy {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
    }
    
    .warning {
        background: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
    }
    
    .critical {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
    }
    
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.7rem;
        padding: 20px;
        margin-top: 80px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    model = joblib.load('model_soh.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model()

# ==================== INIT SESSION STATE ====================
if "step" not in st.session_state:
    st.session_state.step = 0  # 0=input, 1=predicting, 2=result
    st.session_state.data = {}
    st.session_state.messages = [
        {"role": "bot", "content": "👋 Halo! Aku **Battery Assistant** 🤖\n\nAyo cek kesehatan baterai mobil listrikmu!\n\nMulai dengan memasukkan **Aging Cycle** (jumlah siklus charge-discharge, 0-2000):"}
    ]
    st.session_state.current_input = "cycle"
    st.session_state.prediction_result = None

# ==================== PROCESS INPUT ====================
def process_input(user_input):
    current = st.session_state.current_input
    
    if current == "cycle":
        try:
            cycle = float(user_input)
            if 0 <= cycle <= 2000:
                st.session_state.data["cycle"] = cycle
                st.session_state.current_input = "soc"
                return "✅ Aging cycle disimpan. Sekarang masukkan **SOC (%)** (State of Charge, 0-100):"
            else:
                return "⚠️ Aging cycle harus antara 0-2000. Coba lagi:"
        except:
            return "⚠️ Masukkan angka yang valid untuk Aging cycle (0-2000):"
    
    elif current == "soc":
        try:
            soc = float(user_input)
            if 0 <= soc <= 100:
                st.session_state.data["soc"] = soc
                st.session_state.current_input = "rint"
                
                if soc < 20:
                    tip = "⚠️ SOC terlalu rendah! Hindari deep discharge."
                elif soc > 80:
                    tip = "⚠️ SOC terlalu tinggi! Hindari charge penuh terus-menerus."
                else:
                    tip = "✅ SOC dalam rentang optimal (20-80%)!"
                
                return f"{tip}\n\nSekarang masukkan **R_int (%)** (Internal Resistance, normal 100%):"
            else:
                return "⚠️ SOC harus antara 0-100. Coba lagi:"
        except:
            return "⚠️ Masukkan angka yang valid untuk SOC (0-100):"
    
    elif current == "rint":
        try:
            rint = float(user_input)
            if 0 <= rint <= 200:
                st.session_state.data["rint"] = rint
                st.session_state.current_input = "ocv"
                
                if rint > 150:
                    tip = "⚠️ R_int tinggi! Indikasi degradasi baterai."
                elif rint > 120:
                    tip = "⚠️ R_int mulai naik, perlu diwaspadai."
                else:
                    tip = "✅ R_int normal."
                
                return f"{tip}\n\nTerakhir, masukkan **OCV (V)** (Open Circuit Voltage, normal 3.8-4.2V):"
            else:
                return "⚠️ R_int harus antara 0-200. Coba lagi:"
        except:
            return "⚠️ Masukkan angka yang valid untuk R_int (0-200):"
    
    elif current == "ocv":
        try:
            ocv = float(user_input)
            if 3.0 <= ocv <= 4.5:
                st.session_state.data["ocv"] = ocv
                st.session_state.current_input = "done"
                
                if ocv < 3.6:
                    tip = "⚠️ OCV rendah! Tegangan baterai turun."
                elif ocv < 3.8:
                    tip = "⚠️ OCV mulai turun, perlu diwaspadai."
                else:
                    tip = "✅ OCV normal."
                
                return f"{tip}\n\n⏳ Sedang menghitung prediksi SOH... Tunggu sebentar ya!"
            else:
                return "⚠️ OCV harus antara 3.0-4.5V. Coba lagi:"
        except:
            return "⚠️ Masukkan angka yang valid untuk OCV (3.0-4.5V):"
    
    return "Terima kasih!"

# ==================== UI HEADER ====================
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <span style="font-size: 2rem;">🤖</span>
    <h1 style="color: white; margin: 0;">Battery Assistant</h1>
    <p style="color: #888;">Chat-based Battery Health Predictor</p>
</div>
""", unsafe_allow_html=True)

# ==================== CHAT DISPLAY ====================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "bot":
        st.markdown(f"""
        <div class="bot-message">
            <div class="bot-avatar">🤖</div>
            <div class="bot-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="user-message">
            <div class="user-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== INPUT FORM ====================
st.markdown('</div>', unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input("", placeholder="Ketik jawabanmu di sini...", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("📤 Kirim")

if submitted and user_input:
    # Tambah pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Proses input
    bot_response = process_input(user_input)
    
    # Tambah pesan bot
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    
    # Kalo udah selesai semua input, lakukan prediksi
    if st.session_state.current_input == "done":
        # Hitung prediksi
        input_data = np.array([[
            st.session_state.data["cycle"],
            st.session_state.data["soc"],
            st.session_state.data["rint"],
            st.session_state.data["ocv"]
        ]])
        input_scaled = scaler_X.transform(input_data)
        pred_scaled = model.predict(input_scaled)
        soh = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        
        # Tentukan status
        if soh >= 90:
            status = "SEHAT"
            status_color = "healthy"
            status_text = "✅ Baterai dalam kondisi sangat baik!"
            rekomendasi = "Lanjutkan penggunaan normal. Service rutin setiap 6 bulan."
        elif soh >= 70:
            status = "WASPADA"
            status_color = "warning"
            status_text = "⚠️ Baterai mulai menunjukkan degradasi!"
            rekomendasi = "Segera lakukan inspeksi dan balancing cell."
        else:
            status = "KRITIS"
            status_color = "critical"
            status_text = "🔴 Kesehatan baterai kritis!"
            rekomendasi = "Ganti baterai segera untuk keselamatan dan performa."
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=soh,
            title={"text": "SOH Meter", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": "#10B981"},
                "steps": [
                    {"range": [0, 70], "color": "#7F1A1A"},
                    {"range": [70, 90], "color": "#854D0E"},
                    {"range": [90, 100], "color": "#14532D"}
                ]
            }
        ))
        fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
        
        # Tampilkan hasil prediksi
        result_html = f"""
        <div class="result-card">
            <div class="soh-value">{soh:.1f}%</div>
            <div class="status-badge {status_color}">{status}</div>
            <p style="margin: 15px 0 0 0; color: #ccc;">{status_text}</p>
            <hr style="border-color: #3E3E3E; margin: 15px 0;">
            <p style="margin: 0; color: #888; font-size: 0.85rem;">💡 {rekomendasi}</p>
        </div>
        """
        
        st.session_state.messages.append({"role": "bot", "content": result_html})
        st.session_state.messages.append({"role": "bot", "content": "🔄 Mau cek baterai lain? Klik **Refresh** di sidebar atau reload halaman ini."})
        
        # Tambahkan gauge chart ke session state (agak tricky, tapi kita render di bawah)
        st.session_state.prediction_result = fig
        
        # Reset untuk prediksi baru (opsional)
        # st.session_state.current_input = "cycle"
    
    st.rerun()

# ==================== RENDER GAUGE CHART JIKA ADA ====================
if st.session_state.prediction_result:
    st.plotly_chart(st.session_state.prediction_result, use_container_width=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🤖 Battery Assistant")
    st.markdown("---")
    st.markdown("### 📋 Tentang")
    st.info("""
    **Metode:** ANN (MLPRegressor)
    
    **Dataset:** CNR Italy EIS (8 Cells, 2026)
    
    **Parameter:**
    - Aging Cycle (0-2000)
    - SOC (%) (0-100)
    - R_int (%) (0-200)
    - OCV (V) (3.0-4.5)
    
    **Output:** SOH (%) + Rekomendasi
    """)
    
    st.markdown("---")
    if st.button("🔄 Mulai Baru", use_container_width=True):
        for key in ["step", "data", "messages", "current_input", "prediction_result"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    st.caption("© 2026 | Project SC 2026")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    🔋 Battery Assistant — Prediksi SOH Baterai dengan ANN
</div>
""", unsafe_allow_html=True)
