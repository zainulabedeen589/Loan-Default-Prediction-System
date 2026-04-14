import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD ARTIFACTS
# =========================
MODEL_PATH = "artifact/models/trained_model.pkl"
ENCODER_PATH = "artifact/models/encoding_obj.pkl"

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Loan Risk System", layout="wide")

# =========================
# PREMIUM CSS
# =========================
st.markdown("""
<style>

/* Animated Background */
.stApp {
    background: linear-gradient(-45deg, #141e30, #243b55, #0f2027, #2c5364);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
}
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Cards */
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

/* Title */
.title {
    text-align:center;
    font-size:48px;
    font-weight:700;
    color:white;
}

/* Button */
.stButton>button {
    width:100%;
    padding:12px;
    border-radius:12px;
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color:white;
    font-weight:600;
    border:none;
    transition:0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Metric Card */
.metric {
    background: rgba(255,255,255,0.1);
    padding:20px;
    border-radius:15px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div class='title'>💳 AI Loan Risk System</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
with st.form("form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("👤 Personal Info")
        Age = st.number_input("Age", value=30)
        Income = st.number_input("Income", value=50000)
        CreditScore = st.number_input("Credit Score", value=650)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("💰 Loan Details")
        LoanAmount = st.number_input("Loan Amount", value=100000)
        InterestRate = st.number_input("Interest Rate (%)", value=10.0)
        LoanTerm = st.number_input("Loan Term (months)", value=60)
        DTIRatio = st.number_input("DTI Ratio", value=0.3)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Other Info")
        MonthsEmployed = st.number_input("Months Employed", value=60)
        NumCreditLines = st.number_input("Credit Lines", value=5)

        Education = st.selectbox("Education", encoders['Education'].classes_, index=0)
        EmploymentType = st.selectbox("Employment Type", encoders['EmploymentType'].classes_, index=0)
        MaritalStatus = st.selectbox("Marital Status", encoders['MaritalStatus'].classes_, index=0)
        LoanPurpose = st.selectbox("Loan Purpose", encoders['LoanPurpose'].classes_, index=0)

        HasMortgage = st.selectbox("Has Mortgage", encoders['HasMortgage'].classes_, index=0)
        HasDependents = st.selectbox("Has Dependents", encoders['HasDependents'].classes_, index=0)
        HasCoSigner = st.selectbox("Has Co-Signer", encoders['HasCoSigner'].classes_, index=0)
        st.markdown("</div>", unsafe_allow_html=True)

    submit = st.form_submit_button("🚀 Predict Risk")

# =========================
# PREPROCESS
# =========================
def preprocess():
    df = pd.DataFrame([{
        'Age': Age,
        'Income': Income,
        'LoanAmount': LoanAmount,
        'CreditScore': CreditScore,
        'MonthsEmployed': MonthsEmployed,
        'NumCreditLines': NumCreditLines,
        'InterestRate': InterestRate,
        'LoanTerm': LoanTerm,
        'DTIRatio': DTIRatio,
        'Education': Education,
        'EmploymentType': EmploymentType,
        'MaritalStatus': MaritalStatus,
        'HasMortgage': HasMortgage,
        'HasDependents': HasDependents,
        'LoanPurpose': LoanPurpose,
        'HasCoSigner': HasCoSigner
    }])

    # AGE BIN
    df['Age_Group'] = pd.cut(df['Age'],
                            bins=[0, 18, 30, 45, 60, 100],
                            labels=['Child', 'Teenager', 'Adult', 'Senior', "Super Senior"])

    # ENCODING
    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    # FEATURE ENGINEERING
    total_payback = df['LoanAmount'] * (1 + (df['InterestRate'] / 100))
    df['Monthly_Payment'] = total_payback / df['LoanTerm']
    df['PTI_Ratio'] = df['Monthly_Payment'] / (df['Income'] / 12)
    df['Job_Stability_Index'] = df['MonthsEmployed'] / (df['Age'] * 12)
    df['Debt_per_Line'] = df['LoanAmount'] / (df['NumCreditLines'] + 1)
    df['Young_High_Risk'] = ((df['Age'] < 30) & (df['LoanAmount'] > df['LoanAmount'].median())).astype(int)

    df = df.drop(columns=['Age'])

    return df

# =========================
# PREDICTION OUTPUT
# =========================
if submit:
    try:
        df = preprocess()
        df = df[model.feature_names_in_]

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='metric'>", unsafe_allow_html=True)
            st.subheader("📊 Risk Probability")
            st.markdown(f"## {prob:.2%}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='metric'>", unsafe_allow_html=True)
            if pred == 1:
                st.subheader("⚠️ HIGH RISK")
            else:
                st.subheader("✅ LOW RISK")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")