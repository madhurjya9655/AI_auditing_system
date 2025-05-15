# app.py
import os
import json
import pdfplumber
import pandas as pd
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI

# ----- CONFIGURATION -----
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("Please set OPENAI_API_KEY in your .env")
    st.stop()
client = OpenAI(api_key=OPENAI_KEY)

# ----- UI SETUP -----
st.set_page_config(
    layout="wide", 
    page_title="AI PO–Invoice Audit", 
    page_icon="🔍"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        color: #2563EB;
        font-weight: 500;
        border-left: 4px solid #3B82F6;
        padding-left: 10px;
        margin-top: 2rem;
    }
    .upload-container {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .results-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    .compare-button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        padding: 0.6rem 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        border-radius: 8px;
        padding: 15px 25px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        width: 45%;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 8px 0;
    }
    .metric-label {
        color: #6B7280;
        font-size: 14px;
    }
    .df-header {
        background-color: #EFF6FF;
        font-weight: bold;
    }
    .match-row {
        background-color: #ECFDF5;
    }
    .mismatch-row {
        background-color: #FEF2F2;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        color: #6B7280;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----- FUNCTIONS -----
def extract_text(pdf_bytes: bytes) -> str:
    txt = ""
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            txt += t + "\n"
    return txt

def analyze_po_invoice(po_txt: str, inv_txt: str) -> list[dict]:
    prompt = f"""
You are an expert accounts-payable auditor. You will be given two documents: a Purchase Order (PO) and an Invoice.
Your task:

1) **Extract** exactly these nine fields from each:
   - voucher_no
   - description
   - rate
   - consignee
   - supplier_gstin
   - buyer_gstin
   - destination
   - terms_of_delivery
   - payment_terms

2) **Normalize**:
   • Uppercase and trim whitespace for all text.
   • For **rate**, parse numbers (strip commas/units) and treat as floats.
     Allow ±2% tolerance.
   • For **description**, PO may list items comma-separated.
     Invoice description may be free-form.
     Consider "Match" if **any** PO item (case-insensitive) appears in the invoice description.
   • For **voucher_no** and GSTIN codes, allow prefix or partial match (e.g. BOS-PO/322 matches BOS-PO/322/24-25).
   • For **destination** and **terms_of_delivery**, ignore punctuation and compare token overlap; require at least 80% token match.

3) **Compare** each PO field vs Invoice field and mark status:
   - "Match" or "Mismatch"

4) **Return** only a JSON array of objects with:
[
  {{
    "parameter":    "<field name>",
    "po_value":     "<extracted PO value>",
    "invoice_value":"<extracted Invoice value>",
    "status":       "Match" or "Mismatch"
  }},
  …
]

### PURCHASE ORDER TEXT ###
{po_txt}

### INVOICE TEXT ###
{inv_txt}
"""
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"You extract & compare PO vs Invoice fields per spec."},
            {"role":"user","content":prompt}
        ],
        temperature=0
    )
    content = resp.choices[0].message.content
    # isolate JSON array
    start = content.find("[")
    end   = content.rfind("]") + 1
    return json.loads(content[start:end])

# ----- MAIN UI -----
# Header
st.markdown("<h1 class='main-header'> AI-Powered PO vs Invoice Audit</h1>", unsafe_allow_html=True)

# Info box
st.info("""
This tool uses AI to automatically extract and compare key fields between Purchase Orders and Invoices.
Upload your documents below to identify discrepancies and ensure compliance.
""")

# File Upload Section
st.markdown("<h2 class='sub-header'>Document Upload</h2>", unsafe_allow_html=True)

st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Purchase Order")
    po_file = st.file_uploader("", type="pdf", key="po_upload", help="Upload your Purchase Order in PDF format")
    if po_file:
        st.success(f"✅ Uploaded: {po_file.name}")
with col2:
    st.markdown("### Invoice")
    inv_file = st.file_uploader("", type="pdf", key="inv_upload", help="Upload your Invoice in PDF format")
    if inv_file:
        st.success(f"✅ Uploaded: {inv_file.name}")
st.markdown("</div>", unsafe_allow_html=True)

# Action Buttons
if po_file and inv_file:
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        compare_button = st.button(" Compare Documents", type="primary", use_container_width=True)
    
    if compare_button:
        st.markdown("<div class='results-container'>", unsafe_allow_html=True)
        
        # Process files
        po_bytes = po_file.read()
        inv_bytes = inv_file.read()
        
        # Status indicators
        progress_bar = st.progress(0)
        
        with st.spinner(" Extracting text from PDFs..."):
            progress_bar.progress(25)
            po_txt = extract_text(po_bytes)
            inv_txt = extract_text(inv_bytes)
        
        with st.spinner(" Analyzing with GPT-4..."):
            progress_bar.progress(75)
            results = analyze_po_invoice(po_txt, inv_txt)
            progress_bar.progress(100)
        
        # Results header
        st.markdown("<h2 style='color:#1E3A8A; margin-top:20px;'>Analysis Results</h2>", unsafe_allow_html=True)
        
        # Create dataframe
        df = pd.DataFrame(results)
        df["Status"] = df["status"].map({"Match": "✅ Match", "Mismatch": "❌ Mismatch"})
        display_df = df.rename(columns={
            "parameter": "Parameter",
            "po_value": "PO Value",
            "invoice_value": "Invoice Value"
        })[["Parameter", "PO Value", "Invoice Value", "Status"]]
        
        # Style the dataframe
        def highlight_rows(row):
            return ['background-color: #ECFDF5' if row["Status"] == "✅ Match" else 'background-color: #FEF2F2'] * len(row)
        
        styled_df = display_df.style.apply(highlight_rows, axis=1)
        
        # Display the dataframe
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Metrics
        matches = df[df["status"] == "Match"].shape[0]
        mismatches = df[df["status"] == "Mismatch"].shape[0]
        
        st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Matches</div>
                    <div class='metric-value' style='color:#047857'>✅ {matches}</div>
                    <div>Fields that match between documents</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Mismatches</div>
                    <div class='metric-value' style='color:#DC2626'>❌ {mismatches}</div>
                    <div>Fields with discrepancies</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Summary message
        if mismatches > 0:
            st.warning(f"Found {mismatches} discrepancies between Purchase Order and Invoice. Please review highlighted fields.")
        else:
            st.success("✅ All fields match! The Purchase Order and Invoice are consistent.")
            
        # Export options
        st.markdown("<h3 class='sub-header'>Export Options</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name="po_invoice_comparison.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col2:
            st.button(" Export to Google Sheets", use_container_width=True, disabled=True, 
                      help="Feature coming soon")
                      
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Placeholder when no files are uploaded
    st.markdown(
        """
        <div style="background-color:#F9FAFB; padding:40px; border-radius:10px; text-align:center; margin-top:40px;">
            <img src="https://img.icons8.com/fluency/96/000000/upload-to-cloud.png" style="width:80px; height:80px;">
            <h3 style="margin-top:20px;">Ready to Analyze Documents</h3>
            <p style="color:#6B7280; margin-top:10px;">
                Please upload both Purchase Order and Invoice PDFs to begin the analysis
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Footer
st.markdown("<div class='footer'>Powered by OpenAI GPT-4 | Built with Streamlit</div>", unsafe_allow_html=True)
