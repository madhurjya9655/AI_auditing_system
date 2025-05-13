import fitz
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import pandas as pd
import io
import os
import json
from datetime import datetime
import openai
from dotenv import load_dotenv
import base64
from PIL import Image
import tempfile
import numpy as np
import time
import pytesseract               
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import cv2
from dateutil.parser import parse
import re
from difflib import SequenceMatcher


# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="AI-Powered PO vs Invoice Comparison Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  :root {
    --bg: #f0f4f8;
    --surface: #ffffff;
    --primary: #3b82f6;
    --secondary: #10b981;
    --danger: #ef4444;
    --text: #1f2937;
    --muted: #6b7280;
    --shadow: rgba(0,0,0,0.05);
  }
  body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg);
    color: var(--text);
  }
  .main-header {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), #7dd3fc);
    color: #fff;
    padding: 1rem;
    border-radius: 0.75rem;
    box-shadow: 0 4px 12px var(--shadow);
    text-align: center;
    margin-bottom: 2rem;
  }
  .sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--primary);
    position: relative;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }
  .sub-header::after {
    content: '';
    position: absolute;
    left: 0;
    bottom: 0;
    width: 3rem;
    height: 0.25rem;
    background: var(--primary);
    border-radius: 0.25rem;
  }
  .card {
    background-color: var(--surface);
    padding: 1.5rem;
    border-radius: 0.75rem;
    box-shadow: 0 8px 16px var(--shadow);
    margin-bottom: 1.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 32px var(--shadow);
  }
  .status-match,
  .status-mismatch {
    display: inline-block;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.9rem;
  }
  .status-match {
    background-color: var(--secondary);
    color: #fff;
  }
  .status-mismatch {
    background-color: var(--danger);
    color: #fff;
  }
  .instructions {
    background-color: #e0f2fe;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
    color: var(--text);
  }
  .highlight {
    background-color: #fde68a;
    padding: 0.2rem 0.4rem;
    border-radius: 0.25rem;
    font-weight: 600;
  }
  .comparison-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-top: 1rem;
  }
  .comparison-table th,
  .comparison-table td {
    padding: 0.75rem 1rem;
    text-align: left;
  }
  .comparison-table th {
    background-color: #e0f2fe;
    font-weight: 600;
  }
  .comparison-table tr:nth-child(even) {
    background-color: #f9fafb;
  }
  .stProgress > div > div > div > div {
    background-color: var(--primary) !important;
  }
</style>

""", unsafe_allow_html=True)



def extract_text_from_pdf(file):
    """Extract text and images from uploaded PDF file, with OCR for scanned documents"""
    file_bytes = file.read()
    file.seek(0)  # Reset file pointer for potential reuse

    # Extract text
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    images = []
    is_scanned = True  # Default assumption

    for page_num, page in enumerate(doc):
        page_text = page.get_text()

        # Check if the page contains meaningful text (non-scanned)
        if len(page_text.strip()) > 100:  # Arbitrary threshold for "meaningful" text
            full_text += page_text
            is_scanned = False
        else:
            # Use OCR for scanned page
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = preprocess_image_for_ocr(img)
            ocr_text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
            full_text += ocr_text

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append({
                "page": page_num,
                "bytes": image_bytes
            })

    return full_text, images, doc, is_scanned

def preprocess_image_for_ocr(img):
    img = img.convert("L")
    img_np = np.array(img)
    thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.medianBlur(thresh, 3)
    return Image.fromarray(denoised)


def extract_with_ai(text, doc_type, po_context=None):
    if doc_type == "invoice":
        fields = [
            "Buyer", "Invoice Number", "Date", "Customer Order Number",
            "Product Details", "HSN/SAC", "Quantity", "Rate",
            "Term of Delivery", "Destination", "Vehicle Number",
            "GSTIN", "Consignee GSTIN", "Payment Terms"
        ]
        po_summary = "; ".join(f"{k}: {po_context.get(k,'')}" for k in [
            "PO Number", "Buyer", "Supplier GSTIN", "Buyer GSTIN",
            "Destination", "Delivery Terms", "Payment Terms"
        ])
        prompt = f"""
You are an expert at reading semi-structured business documents.
Here are the matching PO fields: {po_summary}
Extract the following fields from this INVOICE document text and return ONLY a valid JSON object.

Required fields:
- Buyer
- Invoice Number
- Date
- Customer Order Number
- Product Details
- HSN/SAC
- Quantity
- Rate
- Term of Delivery
- Destination
- Vehicle Number
- GSTIN
- Consignee GSTIN
- Payment Terms

Document Text:
{text[:4000]}

Respond with a valid JSON object only, without explanation or markdown.
"""
    else:
        fields = [
            "PO Number", "Buyer", "Date", "Product Details",
            "HSN/SAC", "Quantity", "Rate",
            "Delivery Terms", "Destination",
            "GSTIN", "Consignee GSTIN", "Payment Terms"
        ]
        prompt = f"""
You are an expert at reading semi-structured business documents.
Extract the following fields from this PO document text and return ONLY a valid JSON object.

Required fields:
- PO Number
- Buyer
- Date
- Product Details
- HSN/SAC
- Quantity
- Rate
- Delivery Terms
- Destination
- GSTIN
- Consignee GSTIN
- Payment Terms

Document Text:
{text[:4000]}

Respond with a valid JSON object only, without explanation or markdown.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a document analysis assistant that extracts structured data from text. Return ONLY the JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").rstrip("```").strip()
        data = json.loads(result_text)
        for key in fields:
            if data.get(key) is None:
                data[key] = f"No {key} found"
        data["Supplier GSTIN"] = data.get("GSTIN", "No Supplier GSTIN found")
        data["Buyer GSTIN"] = data.get("Consignee GSTIN", "No Buyer GSTIN found")
        if doc_type == "invoice" and data.get("Customer Order Number", "").lower().startswith("no "):
            extracted_po = extract_po_reference_from_invoice(text)
            if extracted_po:
                data["Customer Order Number"] = extracted_po
        return data
    except Exception as e:
        st.error(f"Error extracting data with AI: {str(e)}")
        return fallback_extraction(text, doc_type)



def fallback_extraction(text, doc_type):
    """Fallback method using regex patterns if AI extraction fails"""
    if doc_type == "invoice":
        data = {
            'Buyer': safe_extract(text, r'(?:Buyer|BUYER|Customer|CUSTOMER|Billed\s*To|Invoice\s*To)[.\s:]+([^\n\r]+)', 'Buyer'),
            'Invoice Number': safe_extract(text, r'(?:Invoice\s*Number|INVOICE\s*NO)[.\s:]+([A-Za-z0-9-/]+)', 'Invoice Number'),
            'Date': safe_extract(text, r'(?:Date|DATE|Invoice\s*Dt)[.\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4})', 'Date'),
            'Customer Order Number': safe_extract(text, r'(?:Customer\s*Order\s*Number|Customer\s*PO|Order\s*No|Cust\.\s*PO\s*No|PO\s*No|Purchase\s*Order\s*No)[.\s:]+([A-Za-z0-9-/]+)', 'Customer Order Number'),
            'Product Details': safe_extract(text, r'(?:Product Details|Description|Item|MATERIAL\s*DESCRIPTION|Description\s*of\s*Goods)[.\s:]+([^\n\r]+)', 'Product Details'),
            'Term of Delivery': safe_extract(text, r'(?:Term of Delivery|Delivery Term|Shipping Term|Delivery\s*Terms)[.\s:]+([^\n\r]+)', 'Term of Delivery'),
            'Destination': safe_extract(text, r'(?:Destination|Ship To|Delivery Address|Shipped\s*To|Consignee)[.\s:]+([^\n\r]+)', 'Destination'),
            'Vehicle Number': safe_extract(text, r'(?:Vehicle\s*Number|VEHICLE\s*NO)[.\s:]+([A-Za-z0-9-]+)', 'Vehicle Number'),
            'HSN/SAC': safe_extract(text, r'(?:HSN/SAC|HSN|SAC|HSN\s*Code)[.\s:]+([A-Za-z0-9-]+)', 'HSN/SAC'),
            'Quantity': safe_extract(text, r'(?:Quantity|QTY)[.\s:]+(\d+(?:\.\d+)?)', 'Quantity'),
            'Rate': safe_extract(text, r'(?:Rate|Unit Price|Price|RATE)[.\s:]+(\d+(?:,\d+)*(?:\.\d+)?)', 'Rate'),
            'GSTIN': safe_extract(text, r'(?:GSTIN|GSTIN NO|Supplier\s*GSTIN)[.\s:]+([A-Za-z0-9]+)', 'GSTIN'),
            'Consignee GSTIN': safe_extract(text, r'(?:Consignee\s*GSTIN|Buyer\s*GSTIN)[.\s:]+([A-Za-z0-9]+)', 'Consignee GSTIN'),
            'Payment Terms': safe_extract(text, r'(?:Payment\s*Terms|Payment\s*Condition)[.\s:]+([^\n\r]+)', 'Payment Terms')
        }
        
        # Enhanced PO number extraction in invoice
        extracted_po = extract_po_reference_from_invoice(text)
        if extracted_po:
            data['Customer Order Number'] = extracted_po
        
        # Map to supplier/buyer GSTIN format
        data["Supplier GSTIN"] = data.get("GSTIN", "No Supplier GSTIN found")
        data["Buyer GSTIN"] = data.get("Consignee GSTIN", "No Buyer GSTIN found")
        
    else:  # PO
        data = {
            'PO Number': safe_extract(text, r'(?:PO[\s:.#]*No(?:\.|))[\s:.#]*([A-Za-z0-9/\-]+)', 'PO Number'),
            'Product Details': safe_extract(text, r'(?:Product Details|Description|Item|MATERIAL\s*DESCRIPTION|Description\s*of\s*Goods)[.\s:]+([^\n\r]+)', 'Product Details'),
            'Quantity': safe_extract(text, r'(?:Quantity|QTY)[.\s:]+(\d+(?:\.\d+)?)', 'Quantity'),
            'Rate': safe_extract(text, r'(?:Rate|Unit Price|Price|RATE)[.\s:]+(\d+(?:,\d+)*(?:\.\d+)?)', 'Rate'),
            'Customer Order Number': safe_extract(text, r'(?:Customer Order Number|Reference|Order Reference)[.\s:]+([A-Za-z0-9-/]+)', 'Customer Order Number'),
            'Delivery Terms': safe_extract(text, r'(?:Delivery Terms|Term of Delivery|Shipping)[.\s:]+([^\n\r]+)', 'Delivery Terms'),
            'HSN/SAC': safe_extract(text, r'(?:HSN/SAC|HSN|SAC|HSN\s*Code)[.\s:]+([A-Za-z0-9-]+)', 'HSN/SAC'),
            'Destination': safe_extract(text, r'(?:Destination|Ship To|Delivery Address|Shipped\s*To|Consignee)[.\s:]+([^\n\r]+)', 'Destination'),
            'Buyer': safe_extract(text, r'(?:Buyer|BUYER|Customer|Issued By|Billed\s*To)[.\s:]+([^\n\r]+)', 'Buyer'),
            'Date': safe_extract(text, r'(?:Date|DATE)[.\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4})', 'Date'),
            'GSTIN': safe_extract(text, r'(?:GSTIN|GSTIN NO|Supplier\s*GSTIN)[.\s:]+([A-Za-z0-9]+)', 'GSTIN'),
            'Consignee GSTIN': safe_extract(text, r'(?:Consignee\s*GSTIN|Buyer\s*GSTIN)[.\s:]+([A-Za-z0-9]+)', 'Consignee GSTIN'),
            'Payment Terms': safe_extract(text, r'(?:Payment\s*Terms|Payment\s*Condition)[.\s:]+([^\n\r]+)', 'Payment Terms')
        }
        
        # Map to supplier/buyer GSTIN format
        data["Supplier GSTIN"] = data.get("GSTIN", "No Supplier GSTIN found")
        data["Buyer GSTIN"] = data.get("Consignee GSTIN", "No Buyer GSTIN found")
    
    return data


def safe_extract(text, pattern, field_name):
    """Safely extract data using regex with error handling"""
    try:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return f"No {field_name} found"
    except Exception as e:
        return f"Error extracting {field_name}"


def normalize_data(data):
    normalized = {}
    for key, value in data.items():
        if isinstance(value, str):
            v = value.strip()

            # Normalize numeric fields
            if re.search(r'\d', v) and any(t in key.lower() for t in ['amount', 'quantity', 'rate', 'value', 'total']):
                nums = re.findall(r'[\d,]+\.?\d*', v)
                if nums:
                    try:
                        v = "{:.2f}".format(float(nums[0].replace(',', '')))
                    except:
                        pass

            # Normalize date fields
            elif 'date' in key.lower():
                try:
                    v = str(parse(v, fuzzy=True).date())
                except:
                    pass

            normalized[key] = v

        elif isinstance(value, (list, dict)):
            normalized[key] = json.dumps(value)

        else:
            normalized[key] = value

    return normalized

def match_po_invoice_items(po_items, invoice_items):
    """
    Match PO line items against invoice line items using multiple criteria.
    
    Parameters:
    - po_items (list): List of dictionaries containing PO line items
    - invoice_items (list): List of dictionaries containing invoice line items
    
    Returns:
    - dict: Dictionary containing matched items and matching results
    """
    import re
    from difflib import SequenceMatcher
    
    # If either list is empty, return early
    if not po_items or not invoice_items:
        return {
            "all_matched": False,
            "matches": [],
            "unmatched_po_items": po_items,
            "unmatched_invoice_items": invoice_items
        }
    
    matches = []
    unmatched_po_items = []
    matched_invoice_indices = set()
    
    def extract_steel_grade(text):
        """Extract steel grade specifications from text."""
        # Common steel grade patterns
        grade_patterns = [
            r'(?<!\w)(EN\s*\d{1,4})(?!\w)',  # European standard EN
            r'(?<!\w)(ASTM\s*[A-Z]\d{1,3})(?!\w)',  # ASTM standard
            r'(?<!\w)(AISI\s*\d{3,4}L?)(?!\w)',  # AISI standard
            r'(?<!\w)(SAE\s*\d{3,4}L?)(?!\w)',  # SAE standard
            r'(?<!\w)(SS\s*\d{3,4}L?)(?!\w)',  # Stainless Steel
            r'(?<!\w)(IS\s*\d{1,5})(?!\w)',  # Indian Standard
            r'(?<!\w)(DIN\s*\d{1,5})(?!\w)',  # German standard
            r'(?<!\w)(JIS\s*[A-Z]\d{1,4})(?!\w)',  # Japanese standard
            r'(?<!\w)(BS\s*\d{1,5})(?!\w)',  # British standard
            r'(?<!\w)(CK\s*\d{1,2})(?!\w)',  # Carbon tool steel
            r'(?<!\w)(C\d{1,2})(?!\w)',  # Carbon steel simplified
            r'(?<!\w)(ST\s*\d{1,2})(?!\w)'  # Structural steel
        ]
        
        grades = []
        for pattern in grade_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                grades.append(match.group(0).strip())
        
        return grades
    
    def extract_dimensions(text):
        """Extract dimensional information from text."""
        # Dimension patterns
        dim_patterns = [
            r'(?<!\w)(\d+(?:\.\d+)?\s*(?:mm|MM|cm|CM|m|M))(?!\w)',  # Metric measurements
            r'(?<!\w)(\d+(?:\.\d+)?\s*(?:inch|INCH|in|IN|"|\'))(?!\w)',  # Imperial measurements
            r'(?<!\w)(\d+(?:\.\d+)?\s*(?:DIA|dia|Dia))(?!\w)',  # Diameter
            r'(?<!\w)(L\s*\d+(?:\.\d+)?)(?!\w)',  # Length
            r'(?<!\w)(W\s*\d+(?:\.\d+)?)(?!\w)',  # Width
            r'(?<!\w)(H\s*\d+(?:\.\d+)?)(?!\w)',  # Height
            r'(?<!\w)(T\s*\d+(?:\.\d+)?)(?!\w)',  # Thickness
            r'(?<!\w)(\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?)(?!\w)',  # Dimensions like 10x20
            r'(?<!\w)(\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?)(?!\w)'  # 3D dimensions
        ]
        
        dimensions = []
        for pattern in dim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dimensions.append(match.group(0).strip())
        
        return dimensions
    
    def normalize_hsn_sac(code):
        """Normalize HSN/SAC code by removing non-alphanumeric characters."""
        return re.sub(r'\W+', '', str(code))
    
    def normalize_rate(rate):
        """Normalize rate by converting to float."""
        if not rate:
            return None
        try:
            return float(str(rate).replace(',', ''))
        except (ValueError, TypeError):
            return None
    
    def clean_description(text):
        """Clean description by removing common filler words and standardizing text."""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', str(text).lower()).strip()
        
        # Remove common filler words
        filler_words = [
            r'\bbar(s)?\b', r'\bround\b', r'\bother\b', r'\bmild steel\b', 
            r'\brounder\b', r'\bsteel\b', r'\brod(s)?\b', r'\bmetal\b',
            r'\bitem\b', r'\bproduct\b', r'\bmaterial\b'
        ]
        
        for word in filler_words:
            text = re.sub(word, '', text)
        
        # Normalize whitespace again after removing words
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_tokens(text):
        """Get meaningful tokens from text."""
        if not text:
            return set()
        return set(re.findall(r'\b[a-z0-9]+\b', clean_description(text)))
    
    def token_overlap_ratio(text1, text2):
        """Calculate token overlap ratio between two texts."""
        tokens1 = get_tokens(text1)
        tokens2 = get_tokens(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def get_similarity_score(po_item, inv_item):
        """Calculate overall similarity score between PO and invoice items."""
        score = 0.0
        reasons = []
        
        # Get descriptions
        po_desc = po_item.get('Description', '')
        inv_desc = inv_item.get('Description', '')
        
        # 1. Check steel grades
        po_grades = extract_steel_grade(po_desc)
        inv_grades = extract_steel_grade(inv_desc)
        
        if po_grades and inv_grades:
            for pg in po_grades:
                for ig in inv_grades:
                    if pg.lower() == ig.lower():
                        score += 3.0  # High weight for exact grade match
                        reasons.append(f"Grade match: {pg}")
                        break
        
        # 2. Check dimensions
        po_dims = extract_dimensions(po_desc)
        inv_dims = extract_dimensions(inv_desc)
        
        if po_dims and inv_dims:
            for pd in po_dims:
                for id in inv_dims:
                    if pd.lower() == id.lower():
                        score += 2.0  # Good weight for exact dimension match
                        reasons.append(f"Dimension match: {pd}")
                        break
        
        # 3. Check HSN/SAC codes
        po_hsn = normalize_hsn_sac(po_item.get('HSN/SAC', ''))
        inv_hsn = normalize_hsn_sac(inv_item.get('HSN/SAC', ''))
        
        if po_hsn and inv_hsn:
            if po_hsn == inv_hsn:
                score += 2.0  # Good weight for HSN match
                reasons.append(f"HSN match: {po_hsn}")
            elif len(po_hsn) >= 4 and len(inv_hsn) >= 4 and po_hsn[:4] == inv_hsn[:4]:
                score += 1.0  # Partial HSN match (first 4 digits)
                reasons.append(f"Partial HSN match: {po_hsn[:4]}")
        
        # 4. Check rates
        po_rate = normalize_rate(po_item.get('Rate', None))
        inv_rate = normalize_rate(inv_item.get('Rate', None))
        
        if po_rate is not None and inv_rate is not None:
            rate_diff = abs(po_rate - inv_rate) / max(po_rate, 1)
            if rate_diff <= 0.01:  # Within 1% tolerance
                score += 2.0  # Good weight for rate match
                reasons.append(f"Rate match: {po_rate:.2f} vs {inv_rate:.2f}")
        
        # 5. Check description token overlap
        overlap = token_overlap_ratio(po_desc, inv_desc)
        if overlap >= 0.4:  # At least 40% token overlap
            score += overlap * 3.0  # Scale by overlap amount
            reasons.append(f"Description overlap: {overlap:.2f}")

        
        
        # 6. Check for sequence similarity in descriptions
        if po_desc and inv_desc:
            seq_ratio = SequenceMatcher(None, clean_description(po_desc), 
                                        clean_description(inv_desc)).ratio()
            if seq_ratio >= 0.6:  # Good sequence similarity
                score += seq_ratio * 2.0
                reasons.append(f"Description similarity: {seq_ratio:.2f}")
        
        # 7. Check quantity if available
        po_qty = normalize_rate(po_item.get('Quantity', None))
        inv_qty = normalize_rate(inv_item.get('Quantity', None))
        
        if po_qty is not None and inv_qty is not None:
            if abs(po_qty - inv_qty) / max(po_qty, 1) <= 0.05:  # Within 5% tolerance
                score += 1.0
                reasons.append(f"Quantity match: {po_qty:.2f} vs {inv_qty:.2f}")
        
        return {
            "score": score,
            "reasons": reasons
        }
    
    # Process each PO item
    for po_index, po_item in enumerate(po_items):
        best_match = None
        best_score = 0
        best_reasons = []
        best_invoice_index = -1
        
        # Find best matching invoice item
        for inv_index, inv_item in enumerate(invoice_items):
            if inv_index in matched_invoice_indices:
                continue  # Skip already matched invoice items
                
            similarity = get_similarity_score(po_item, inv_item)
            if similarity["score"] > best_score:
                best_score = similarity["score"]
                best_match = inv_item
                best_reasons = similarity["reasons"]
                best_invoice_index = inv_index
        
        # Threshold for considering a match
        if best_score >= 3.0:  # Minimum score threshold
            matches.append({
                "po_item": po_item,
                "invoice_item": best_match,
                "match_score": best_score,
                "match_reasons": best_reasons
            })
            matched_invoice_indices.add(best_invoice_index)
        else:
            unmatched_po_items.append(po_item)
    
    # Collect unmatched invoice items
    unmatched_invoice_items = [item for i, item in enumerate(invoice_items) 
                             if i not in matched_invoice_indices]
    
    return {
        "all_matched": len(unmatched_po_items) == 0,
        "matches": matches,
        "unmatched_po_items": unmatched_po_items,
        "unmatched_invoice_items": unmatched_invoice_items
    }

# Helper function to be used in the compare_data function
def item_matches(po_item, invoice_items):
    """
    Check if a PO item matches any invoice item
    
    Parameters:
    - po_item (dict): Dictionary containing PO line item
    - invoice_items (list): List of dictionaries containing invoice line items
    
    Returns:
    - bool: True if the PO item matches any invoice item
    """
    match_result = match_po_invoice_items([po_item], invoice_items)
    return len(match_result["matches"]) > 0


def fuzzy_match_tokens(a, b):
    tokens_a = set(re.findall(r'\w+', a.lower()))
    tokens_b = set(re.findall(r'\w+', b.lower()))
    return len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1) >= 0.6

def similar_rate(a, b):
    try:
        a_val = float(a.replace(',', ''))
        b_val = float(b.replace(',', ''))
        return abs(a_val - b_val) / a_val <= 0.02
    except:
        return False


def match_po_number(po_number: str, invoice_text: str) -> bool:
    """
    Match PO number from PO document against invoice text.
    
    Strategy:
    1. Normalize both the PO number and invoice text (uppercase, remove non-alphanumeric except slashes)
    2. Extract prefix from PO number up to second slash (/)
    3. Search for prefix match in invoice text with various fallback patterns
    
    Args:
        po_number (str): The PO number from the PO document
        invoice_text (str): The full text of the invoice document
        
    Returns:
        bool: True if a match is found, False otherwise
    """
    if not po_number or not invoice_text:
        return False
    
    # Step 1: Normalize PO number
    clean_po = re.sub(r'[^A-Za-z0-9/]', '', po_number).upper()
    
    # Step 2: Extract prefix (up to second slash or full PO if less than 2 slashes)
    parts = clean_po.split('/')
    if len(parts) >= 2:
        prefix = '/'.join(parts[:2])
    else:
        prefix = clean_po
    
    # Add base case - if no valid prefix, return False
    if not prefix or len(prefix) < 3:  # Require at least 3 chars to avoid false positives
        return False
        
    # Step 3: Normalize invoice text
    clean_invoice = re.sub(r'[^A-Za-z0-9/]', '', invoice_text).upper()
    
    # Step 4: Direct match attempt
    if prefix in clean_invoice:
        return True
        
    # Step 5: Fallback - try with alternative delimiters that might be in original text
    # Common variations in invoice formats
    invoice_variations = [
        re.sub(r'\s+', '', invoice_text).upper(),  # Remove all whitespace
        re.sub(r'[^\w/]', '', invoice_text).upper(),  # Keep alphanumeric and slashes
        re.sub(r'[^A-Za-z0-9/-]', '', invoice_text).upper(),  # Keep alphanumeric, slashes and hyphens
    ]
    
    # Try variations of the prefix (with spaces, etc.)
    prefix_variations = [
        prefix,
        prefix.replace('/', ' / '),  # Add spaces around slashes
        prefix.replace('/', ''),     # Remove slashes entirely
        '-'.join(prefix.split('/')), # Replace slashes with hyphens
    ]
    
    # Try each combination
    for p_var in prefix_variations:
        if len(p_var) < 3:  # Skip very short variations
            continue
            
        # First check in cleaned invoice text
        if p_var in clean_invoice:
            return True
            
        # Then try each invoice variation
        for inv_var in invoice_variations:
            if p_var in inv_var:
                return True
    
    # Step 6: Pattern-based search for partial matches
    # For cases where the PO number format changes significantly
    parts_no_slash = [p for p in parts if p and len(p) >= 2]
    if len(parts_no_slash) >= 2:
        pattern = f"{parts_no_slash[0]}.*{parts_no_slash[1]}"
        if re.search(pattern, clean_invoice):
            return True
            
    # No match found with any method
    return False

def extract_po_reference_from_invoice(invoice_text):
    """
    Specifically extract PO reference from invoice text using common patterns.
    
    Args:
        invoice_text (str): The full text content of the invoice
        
    Returns:
        str: Extracted PO reference or empty string if not found
    """
    # Common patterns for PO references in invoices
    po_patterns = [
        r'PO\s*No\s*&\s*Date\s*[:#\s]*([A-Za-z0-9/\-]+)',

        # Various ways "Customer PO" might appear
        r'(?:Customer\s*(?:Order|PO|P\.O\.)(?:\s*No)?|Cust\.?\s*(?:Order|PO|P\.O\.)(?:\s*No)?|Purchase\s*Order(?:\s*No)?|Order\s*(?:No|Number)|PO\s*(?:No|Number)|Your\s*(?:Order|PO|P\.O\.)(?:\s*No)?|Buyer(?:\'s)?\s*(?:Order|PO|P\.O\.)(?:\s*No)?|Reference(?:\s*No)?)[\.:#\s]*([A-Za-z0-9][\w\s\-/]*?\d+[\w\s\-/]*)',
        
        # Look for any alphanumeric with slash pattern that might be a PO
        r'(?:ref|ref(?:erence)?|po|order)[\.:#\s]*([A-Za-z0-9][\w\-]*/\d+[\w\-/]*)',
        
        # Look for specific PO-like formats (letters-numbers/numbers pattern)
        r'([A-Za-z]{2,}[\-]?[A-Za-z\d]{0,4}[\-]?(?:PO|P\.O\.)?/\d+(?:/\d+)?(?:/\d+)?)'
    ]
    
    for pattern in po_patterns:
        matches = re.finditer(pattern, invoice_text, re.IGNORECASE)
        for match in matches:
            candidate = match.group(1).strip()
            # Additional validation to filter out dates, invoice numbers, etc.
            if validate_po_candidate(candidate):
                return candidate
                
    return ""

def validate_po_candidate(candidate):
    """
    Validate if a candidate string looks like a legitimate PO reference.
    
    Args:
        candidate (str): Potential PO reference string
        
    Returns:
        bool: True if it appears to be a valid PO reference
    """
    # Filter out common false positives
    if len(candidate) < 4:  # Too short to be a PO number
        return False
        
    # Filter out date-like patterns
    if re.match(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', candidate):
        return False
        
    # Filter out pure numbers (likely to be invoice numbers, quantities, etc.)
    if re.match(r'^\d+$', candidate):
        return False
        
    # Look for typical PO number patterns (mix of letters, numbers, and special chars)
    if re.search(r'[A-Za-z].*\d|\d.*[A-Za-z]', candidate):
        return True
        
    return False

def match_po_number(po_number, invoice_text):
    """
    Match PO number from PO document against invoice text.
    
    Args:
        po_number (str): The PO number from the PO document
        invoice_text (str): The full text of the invoice document
        
    Returns:
        bool: True if a match is found, False otherwise
    """
    if not po_number or not invoice_text:
        return False
        
    # Try to extract specific PO references from invoice first
    extracted_po_ref = extract_po_reference_from_invoice(invoice_text)
    
    # Clean both values
    clean_po = re.sub(r'[^A-Za-z0-9/\-]', '', po_number).upper()
    
    # If we have an extracted reference, prioritize direct matching with it
    if extracted_po_ref:
        clean_ref = re.sub(r'[^A-Za-z0-9/\-]', '', extracted_po_ref).upper()
        
        # Direct match check
        if clean_po == clean_ref:
            return True
            
        # Check if one contains the other
        if clean_po in clean_ref or clean_ref in clean_po:
            return True
            
        # Extract prefix from both (up to second slash)
        po_parts = clean_po.split('/')
        ref_parts = clean_ref.split('/')
        
        if len(po_parts) >= 2 and len(ref_parts) >= 2:
            po_prefix = '/'.join(po_parts[:2])
            ref_prefix = '/'.join(ref_parts[:2])
            
            if po_prefix == ref_prefix:
                return True
                
            # Check if partial match (first parts match)
            if po_parts[0] == ref_parts[0] and (
                (len(po_parts) > 1 and len(ref_parts) > 1 and po_parts[1] == ref_parts[1]) or
                (len(po_parts) > 1 and ref_parts[0].endswith(po_parts[1])) or
                (len(ref_parts) > 1 and po_parts[0].endswith(ref_parts[1]))
            ):
                return True
    
    # Fall back to full text search if targeted extraction failed
    # Extract prefix from PO number (up to second slash)
    parts = clean_po.split('/')
    if len(parts) >= 2:
        prefix = '/'.join(parts[:2])
    else:
        prefix = clean_po
        
    # Add base case
    if not prefix or len(prefix) < 3:
        return False
    
    # Clean invoice text for general search
    clean_invoice = re.sub(r'[^A-Za-z0-9/\-]', '', invoice_text).upper()
    
    # Direct match in full text
    if prefix in clean_invoice:
        return True
        
    # Try alternative formats
    prefix_variations = [
        prefix,
        prefix.replace('/', ' / '),
        prefix.replace('/', ''),
        prefix.replace('/', '-'),
        '-'.join(prefix.split('/'))
    ]
    
    for p_var in prefix_variations:
        if len(p_var) < 3:
            continue
            
        if p_var in clean_invoice:
            return True
    
    # Pattern-based search for complex cases
    parts_no_slash = [p for p in parts if p and len(p) >= 2]
    if len(parts_no_slash) >= 2:
        pattern = f"{parts_no_slash[0]}.*{parts_no_slash[1]}"
        if re.search(pattern, clean_invoice):
            return True
            
    return False

def compare_data(po_data, invoice_data):
    """
    Compare PO and invoice data to identify discrepancies,
    with a fallback that simply looks for any PO-derived token
    (e.g. EN codes or long words) in the invoice raw text.
    """
    import re
    from difflib import SequenceMatcher

    discrepancies = []
    comparison_results = {}

    # get the raw invoice text for simple substring searches
    raw_invoice = invoice_data.get('raw_text', '').lower()

    # normalize Product Details into lists of items
    po_items = po_data.get("Product Details", [])
    if not isinstance(po_items, list):
        po_items = [{
            "Description": po_data.get("Product Details", ""),
            "HSN/SAC":      po_data.get("HSN/SAC", ""),
            "Quantity":     po_data.get("Quantity", ""),
            "Rate":         po_data.get("Rate", "")
        }]
    invoice_items = invoice_data.get("Product Details", [])
    if not isinstance(invoice_items, list):
        invoice_items = [{
            "Description": invoice_data.get("Product Details", ""),
            "HSN/SAC":      invoice_data.get("HSN/SAC", ""),
            "Quantity":     invoice_data.get("Quantity", ""),
            "Rate":         invoice_data.get("Rate", "")
        }]

    # normalized scalar fields
    npo  = normalize_data(po_data)
    ninv = normalize_data(invoice_data)

    # the rest of your field-by-field mapping
    field_map = {
        'PO Number':      'Customer Order Number',
        'Buyer':          'Buyer',
        'Supplier GSTIN': 'GSTIN',
        'Buyer GSTIN':    'Consignee GSTIN',
        'Destination':    'Destination',
        'Delivery Terms': 'Term of Delivery',
        'Payment Terms':  'Payment Terms'
    }

    # ─── 1) Attempt your existing line‐item matcher ─────────────────────────────
    mr = match_po_invoice_items(po_items, invoice_items)
    product_match = mr["all_matched"] or len(mr["matches"]) > 0
    matching_details = {
        "matches_found":      len(mr["matches"]),
        "total_po_items":     len(po_items),
        "matched_items":      mr["matches"],
        "unmatched_po_items": mr["unmatched_po_items"]
    }
    hsn_match  = any("HSN match"  in r for m in mr["matches"] for r in m.get("match_reasons", []))
    rate_match = any("Rate match" in r for m in mr["matches"] for r in m.get("match_reasons", []))

    # ─── 2) FALLBACK: simple substring or token check ──────────────────────────
    if not product_match:
        # A) Look for any EN### code from the PO in the invoice raw text
        po_desc = " ".join(i.get("Description","") for i in po_items)
        grades = re.findall(r'\bEN\d+\b', po_desc, flags=re.IGNORECASE)
        if any(g.lower() in raw_invoice for g in grades):
            product_match = True
            matching_details["grade_fallback"] = grades
        else:
            # B) Token overlap: any long word (>=4 chars) in both?
            po_tokens  = {w for w in re.findall(r'\w+', po_desc.lower()) if len(w)>=4}
            inv_tokens = set(re.findall(r'\w+', raw_invoice))
            overlap   = po_tokens & inv_tokens
            if overlap:
                product_match = True
                # only keep up to 5 examples
                matching_details["token_fallback"] = list(overlap)[:5]

    comparison_results['Product Details'] = {
        "po_value":      format_product_details(po_items),
        "invoice_value": format_product_details(invoice_items),
        "match":         product_match,
        "match_details": matching_details
    }
    comparison_results['HSN/SAC'] = {
        "po_value":      extract_hsn_codes(po_items),
        "invoice_value": extract_hsn_codes(invoice_items),
        "match":         hsn_match
    }
    comparison_results['Rate'] = {
        "po_value":      extract_rates(po_items),
        "invoice_value": extract_rates(invoice_items),
        "match":         rate_match
    }

    # ─── 3) Other scalar fields ────────────────────────────────────────────────
    for pf, iv in field_map.items():
        pv  = npo.get(pf, "Not found")
        ivv = ninv.get(iv, "Not found")
        match = False

        if pf == 'PO Number':
            match = match_po_number(pv, raw_invoice)

        elif pf == 'Destination':
            # first try substring
            if ivv.strip().lower() in pv.strip().lower():
                match = True
            else:
                match = token_overlap(pv, ivv, threshold=0.4)

        elif pf == 'Buyer':
            match = token_overlap(pv, ivv, threshold=0.5) \
                    if not (pv.lower().startswith("no ") or ivv.lower().startswith("no ")) \
                    else False

        elif pf == 'Delivery Terms':
            kws = ['dispatch','ex','plant','from','delivery']
            if any(k in pv.lower() for k in kws) and any(k in ivv.lower() for k in kws):
                match = True
            else:
                match = token_overlap(pv, ivv, threshold=0.4)

        elif pf == 'Payment Terms':
            if "advance" in pv.lower() and "advance" in ivv.lower():
                match = True
            else:
                match = token_overlap(pv, ivv)

        elif 'GSTIN' in pf:
            match = re.sub(r'\W+','',pv).upper() == re.sub(r'\W+','',ivv).upper()

        else:
            match = token_overlap(pv, ivv)

        comparison_results[pf] = {"po_value": pv, "invoice_value": ivv, "match": match}
        if not match:
            discrepancies.append({
                "field": pf,
                "po_value": pv,
                "invoice_value": ivv
            })

    # ─── 4) Collect leftover mismatches ─────────────────────────────────────────
    if not product_match:
        discrepancies.append({
            "field": "Product Details",
            "po_value": comparison_results['Product Details']["po_value"],
            "invoice_value": comparison_results['Product Details']["invoice_value"]
        })
    if not hsn_match:
        discrepancies.append({
            "field": "HSN/SAC",
            "po_value": comparison_results['HSN/SAC']["po_value"],
            "invoice_value": comparison_results['HSN/SAC']["invoice_value"]
        })
    if not rate_match:
        discrepancies.append({
            "field": "Rate",
            "po_value": comparison_results['Rate']["po_value"],
            "invoice_value": comparison_results['Rate']["invoice_value"]
        })

    return discrepancies, comparison_results



def format_product_details(items):
    """Format product details for display"""
    if not items:
        return "Not found"
    
    if isinstance(items, str):
        return items
    
    formatted = []
    for item in items:
        desc = item.get("Description", "")
        qty = item.get("Quantity", "")
        if desc:
            if qty:
                formatted.append(f"{desc} (Qty: {qty})")
            else:
                formatted.append(desc)
    
    return "\n".join(formatted) if formatted else "Not found"


def extract_hsn_codes(items):
    """Extract HSN codes from items"""
    if not items:
        return "Not found"
    
    if isinstance(items, str):
        return items
    
    codes = []
    for item in items:
        code = item.get("HSN/SAC", "")
        if code:
            codes.append(code)
    
    return ", ".join(codes) if codes else "Not found"


def extract_rates(items):
    """Extract rates from items"""
    if not items:
        return "Not found"
    
    if isinstance(items, str):
        return items
    
    rates = []
    for item in items:
        rate = item.get("Rate", "")
        if rate:
            rates.append(str(rate))
    
    return ", ".join(rates) if rates else "Not found"


def token_overlap(a, b, threshold=0.4):
    """Calculate token overlap between two strings"""
    import re
    
    def clean_tokens(text):
        return set(re.findall(r'\w+', text.lower()))
    
    a_tokens = clean_tokens(a)
    b_tokens = clean_tokens(b)
    
    if not a_tokens or not b_tokens:
        return False
    
    intersection = a_tokens.intersection(b_tokens)
    union = a_tokens.union(b_tokens)
    
    return len(intersection) / len(union) >= threshold


def highlight_discrepancies_in_pdf(doc, comparison_results):
    """Add highlights to the PDF where discrepancies are found"""
    try:
        # Create a temporary file to store the highlighted PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            temp_path = tmp.name
        
        # Fields that we might find in the PDF
        fields_to_highlight = [field for field, result in comparison_results.items() 
                              if not result['match']]
        
        # Go through each page and highlight instances of these fields
        for page_num, page in enumerate(doc):
            for field in fields_to_highlight:
                # Try to find the values in the text
                po_value = comparison_results[field]['po_value']
                invoice_value = comparison_results[field]['invoice_value']
                
                if len(po_value) > 3 and po_value != "Not found":  # Only search for substantial values
                    instances = page.search_for(po_value)
                    for inst in instances:
                        page.add_highlight_annot(inst)
                
                if len(invoice_value) > 3 and invoice_value != "Not found":
                    instances = page.search_for(invoice_value)
                    for inst in instances:
                        page.add_highlight_annot(inst)
        
        # Save the document with highlights
        doc.save(temp_path)
        return temp_path
    
    except Exception as e:
        st.warning(f"Could not highlight discrepancies in PDF: {str(e)}")
        return None


def log_to_google_sheets(data):
    """Log comparison results to Google Sheets"""
    try:
        # Load credentials from environment variable or secrets
        creds_json = os.getenv("GOOGLE_SHEETS_CREDS")
        if not creds_json:
            creds_json = st.secrets.get("GOOGLE_SHEETS_CREDS", None)
        
        if creds_json:
            # Convert string to JSON if needed
            if isinstance(creds_json, str):
                creds_dict = json.loads(creds_json)
            else:
                creds_dict = creds_json
                
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
                json.dump(creds_dict, tmp)
                tmp_creds_file = tmp.name
            
            # Use the credentials
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            credentials = ServiceAccountCredentials.from_json_keyfile_name(tmp_creds_file, scope)
            client = gspread.authorize(credentials)
            
            # Open sheet by name from environment or create if it doesn't exist
            sheet_name = os.getenv("GOOGLE_SHEET_NAME", "PO_Invoice_Comparison")
            try:
                sheet = client.open(sheet_name).sheet1
            except:
                # Create a new sheet if it doesn't exist
                spreadsheet = client.create(sheet_name)
                # Define who it's shared with
                spreadsheet.share(os.getenv("SHARE_EMAIL", "example@example.com"), perm_type='user', role='writer')
                sheet = spreadsheet.sheet1
                # Add header row
                headers = [
                    "Timestamp", "PO Number", "Invoice Number", "Product Details",
                    "Quantity (PO vs Invoice)", "Rate (PO vs Invoice)", 
                    "Supplier GSTIN (PO vs Invoice)", "Buyer GSTIN (PO vs Invoice)", 
                    "Status", "Discrepancies"
                ]
                sheet.append_row(headers)
            
            # Prepare row data
            discrepancy_fields = ", ".join([d['field'] for d in data['discrepancies']]) if data['discrepancies'] else "None"
            
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data['po_data'].get('PO Number', 'Not found'),
                data['invoice_data'].get('Invoice Number', 'Not found'),
                data['po_data'].get('Product Details', 'Not found'),
                f"{data['po_data'].get('Quantity', 'Not found')} vs {data['invoice_data'].get('Quantity', 'Not found')}",
                f"{data['po_data'].get('Rate', 'Not found')} vs {data['invoice_data'].get('Rate', 'Not found')}",
                f"{data['po_data'].get('Supplier GSTIN', 'Not found')} vs {data['invoice_data'].get('Supplier GSTIN', 'Not found')}",
                f"{data['po_data'].get('Buyer GSTIN', 'Not found')} vs {data['invoice_data'].get('Buyer GSTIN', 'Not found')}",
                "Match" if not data['discrepancies'] else "Mismatch",
                discrepancy_fields
            ]
            
            # Append row to sheet
            sheet.append_row(row)
            
            # Clean up temporary file
            os.unlink(tmp_creds_file)
            
            return True
        else:
            st.warning("Google Sheets credentials not found in environment or secrets.")
            return False
    
    except Exception as e:
        st.error(f"Error logging to Google Sheets: {str(e)}")
        return False

def send_email(data):
    """Send email notification with comparison results"""
    try:
        # Get email credentials from environment or secrets
        sender_email = os.getenv("EMAIL_ADDRESS")
        if not sender_email:
            sender_email = st.secrets.get("EMAIL_ADDRESS", "notifications@example.com")
            
        password = os.getenv("EMAIL_PASSWORD")
        if not password:
            password = st.secrets.get("EMAIL_PASSWORD", "password")
            
        to_email = os.getenv("CLIENT_EMAIL")
        if not to_email:
            to_email = st.secrets.get("CLIENT_EMAIL", "client@example.com")
        
        # Create email content
        subject = "PO vs Invoice Comparison: "
        subject += "No Discrepancies Found" if not data['discrepancies'] else "Discrepancies Found"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Email body - HTML version
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .container {{ padding: 20px; }}
                .header {{ background-color: #f3f4f6; padding: 10px; }}
                .match {{ color: green; }}
                .mismatch {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>PO vs Invoice Comparison Report</h2>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <h3>Document Information</h3>
                <p><strong>PO Number:</strong> {data['po_data'].get('PO Number', 'Not found')}</p>
                <p><strong>Invoice Number:</strong> {data['invoice_data'].get('Invoice Number', 'Not found')}</p>
                <p><strong>Buyer:</strong> {data['po_data'].get('Buyer', 'Not found')}</p>
                
                <h3>Comparison Status</h3>
                <p><strong>Status:</strong> <span class="{'match' if not data['discrepancies'] else 'mismatch'}">
                {"All fields match" if not data['discrepancies'] else f"{len(data['discrepancies'])} discrepancies found"}
                </span></p>
        """
        
        # Add comparison table
        if data['comparison_results']:
            html_body += """
                <h3>Detailed Comparison</h3>
                <table>
                    <tr>
                        <th>Field</th>
                        <th>PO Value</th>
                        <th>Invoice Value</th>
                        <th>Status</th>
                    </tr>
            """
            
            for field, values in data['comparison_results'].items():
                status = "✓ Match" if values['match'] else "✗ Mismatch"
                status_class = "match" if values['match'] else "mismatch"
                
                html_body += f"""
                    <tr>
                        <td>{field}</td>
                        <td>{values['po_value']}</td>
                        <td>{values['invoice_value']}</td>
                        <td class="{status_class}">{status}</td>
                    </tr>
                """
            
            html_body += "</table>"
        
        # Close HTML
        html_body += """
            </div>
        </body>
        </html>
        """
        
        # Attach HTML version
        msg.attach(MIMEText(html_body, 'html'))
        
        # For demo purposes, return what would be sent
        email_info = {
            'subject': subject,
            'body': html_body,
            'to': to_email
        }
        
        # Check if we should actually send the email
        should_send = os.getenv("SEND_EMAILS", "False").lower() == "true"
        
        if should_send:
            # Send the email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, to_email, msg.as_string())
            return email_info, True
        else:
            return email_info, False
            
    except Exception as e:
        st.error(f"Error preparing email: {str(e)}")
        return None, False


def generate_report(data):
    """Generate a downloadable report of the comparison results"""
    report = io.StringIO()
    
    report.write("# PO vs Invoice Comparison Report\n\n")
    report.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report.write("## Document Information\n\n")
    report.write(f"- PO Number: {data['po_data'].get('PO Number', 'Not found')}\n")
    report.write(f"- Invoice Number: {data['invoice_data'].get('Invoice Number', 'Not found')}\n")
    report.write(f"- Buyer: {data['po_data'].get('Buyer', 'Not found')}\n\n")
    
    report.write("## Comparison Results\n\n")
    report.write("| Field | PO Value | Invoice Value | Status |\n")
    report.write("|-------|----------|--------------|--------|\n")
    
    for field, values in data['comparison_results'].items():
        status = "✅ Match" if values['match'] else "❌ Mismatch"
        report.write(f"| {field} | {values['po_value']} | {values['invoice_value']} | {status} |\n")
    
    if data['discrepancies']:
        report.write("\n## Discrepancies Found\n\n")
        for disc in data['discrepancies']:
            report.write(f"- **{disc['field']}**:\n")
            report.write(f"  - PO: {disc['po_value']}\n")
            report.write(f"  - Invoice: {disc['invoice_value']}\n\n")
    else:
        report.write("\n## No Discrepancies Found\n\n")
        report.write("All fields match between PO and Invoice.\n")
    
    return report


def convert_pdf_page_to_image(pdf_doc, page_num=0):
    """Convert a page of the PDF to an image for display"""
    try:
        page = pdf_doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        return img_bytes
    except Exception as e:
        st.warning(f"Could not convert PDF page to image: {str(e)}")
        return None
    
def display_match_details(matching_details):
    """
    Display a detailed breakdown of matching results for the Streamlit UI
    
    Parameters:
    - matching_details (dict): Output from match_po_invoice_items
    
    Returns:
    - None: Displays directly to Streamlit
    """
    import streamlit as st
    import pandas as pd
    
    if not matching_details:
        st.info("No matching details available")
        return
    
    st.markdown("### Line Item Matching Details")
    
    # Summary statistics
    total_po = matching_details.get("total_po_items", 0)
    matches_found = matching_details.get("matches_found", 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total PO Items", total_po)
    with col2:
        st.metric("Matched Items", matches_found)
    with col3:
        match_percent = (matches_found / total_po * 100) if total_po > 0 else 0
        st.metric("Match Rate", f"{match_percent:.1f}%")
    
    # Matched items
    if matching_details.get("matched_items"):
        st.markdown("#### ✅ Matched Items")
        for i, match in enumerate(matching_details["matched_items"]):
            with st.expander(f"Match #{i+1} (Score: {match['match_score']:.2f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**PO Item:**")
                    po_item = match["po_item"]
                    st.markdown(f"**Description:** {po_item.get('Description', 'N/A')}")
                    st.markdown(f"**HSN/SAC:** {po_item.get('HSN/SAC', 'N/A')}")
                    st.markdown(f"**Quantity:** {po_item.get('Quantity', 'N/A')}")
                    st.markdown(f"**Rate:** {po_item.get('Rate', 'N/A')}")
                
                with col2:
                    st.markdown("**Invoice Item:**")
                    inv_item = match["invoice_item"]
                    st.markdown(f"**Description:** {inv_item.get('Description', 'N/A')}")
                    st.markdown(f"**HSN/SAC:** {inv_item.get('HSN/SAC', 'N/A')}")
                    st.markdown(f"**Quantity:** {inv_item.get('Quantity', 'N/A')}")
                    st.markdown(f"**Rate:** {inv_item.get('Rate', 'N/A')}")
                
                st.markdown("**Match Evidence:**")
                for reason in match["match_reasons"]:
                    st.markdown(f"- {reason}")
    
    # Unmatched PO items
    if matching_details.get("unmatched_po_items"):
        st.markdown("#### ❌ Unmatched PO Items")
        for i, item in enumerate(matching_details["unmatched_po_items"]):
            with st.expander(f"Unmatched PO Item #{i+1}"):
                st.markdown(f"**Description:** {item.get('Description', 'N/A')}")
                st.markdown(f"**HSN/SAC:** {item.get('HSN/SAC', 'N/A')}")
                st.markdown(f"**Quantity:** {item.get('Quantity', 'N/A')}")
                st.markdown(f"**Rate:** {item.get('Rate', 'N/A')}")
                
    # No matches found
    if total_po > 0 and matches_found == 0:
        st.warning("No matches found between PO and Invoice items!")


def main():
    email_info = None

    if not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning("⚠️ OpenAI API key not found in environment variables. Some AI features may not work.")
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key
            st.sidebar.success("✅ API key set successfully!")
    else:
        st.sidebar.success("✅ OpenAI API key detected!")

    if 'results' not in st.session_state:
        st.session_state['results'] = None

        st.markdown('<h1 class="main-header">AI-Powered PO vs Invoice Comparison Tool</h1>', unsafe_allow_html=True)
    with st.expander("How to use this tool", expanded=False):
        st.markdown("""
1. **Upload PO**  
   • In the left card, click **Upload PO PDF** and select your Purchase Order file.  
2. **Preview PO** *(optional)*  
   • Expand **Preview PO Document** to verify the uploaded file.  
3. **Upload Invoice**  
   • In the right card, click **Upload Invoice PDF** and select your Invoice file.  
4. **Preview Invoice** *(optional)*  
   • Expand **Preview Invoice Document** to verify the upload.  
5. **Process Documents**  
   • Click **📊 Process Documents** to extract data, compare fields, and detect discrepancies.  
6. **View Results**  
   • Scroll down to see a summary status, detailed table, and line‐item matching gauge.  
7. **Download Outputs**  
   • Download highlighted PDFs or the markdown report via the buttons at the bottom.  
""", unsafe_allow_html=True)


    col1, col2 = st.columns(2)
    uploaded_po, uploaded_invoice = None, None
    po_doc, invoice_doc = None, None

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">📄 Purchase Order (PO)</h3>', unsafe_allow_html=True)
        uploaded_po = st.file_uploader("Upload PO PDF", type="pdf")
        if uploaded_po:
            st.success(f"Uploaded: {uploaded_po.name}")
            with st.expander("Preview PO Document", expanded=False):
                po_text, po_images, po_doc, po_is_scanned = extract_text_from_pdf(uploaded_po)
                img = convert_pdf_page_to_image(po_doc)
                if img:
                    st.image(img, caption="PO Document Preview", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">📝 Invoice</h3>', unsafe_allow_html=True)
        uploaded_invoice = st.file_uploader("Upload Invoice PDF", type="pdf")
        if uploaded_invoice:
            st.success(f"Uploaded: {uploaded_invoice.name}")
            with st.expander("Preview Invoice Document", expanded=False):
                invoice_text, invoice_images, invoice_doc, invoice_is_scanned = extract_text_from_pdf(uploaded_invoice)
                img = convert_pdf_page_to_image(invoice_doc)
                if img:
                    st.image(img, caption="Invoice Document Preview", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    process_button = st.button("📊 Process Documents", use_container_width=True, disabled=not (uploaded_po and uploaded_invoice))
    if process_button and uploaded_po and uploaded_invoice:
        with st.spinner('Processing documents...'):
            progress_bar = st.progress(0)
            progress_bar.progress(10)
            po_text, po_images, po_doc, po_is_scanned = extract_text_from_pdf(uploaded_po)
            progress_bar.progress(20)
            invoice_text, invoice_images, invoice_doc, invoice_is_scanned = extract_text_from_pdf(uploaded_invoice)
            progress_bar.progress(40)
            po_data = extract_with_ai(po_text, "po")
            progress_bar.progress(60)
            invoice_data = extract_with_ai(invoice_text, "invoice", po_data)
            invoice_data['raw_text'] = invoice_text
            progress_bar.progress(80)
            discrepancies, comparison_results = compare_data(po_data, invoice_data)
            st.session_state.results = {
                'po_data': po_data,
                'invoice_data': invoice_data,
                'discrepancies': discrepancies,
                'comparison_results': comparison_results,
                'po_doc': po_doc,
                'invoice_doc': invoice_doc,
                'po_is_scanned': po_is_scanned,
                'invoice_is_scanned': invoice_is_scanned
            }
            progress_bar.progress(100)

            if os.getenv("GOOGLE_SHEETS_CREDS") or st.secrets.get("GOOGLE_SHEETS_CREDS", None):
                with st.spinner("Logging results to Google Sheets..."):
                    if log_to_google_sheets(st.session_state.results):
                        st.success("Results logged to Google Sheets")

            with st.spinner("Preparing email notification..."):
                email_info, sent = send_email(st.session_state.results)
                if sent:
                    st.success("Email sent to client")
                elif email_info:
                    st.info("Email would be sent in production")

    if st.session_state.results:
        results = st.session_state.results
        status = "✅ All Fields Match" if not results['discrepancies'] else f"❌ {len(results['discrepancies'])} Discrepancies Found"
        status_color = "status-match" if not results['discrepancies'] else "status-mismatch"

        st.markdown(f'<h2 class="sub-header">Comparison Results</h2>', unsafe_allow_html=True)
        st.markdown(f'<h3 class="{status_color}">{status}</h3>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <p><strong>PO Document Type:</strong> {"Scanned (OCR)" if results['po_is_scanned'] else "Digital"}</p>
            <p><strong>Invoice Document Type:</strong> {"Scanned (OCR)" if results['invoice_is_scanned'] else "Digital"}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>Document Information</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <p><strong>PO Number:</strong> {results['po_data'].get('PO Number', 'Not found')}</p>
            <p><strong>Buyer:</strong> {results['po_data'].get('Buyer', 'Not found')}</p>
            <p><strong>PO Date:</strong> {results['po_data'].get('Date', 'Not found')}</p>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <p><strong>Invoice Number:</strong> {results['invoice_data'].get('Invoice Number', 'Not found')}</p>
            <p><strong>Invoice Date:</strong> {results['invoice_data'].get('Date', 'Not found')}</p>
            <p><strong>Product:</strong> {results['invoice_data'].get('Product Details', 'Not found')}</p>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>Detailed Comparison</h3>', unsafe_allow_html=True)
        comparison_data = []
        for field, values in results['comparison_results'].items():
            comparison_data.append({
                'Field': field,
                'PO Value': values['po_value'],
                'Invoice Value': values['invoice_value'],
                'Status': "✅ Match" if values['match'] else "❌ Mismatch"
            })
        df = pd.DataFrame(comparison_data)
        styled_df = df.style.applymap(
            lambda val: 'color: green; font-weight: bold' if val == "✅ Match" else 'color: red; font-weight: bold',
            subset=['Status']
        )
        st.dataframe(styled_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if "comparison_results" in results and "Product Details" in results["comparison_results"]:
            match_details = results["comparison_results"]["Product Details"].get("match_details", {})
            display_match_details(match_details)

            if match_details:
                total_po = match_details.get("total_po_items", 0)
                matches_found = match_details.get("matches_found", 0)
                if total_po > 0:
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=matches_found / total_po * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "PO-Invoice Item Match Rate", 'font': {'size': 24}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'steps': [
                                {'range': [0, 50], 'color': 'lightcoral'},
                                {'range': [50, 80], 'color': 'khaki'},
                                {'range': [80, 100], 'color': 'lightgreen'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'value': 90
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

        if results['discrepancies']:
            st.markdown('<div class="card"><h3 class="status-mismatch">Discrepancies Summary</h3>', unsafe_allow_html=True)
            for i, disc in enumerate(results['discrepancies']):
                st.markdown(f"""
                <p><strong>{i+1}. {disc['field']}</strong></p>
                <ul>
                    <li>PO Value: <span class="highlight">{disc['po_value']}</span></li>
                    <li>Invoice Value: <span class="highlight">{disc['invoice_value']}</span></li>
                </ul>
                """, unsafe_allow_html=True)

        try:
            col1, col2 = st.columns(2)
            with col1:
                file = highlight_discrepancies_in_pdf(results['po_doc'], results['comparison_results'])
                if file:
                    with open(file, "rb") as f:
                        st.download_button("Download PO with Highlights", f, "PO_highlighted.pdf", "application/pdf")
            with col2:
                file = highlight_discrepancies_in_pdf(results['invoice_doc'], results['comparison_results'])
                if file:
                    with open(file, "rb") as f:
                        st.download_button("Download Invoice with Highlights", f, "Invoice_highlighted.pdf", "application/pdf")
        except Exception:
            st.warning("Unable to generate highlighted PDFs.")

        report = generate_report(results)
        st.download_button(
            label="📥 Download Comparison Report",
            data=report.getvalue(),
            file_name=f"PO_Invoice_Comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )

        if email_info:
            with st.expander("View Email Notification Details"):
                st.markdown(f"**Subject:** {email_info['subject']}")
                st.markdown(f"**To:** {email_info['to']}")
                st.markdown("**Email Body:**")
                st.markdown(email_info['body'], unsafe_allow_html=True)

if __name__ == "__main__":
    main()
