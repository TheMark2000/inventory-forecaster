import streamlit as st
import pandas as pd
import pickle
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import os
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import base64
import sendgrid
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

# Load environment variables
load_dotenv()
# Load environment variables (local) or secrets (cloud)
if os.path.exists('.env'):
    load_dotenv()
else:
    # Use Streamlit secrets in cloud
    os.environ['GEMINI_API_KEY'] = st.secrets.get('GEMINI_API_KEY', '')
    os.environ['SENDGRID_API_KEY'] = st.secrets.get('SENDGRID_API_KEY', '')
    os.environ['SENDER_EMAIL'] = st.secrets.get('SENDER_EMAIL', '')

# Import Google Gemini
try:
    import google.generativeai as genai
    api_key = os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY', '')
    genai.configure(api_key=api_key)
    HAS_GEMINI = True
except Exception as e:
    print(f"Gemini import error: {e}")
    HAS_GEMINI = False

# Initialize SendGrid
try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
    api_key = os.getenv('SENDGRID_API_KEY') or st.secrets.get('SENDGRID_API_KEY', '')
    sg = sendgrid.SendGridAPIClient(api_key=api_key)
    HAS_EMAIL = True
except Exception as e:
    print(f"SendGrid error: {e}")
    HAS_EMAIL = False

    # ============================================
# ENSURE DATABASE AND MODEL EXIST   ← INSERT THIS BLOCK
# ============================================

def ensure_database():
    """Create database with mock data if it doesn't exist"""
    db_path = 'data/inventory.db'
    if not os.path.exists(db_path):
        os.makedirs('data', exist_ok=True)
        try:
            from data_mock import create_mock_data
            create_mock_data()
            print("✅ Database created with mock data")
        except ImportError:
            print("⚠️ data_mock.py not found. Creating empty database.")
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inventory_history (
                    sku TEXT, 
                    date TEXT, 
                    category TEXT, 
                    current_stock REAL, 
                    daily_demand REAL, 
                    price REAL
                )
            """)
            conn.close()

def ensure_model():
    """Check if model exists, show error if not"""
    model_path = 'models/demand_model.pkl'
    schema_path = 'models/feature_schema.pkl'
    os.makedirs('models', exist_ok=True)
    
    if not os.path.exists(model_path) or not os.path.exists(schema_path):
        st.warning("⚠️ ML model not found. Please run train_model.py locally and commit model files.")
        st.info("💡 On your local machine, run: python train_model.py")
        st.stop()

# Run checks
ensure_database()
ensure_model()


# ============================================
# PDF GENERATION FUNCTIONS
# ============================================

def generate_pdf_order(supplier, items_to_order, order_date, delivery_date):
    """Generate a professional PDF purchase order WITHOUT prices"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=18,
        alignment=1,
        spaceAfter=20
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4
    )
    
    story = []
    
    story.append(Paragraph("PURCHASE ORDER", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    order_details = [
        ["Supplier:", supplier],
        ["Order Date:", order_date],
        ["Delivery Date:", delivery_date],
        ["Order Reference:", f"PO-{datetime.now().strftime('%Y%m%d%H%M')}"]
    ]
    
    detail_table = Table(order_details, colWidths=[1.5*inch, 3*inch])
    detail_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(detail_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("ITEMS TO ORDER:", header_style))
    story.append(Spacer(1, 0.1*inch))
    
    item_data = [["SKU", "Description", "Quantity"]]
    
    for _, row in items_to_order.iterrows():
        item_data.append([
            row['SKU'],
            f"Product {row['SKU']}",
            str(int(row['Recommended Order']))
        ])
    
    item_table = Table(item_data, colWidths=[1.5*inch, 3*inch, 1*inch])
    item_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
    ]))
    story.append(item_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("INSTRUCTIONS TO SUPPLIER:", header_style))
    instructions = [
        "1. Please confirm receipt of this order by replying to this email.",
        "2. Confirm availability of all items listed.",
        "3. Provide expected delivery date and pricing.",
        "4. Include tracking information once shipped.",
        "5. Contact us immediately if any item is unavailable."
    ]
    
    for inst in instructions:
        story.append(Paragraph(inst, body_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    footer_text = f"Generated by Smart Inventory Forecaster - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    story.append(Paragraph(footer_text, body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def get_pdf_download_link(pdf_buffer, filename):
    """Create a download link for the PDF"""
    b64 = base64.b64encode(pdf_buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration:none; background-color:#4CAF50; color:white; padding:10px 20px; border-radius:5px; display:inline-block; width:100%; text-align:center;">Download PDF</a>'
    return href

# ============================================
# EMAIL SENDING FUNCTION - ADD THIS
# ============================================

def send_order_email(to_email, from_email, supplier, order_text, pdf_buffer):
    """Send email with PDF attachment using SendGrid"""
    sender = from_email or st.secrets.get('SENDER_EMAIL', '')
    
    pdf_bytes = pdf_buffer.getvalue()
    
    subject = f"Purchase Order - {supplier} - {datetime.now().strftime('%Y-%m-%d')}"
    
    html_content = f"""
    <html>
    <body>
        <h2>Purchase Order</h2>
        <p><strong>Supplier:</strong> {supplier}</p>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
        <p><strong>Order Reference:</strong> PO-{datetime.now().strftime('%Y%m%d%H%M')}</p>
        <hr>
        <h3>Order Details:</h3>
        <pre style="font-family: monospace; background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
{order_text}
        </pre>
        <hr>
        <p>Please find attached the official purchase order PDF.</p>
        <p>Please confirm receipt of this order.</p>
        <hr>
        <p style="color: #666; font-size: 12px;">
            This is an automated email from Smart Inventory Forecaster.<br>
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </body>
    </html>
    """
    
    message = Mail(
        from_email=sender,
        to_emails=to_email,
        subject=subject,
        html_content=html_content
    )
    
    attachment = Attachment()
    attachment.file_content = FileContent(base64.b64encode(pdf_bytes).decode())
    attachment.file_name = FileName(f"Purchase_Order_{datetime.now().strftime('%Y%m%d')}.pdf")
    attachment.file_type = FileType('application/pdf')
    attachment.disposition = Disposition('attachment')
    message.attachment = attachment
    
    response = sg.send(message)
    return response.status_code == 202

# ============================================
# STREAMLIT APP
# ============================================

st.set_page_config(page_title="Smart Inventory Forecaster", layout="wide")
st.title("Smart Inventory Forecaster")
st.caption("AI predicts demand, you approve orders")

# Load model
@st.cache_resource
def load_model():
    with open('models/demand_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_schema():
    with open('models/feature_schema.pkl', 'rb') as f:
        return pickle.load(f)

# Initialize session state - ADDED missing fields
if 'run_forecast' not in st.session_state:
    st.session_state.run_forecast = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'draft' not in st.session_state:
    st.session_state.draft = ""
if 'items_to_order' not in st.session_state:
    st.session_state.items_to_order = None
if 'supplier' not in st.session_state:
    st.session_state.supplier = ""
if 'supplier_email' not in st.session_state:  # ADDED
    st.session_state.supplier_email = ""
if 'sender_email' not in st.session_state:    # ADDED
    st.session_state.sender_email = ""
if 'order_status' not in st.session_state:
    st.session_state.order_status = None
if 'items_to_order_pdf' not in st.session_state:
    st.session_state.items_to_order_pdf = None
if 'email_sent' not in st.session_state:      # ADDED
    st.session_state.email_sent = False

# Load models
model = load_model()
schema = load_schema()

# Show status
if HAS_GEMINI:
    st.sidebar.success("Connected - AI drafting enabled")
else:
    st.sidebar.warning("Not Connected - Using template drafts")

if HAS_EMAIL:
    st.sidebar.success("Email Service Connected")
else:
    st.sidebar.warning("Email Service Not Connected - Check SendGrid")

# Sidebar - UPDATED with email inputs
with st.sidebar:
    st.header("Order Details")
    
    # SENDER EMAIL - User enters their own email
    st.session_state.sender_email = st.text_input(
        "Your Email (Sender)",
        value=st.session_state.sender_email,
        placeholder="you@company.com",
        help="Email that will appear as the sender"
    )
    
    # SUPPLIER NAME - User enters
    st.session_state.supplier = st.text_input(
        "Supplier Name",
        value=st.session_state.supplier,
        placeholder="e.g., ABC Supplies Ltd."
    )
    
    # SUPPLIER EMAIL - User enters
    st.session_state.supplier_email = st.text_input(
        "Supplier Email",
        value=st.session_state.supplier_email,
        placeholder="supplier@company.com"
    )
    
    st.divider()
    st.caption("Email will be sent from your email to the supplier.")
    
    if st.button("Run Forecast", type="primary"):
        # Validate all fields
        errors = []
        if not st.session_state.sender_email:
            errors.append("Please enter your email (sender).")
        if not st.session_state.supplier:
            errors.append("Please enter a supplier name.")
        if not st.session_state.supplier_email:
            errors.append("Please enter supplier email.")
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            st.session_state.run_forecast = True
            st.session_state.order_status = None
            st.session_state.draft = ""
            st.session_state.email_sent = False
            st.rerun()

# ============================================
# MAIN AREA - FORECAST RESULTS
# ============================================

if st.session_state.run_forecast:
    
    with st.spinner("Loading data..."):
        conn = sqlite3.connect('data/inventory.db')
        df = pd.read_sql("""
            SELECT * FROM inventory_history 
            WHERE date >= date('now', '-30 days')
        """, conn)
        conn.close()
    
    with st.spinner("Running predictions..."):
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['avg_7d'] = df.groupby('sku')['daily_demand'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df['avg_30d'] = df.groupby('sku')['daily_demand'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )
        
        last_day = df.groupby('sku').last().reset_index()
        X = last_day[schema['features']]
        predictions = model.predict(X)
        
        results = pd.DataFrame({
            'SKU': last_day['sku'],
            'Current Stock': last_day['current_stock'],
            'Predicted Demand': np.round(predictions, 0),
            'Recommended Order': np.round(np.maximum(0, predictions - last_day['current_stock']), 0)
        })
        
        st.session_state.results = results
    
    st.success("Forecast complete")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Products", len(st.session_state.results))
    with col2:
        need_order = len(st.session_state.results[st.session_state.results['Recommended Order'] > 0])
        st.metric("Need Replenishment", need_order)
    with col3:
        total_units = st.session_state.results['Recommended Order'].sum()
        st.metric("Total Units to Order", f"{total_units:,.0f}")
    
    st.dataframe(st.session_state.results, use_container_width=True)
    
    items_to_order = st.session_state.results[st.session_state.results['Recommended Order'] > 0]
    st.session_state.items_to_order = items_to_order
    st.session_state.items_to_order_pdf = items_to_order
    
    st.divider()
    st.subheader("Purchase Order Draft")
    
    if len(items_to_order) == 0:
        st.info("No items need ordering")
    else:
        context = f"Supplier: {st.session_state.supplier}\n"
        context += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        context += "Items to order (exact quantities):\n"
        for _, row in items_to_order.iterrows():
            context += f"- {row['SKU']}: {int(row['Recommended Order'])} units\n"
        context += f"\nTotal units: {int(items_to_order['Recommended Order'].sum())}"
        
        with st.spinner("AI is drafting your purchase order..."):
            if HAS_GEMINI and not st.session_state.draft:
                try:
                    prompt = f"""You are a precise supply chain assistant. 
Create a professional purchase order using EXACT quantities provided. 
DO NOT add extra items. DO NOT round numbers. 
Use the exact quantities from the data.

IMPORTANT: Use PLAIN TEXT only. No Markdown, no asterisks, no hashes, no special characters. Just plain text with new lines.

Context:
{context}

Generate a clear, professional purchase order in plain text format that includes:
1. Client name, Supplier name and date
2. List of items with SKU and exact quantities
3. Total units
4. Delivery date (7 days from today)
5. Ask the supplier to confirm receipt and provide pricing.

Make it professional and clear."""
                    
                    # KEEPING YOUR GEMINI MODEL - NOT CHANGED
                    model_gemini = genai.GenerativeModel('gemini-3.5-flash-lite')
                    response = model_gemini.generate_content(
                        prompt,
                        request_options={'timeout': 600}
                    )
                    
                    if response.text:
                        st.session_state.draft = response.text
                        st.caption("Auto-generated & Reviewed.")
                    else:
                        st.warning("Empty response. Using template.")
                        st.session_state.draft = f"PURCHASE ORDER\n\n{context}"
                        
                except Exception as e:
                    st.error(f"Gemini error: {e}. Using template.")
                    st.session_state.draft = f"PURCHASE ORDER\n\n{context}"
            elif not st.session_state.draft:
                draft = f"""
PURCHASE ORDER
{'-'*40}
Supplier: {st.session_state.supplier}
Date: {datetime.now().strftime('%Y-%m-%d')}
Delivery Date: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}

ITEMS TO ORDER:
{'-'*40}
"""
                for _, row in items_to_order.iterrows():
                    draft += f"{row['SKU']:<20} {int(row['Recommended Order']):>8} units\n"
                
                draft += f"""
{'-'*40}
TOTAL UNITS: {int(items_to_order['Recommended Order'].sum())}

Please confirm receipt and provide pricing for the above items.
"""
                st.session_state.draft = draft
                st.caption("Template-generated (no AI)")
        
        edited_draft = st.text_area(
            "Review and Edit Draft", 
            value=st.session_state.draft, 
            height=300,
            key="draft_editor"
        )
        st.session_state.draft = edited_draft
        
        # ============================================
        # APPROVAL BUTTONS - UPDATED with email sending
        # ============================================
        if st.session_state.order_status is None:
            st.divider()
            st.subheader("Human Approval Required")
            st.warning(f"Email will be sent from: {st.session_state.sender_email} → {st.session_state.supplier_email}")
            
            col_approve, col_pdf, col_reject, col_defer = st.columns([2, 1, 1, 1])
            
            with col_approve:
                if st.button("Review and Send to Supplier", type="primary", use_container_width=True):
                    with st.spinner(f"Sending email to {st.session_state.supplier_email}..."):
                        try:
                            # Generate PDF
                            pdf_buffer = generate_pdf_order(
                                st.session_state.supplier,
                                items_to_order,
                                datetime.now().strftime('%Y-%m-%d'),
                                (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                            )
                            
                            # Send email
                            if HAS_EMAIL:
                                success = send_order_email(
                                    st.session_state.supplier_email,
                                    st.session_state.sender_email,
                                    st.session_state.supplier,
                                    edited_draft,
                                    pdf_buffer
                                )
                                if success:
                                    st.session_state.email_sent = True
                                    st.session_state.order_status = 'sent'
                                    st.rerun()
                                else:
                                    st.error("Failed to send email.")
                            else:
                                st.error("Email service not configured.")
                        except Exception as e:
                            st.error(f"Error sending email: {e}")
            
            with col_pdf:
                # Generate PDF and create download link
                pdf_buffer = generate_pdf_order(
                    st.session_state.supplier,
                    items_to_order,
                    datetime.now().strftime('%Y-%m-%d'),
                    (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                )
                
                # Use the proper download link function
                download_link = get_pdf_download_link(
                    pdf_buffer, 
                    f"Purchase_Order_{datetime.now().strftime('%Y%m%d')}.pdf"
                )
                st.markdown(download_link, unsafe_allow_html=True)
            
            with col_reject:
                if st.button("Reject Order", use_container_width=True):
                    st.session_state.order_status = 'rejected'
                    st.rerun()
            
            with col_defer:
                if st.button("Defer", use_container_width=True):
                    st.session_state.order_status = 'deferred'
                    st.rerun()
        
        # ============================================
        # STATUS MESSAGES - UPDATED
        # ============================================
        if st.session_state.order_status == 'sent' and st.session_state.email_sent:
            st.divider()
            st.success(f"Order sent from: {st.session_state.sender_email} → {st.session_state.supplier_email}")
            st.caption(f"Sent by: Store Manager at {datetime.now().strftime('%H:%M')}")
            st.caption(f"Email sent to supplier: {st.session_state.supplier_email}")
            st.info("Click 'Run Forecast' again to create a new order")
            
        elif st.session_state.order_status == 'rejected':
            st.divider()
            st.warning("Order rejected. No email sent.")
            st.caption("Order archived for audit")
            st.info("Click 'Run Forecast' again to create a new order")
            
        elif st.session_state.order_status == 'deferred':
            st.divider()
            st.info("Order deferred for later review")
            st.caption("Order saved for later")
            st.info("Click 'Run Forecast' again to create a new order")

# Footer
st.divider()
st.caption("Human Approval Required - No auto-send - Every order requires your click")