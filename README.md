# Smart Inventory Forecaster

AI-powered inventory forecasting and purchase order automation with 100% human approval.

## Features

- **ML Demand Forecasting** - XGBoost predicts future demand
- **AI Purchase Orders** - Gemini generates professional drafts
- **PDF Generation** - Clean, professional PDFs (no prices shown)
- **Email Sending** - SendGrid delivers orders to suppliers
- **Human Approval** - Every order requires manual review
- **No Auto-Send** - Orders only send when you click "Review and Send"

## Tech Stack

- **Frontend:** Streamlit
- **ML:** XGBoost, Pandas, NumPy, Scikit-learn
- **AI:** Google Gemini 3.5 Flash Lite
- **Email:** SendGrid API
- **PDF:** ReportLab
- **Database:** SQLite (mock data for demo)

## How It Works

1. User enters supplier name and email
2. Clicks "Run Forecast"
3. ML model predicts demand for each SKU
4. Gemini AI drafts a professional purchase order
5. Human reviews and edits the draft
6. Clicks "Review and Send"
7. PDF generated and attached
8. Email sent to supplier ✅

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/TheMark2000/inventory-forecaster.git
cd inventory-forecaster