import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime
import calendar
import io
from itertools import combinations
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ----------------------------------------------------
# INDIAN NUMBER FORMATTING HELPERS
# ----------------------------------------------------
def format_inr(value):
    """Format a numeric value as Indian Rupees using Crore/Lakh notation."""
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value / 1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"

def format_units(value):
    """Format a unit quantity using Crore/Lakh notation (no ₹ symbol)."""
    if value >= 1e7:
        return f"{value / 1e7:.2f} Cr"
    elif value >= 1e5:
        return f"{value / 1e5:.2f} L"
    else:
        return f"{value:,}"

# ----------------------------------------------------

# ─────────────────────────────────────────────────────────────
# ABC INVENTORY HELPER
# ─────────────────────────────────────────────────────────────
def compute_abc(sku_agg):
    """Assigns A / B / C class to each SKU based on cumulative revenue contribution."""
    if len(sku_agg) == 0:
        return sku_agg
    df = sku_agg.sort_values('Total_Revenue', ascending=False).copy()
    df['Cum_Rev'] = df['Total_Revenue'].cumsum()
    total = df['Total_Revenue'].sum()
    df['Cum_Pct'] = df['Cum_Rev'] / total * 100
    df['ABC_Class'] = df['Cum_Pct'].apply(
        lambda x: 'A' if x <= 70 else ('B' if x <= 90 else 'C')
    )
    return df


# ─────────────────────────────────────────────────────────────
# COHORT RETENTION HELPER
# ─────────────────────────────────────────────────────────────
def compute_cohort(df):
    """Builds a monthly cohort retention matrix from transaction data."""
    if len(df) == 0:
        return pd.DataFrame()
    tmp = df[['Customer_ID', 'Billing_Date']].copy()
    tmp['OrderMonth'] = tmp['Billing_Date'].dt.to_period('M')
    cohort_month = tmp.groupby('Customer_ID')['OrderMonth'].min().rename('CohortMonth')
    tmp = tmp.join(cohort_month, on='Customer_ID')
    tmp['CohortIndex'] = (tmp['OrderMonth'] - tmp['CohortMonth']).apply(lambda x: x.n)
    cohort_data = tmp.groupby(['CohortMonth', 'CohortIndex'])['Customer_ID'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='Customer_ID')
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100
    return retention


# ─────────────────────────────────────────────────────────────
# MARKET BASKET HELPER
# ─────────────────────────────────────────────────────────────
def compute_market_basket(df, top_n=30):
    """Finds top SKU pairs by co-occurrence and computes confidence."""
    if len(df) == 0:
        return pd.DataFrame()
    # Use agg(list) and cast SKU to str first to prevent NaN float entries
    tmp = df[['Customer_ID', 'Billing_Date', 'SKU']].copy()
    tmp['SKU'] = tmp['SKU'].astype(str)
    baskets = tmp.groupby(['Customer_ID', 'Billing_Date'])['SKU'].agg(list).reset_index()
    # Guard: only keep rows where SKU is actually a list with >1 items
    baskets = baskets[baskets['SKU'].apply(lambda x: isinstance(x, list) and len(x) > 1)]
    if len(baskets) == 0:
        return pd.DataFrame()
    pair_counts = {}
    sku_counts = {}
    for _, row in baskets.iterrows():
        skus = list(set(row['SKU']))
        for sku in skus:
            sku_counts[sku] = sku_counts.get(sku, 0) + 1
        for pair in combinations(sorted(skus), 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    total_baskets = len(baskets)
    rows = []
    for (a, b), cnt in pair_counts.items():
        conf_ab = cnt / sku_counts.get(a, 1) * 100
        conf_ba = cnt / sku_counts.get(b, 1) * 100
        support = cnt / total_baskets * 100
        rows.append({'SKU_A': a, 'SKU_B': b, 'Co_Occurrences': cnt,
                     'Support_%': round(support, 2),
                     'Confidence_AtoB_%': round(conf_ab, 1),
                     'Confidence_BtoA_%': round(conf_ba, 1)})
    result = pd.DataFrame(rows).sort_values('Co_Occurrences', ascending=False).head(top_n)
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# PDF EXPORT HELPER
# ─────────────────────────────────────────────────────────────
def generate_pdf_report(kpi_rev, kpi_qty, kpi_cust, kpi_skus,
                         top_skus_df, top_cust_df, rfm_summary_df,
                         abc_summary_df, date_range):
    """Generates an executive PDF report using reportlab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=20, textColor=rl_colors.HexColor('#00ADB5'),
                                  spaceAfter=6, alignment=TA_CENTER)
    story.append(Paragraph('Billing & Transaction Analytics', title_style))
    story.append(Paragraph('Executive Summary Report', styles['Normal']))
    story.append(Spacer(1, 0.3*cm))

    if isinstance(date_range, tuple) and len(date_range) == 2:
        dr_text = f"Period: {date_range[0].strftime('%d %b %Y')} \u2013 {date_range[1].strftime('%d %b %Y')}"
    else:
        dr_text = "Period: Full Dataset"
    story.append(Paragraph(dr_text, styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width='100%', thickness=1, color=rl_colors.HexColor('#00ADB5')))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph('Key Performance Indicators', styles['Heading2']))
    kpi_data = [
        ['Metric', 'Value'],
        ['Total Revenue', kpi_rev],
        ['Total Quantity Sold', f'{kpi_qty:,}'],
        ['Active Customers', f'{kpi_cust:,}'],
        ['Active SKUs', f'{kpi_skus:,}'],
    ]
    kpi_table = Table(kpi_data, colWidths=[8*cm, 8*cm])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#00ADB5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.HexColor('#f5f5f5'), rl_colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#cccccc')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.6*cm))

    story.append(Paragraph('Top 10 SKUs by Revenue', styles['Heading2']))
    sku_data = [['SKU', 'Revenue (\u20b9)', 'Qty', 'Transactions']] + [
        [r['SKU'], f"\u20b9{r['Total_Revenue']:,.0f}", f"{int(r['Total_Quantity']):,}", f"{int(r['Tx_Count']):,}"]
        for _, r in top_skus_df.iterrows()
    ]
    sku_table = Table(sku_data, colWidths=[4*cm, 5*cm, 3*cm, 4*cm])
    sku_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#222831')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.HexColor('#f5f5f5'), rl_colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#cccccc')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(sku_table)
    story.append(Spacer(1, 0.6*cm))

    story.append(Paragraph('Top 10 Customers by Spending', styles['Heading2']))
    cust_data = [['Customer ID', 'Revenue (\u20b9)', 'Qty', 'Transactions']] + [
        [r['Customer_ID'], f"\u20b9{r['Total_Revenue']:,.0f}", f"{int(r['Total_Quantity']):,}", f"{int(r['Tx_Count']):,}"]
        for _, r in top_cust_df.iterrows()
    ]
    cust_table = Table(cust_data, colWidths=[4*cm, 5*cm, 3*cm, 4*cm])
    cust_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#222831')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.HexColor('#f5f5f5'), rl_colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#cccccc')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(cust_table)
    story.append(Spacer(1, 0.6*cm))

    if rfm_summary_df is not None and len(rfm_summary_df) > 0:
        story.append(Paragraph('RFM Customer Segment Summary', styles['Heading2']))
        rfm_data = [['Segment', 'Customers', 'Avg Recency (days)', 'Avg Freq', 'Total Revenue']] + [
            [r['Segment'], f"{int(r['Customers']):,}",
             f"{r['Avg_Recency']:.1f}", f"{r['Avg_Freq']:.1f}",
             f"\u20b9{r['Total_Revenue']:,.0f}"]
            for _, r in rfm_summary_df.iterrows()
        ]
        rfm_table = Table(rfm_data, colWidths=[4*cm, 3*cm, 4*cm, 3*cm, 4*cm])
        rfm_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#00ADB5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.HexColor('#f5f5f5'), rl_colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#cccccc')),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(rfm_table)
        story.append(Spacer(1, 0.6*cm))

    if abc_summary_df is not None and len(abc_summary_df) > 0:
        story.append(Paragraph('ABC Inventory Class Summary', styles['Heading2']))
        abc_group = abc_summary_df.groupby('ABC_Class').agg(
            SKU_Count=('SKU', 'count'),
            Total_Revenue=('Total_Revenue', 'sum')
        ).reset_index()
        abc_data = [['Class', 'SKU Count', 'Total Revenue']] + [
            [r['ABC_Class'], f"{int(r['SKU_Count']):,}", f"\u20b9{r['Total_Revenue']:,.0f}"]
            for _, r in abc_group.iterrows()
        ]
        abc_table = Table(abc_data, colWidths=[4*cm, 5*cm, 7*cm])
        abc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#222831')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.HexColor('#f5f5f5'), rl_colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#cccccc')),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(abc_table)

    doc.build(story)
    buffer.seek(0)
    return buffer


def compute_rfm(df_full, customer_agg):
    if len(customer_agg) == 0:
        customer_agg['Segment'] = []
        return customer_agg
    snapshot_date = df_full['Billing_Date'].max() + pd.Timedelta(days=1)
    customer_agg['Recency'] = (snapshot_date - customer_agg['Last_Purchase']).dt.days
    customer_agg['R_Score'] = pd.qcut(customer_agg['Recency'].rank(method='first'), q=4, labels=[4, 3, 2, 1])
    customer_agg['F_Score'] = pd.qcut(customer_agg['Tx_Count'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
    customer_agg['M_Score'] = pd.qcut(customer_agg['Total_Revenue'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
    def assign_segment(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        if r >= 3 and f >= 3 and m >= 3: return 'Champions'
        elif f >= 3: return 'Loyal Customers'
        elif m >= 3: return 'Big Spenders'
        elif r <= 2: return 'At Risk'
        else: return 'Lost Customers'
    customer_agg['Segment'] = customer_agg.apply(assign_segment, axis=1)
    return customer_agg

# 1. PAGE CONFIGURATION & THEME STYLING
# ----------------------------------------------------
st.set_page_config(
    page_title="Billing & Transaction Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Variables using requested palette
theme = 'dark'
bg_gradient = "linear-gradient(135deg, #222831 0%, #1A1F26 100%)"
text_primary = "#EEEEEE"
text_secondary = "#EEEEEE"
text_muted = "#00ADB5"
sidebar_bg = "rgba(34, 40, 49, 0.95)"
border_color = "rgba(57, 62, 70, 0.8)"
card_bg = "rgba(57, 62, 70, 0.4)"
card_hover_border = "#00ADB5"
header_bg = "linear-gradient(90deg, rgba(34, 40, 49, 0.8) 0%, rgba(57, 62, 70, 0.6) 100%)"
tab_bg = "rgba(57, 62, 70, 0.5)"
tab_active = "rgba(0, 173, 181, 0.2)"
scroll_track = "#222831"
scroll_thumb = "#393E46"
scroll_thumb_hover = "#00ADB5"
plot_bg = "#222831"
plot_face = "rgba(57, 62, 70, 0.2)"
grid_color = "rgba(238, 238, 238, 0.1)"

# Custom Premium CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {{
        background: {bg_gradient} !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: {text_secondary} !important;
    }}
    
    /* Override sidebar style */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}
    
    /* Glassmorphic KPI Cards */
    .kpi-container {{
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }}
    .kpi-card {{
        flex: 1 1 220px;
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 22px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);
        border-color: {card_hover_border};
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }}
    .kpi-title {{
        font-size: 0.78rem;
        font-weight: 600;
        color: {text_muted};
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }}
    .kpi-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {text_primary};
        margin-top: 8px;
        letter-spacing: -0.02em;
    }}
    .kpi-subtext {{
        font-size: 0.73rem;
        color: #00ADB5;
        margin-top: 6px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    .kpi-glow {{
        position: absolute;
        bottom: -35px;
        right: -35px;
        width: 110px;
        height: 110px;
        border-radius: 50%;
        filter: blur(40px);
        opacity: 0.16;
        pointer-events: none;
    }}
    
    /* Elegant Title Banner */
    .dashboard-header {{
        background: {header_bg};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
    }}
    
    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        background-color: transparent !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {tab_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 10px 24px !important;
        color: {text_muted} !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {tab_active} !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
        color: #00ADB5 !important;
    }}
    
    /* Scrollbars customization */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {scroll_track};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {scroll_thumb};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {scroll_thumb_hover};
    }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. DATA PIPELINE WITH OPTIMIZED CACHING
# ----------------------------------------------------
@st.cache_data
def load_transaction_data():
    # Make sure target file exists
    data_path = "processed_billing_data.csv"
    if not os.path.exists(data_path):
        st.error(f"FATAL ERROR: Processed dataset '{data_path}' not found. Please run the model prep or ensure it is available.")
        st.stop()
        
    # Read the data, optimizing pandas memory using specific formats
    df = pd.read_csv(data_path)
    df['Billing_Date'] = pd.to_datetime(df['Billing_Date'])
    
    # Compress numeric columns to save memory & boost indexing speed
    df['Year'] = df['Year'].astype(np.int16)
    df['Month'] = df['Month'].astype(np.int8)
    df['Day'] = df['Day'].astype(np.int8)
    df['Quarter'] = df['Quarter'].astype(np.int8)
    df['Is_Weekend'] = df['Is_Weekend'].astype(bool)
    df['Price'] = df['Price'].astype(np.float32)
    df['Revenue'] = df['Revenue'].astype(np.float32)
    df['Billing_Quantity'] = df['Billing_Quantity'].astype(np.int32)
    
    # Convert large repetitive string columns into Categorical dtype
    df['SKU'] = df['SKU'].astype('category')
    df['Customer_ID'] = df['Customer_ID'].astype('category')
    df['Month_Name'] = df['Month_Name'].astype('category')
    df['Weekday'] = df['Weekday'].astype('category')
    
    return df

@st.cache_data
def precompute_global_aggregates(df_full):
    """
    Computes static dashboards metrics and analytics to allow instant load.
    These are the default unfiltered sets.
    """
    # Daily time-series totals
    daily_ts = df_full.groupby('Billing_Date')[['Billing_Quantity', 'Revenue']].sum().reset_index()
    daily_ts = daily_ts.sort_values('Billing_Date')
    daily_ts['Cumulative_Revenue'] = daily_ts['Revenue'].cumsum()
    
    # SKU specific aggregates
    sku_agg = df_full.groupby('SKU').agg(
        Total_Revenue=('Revenue', 'sum'),
        Total_Quantity=('Billing_Quantity', 'sum'),
        Avg_Price=('Price', 'mean'),
        Tx_Count=('Billing_Date', 'count')
    ).reset_index()
    sku_agg = sku_agg[sku_agg['Total_Quantity'] > 0] # Filter out dead stock
    
    # Customer specific aggregates
    customer_agg = df_full.groupby('Customer_ID').agg(
        Total_Revenue=('Revenue', 'sum'),
        Total_Quantity=('Billing_Quantity', 'sum'),
        Tx_Count=('Billing_Date', 'count'),
        Last_Purchase=('Billing_Date', 'max')
    ).reset_index()
    customer_agg = customer_agg[customer_agg['Total_Revenue'] > 0]
    customer_agg = compute_rfm(df_full, customer_agg)
    
    # Weekday metrics
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_agg = df_full.groupby('Weekday', observed=False)[['Billing_Quantity', 'Revenue']].sum().reset_index()
    weekday_agg['Weekday'] = pd.Categorical(weekday_agg['Weekday'], categories=days_order, ordered=True)
    weekday_agg = weekday_agg.sort_values('Weekday')
    
    # Month metrics
    cal_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_agg = df_full.groupby('Month_Name', observed=False)[['Billing_Quantity', 'Revenue']].sum().reset_index()
    month_agg['Month_Name'] = pd.Categorical(month_agg['Month_Name'], categories=cal_months, ordered=True)
    month_agg = month_agg.sort_values('Month_Name')
    
    # Seasonality Matrix: Month vs Day of Week
    heatmap_matrix = df_full.groupby(['Month_Name', 'Weekday'], observed=False)['Billing_Quantity'].sum().unstack().fillna(0)
    heatmap_matrix = heatmap_matrix.reindex(index=cal_months, columns=days_order)
    
    return daily_ts, sku_agg, customer_agg, weekday_agg, month_agg, heatmap_matrix

# Initialize Data Loading
with st.spinner("🚀 Loading 1,000,000 billing records into high-performance cache..."):
    df_raw = load_transaction_data()

with st.spinner("📊 Analyzing structural indexes and temporal features..."):
    cached_daily, cached_sku, cached_cust, cached_weekday, cached_month, cached_heatmap = precompute_global_aggregates(df_raw)

# ----------------------------------------------------
# 3. INTERACTIVE SIDEBAR & FILTERS
# ----------------------------------------------------
st.sidebar.markdown(f"""
<div style='text-align: center; padding: 10px; margin-bottom: 1.5rem;'>
    <h2 style='color: {text_primary}; font-weight: 700; margin: 0; font-size: 1.4rem;'>🧭 CONTROL PANELS</h2>
    <span style='color: #00ADB5; font-size: 0.8rem; font-weight: 500;'>SYSTEM CONFIGURATIONS</span>
</div>
""", unsafe_allow_html=True)

# 1. Date Range Slider
min_date = df_raw['Billing_Date'].min().date()
max_date = df_raw['Billing_Date'].max().date()
st.sidebar.markdown("<h4 style='color: {text_secondary}; font-size: 0.95rem; font-weight: 600; margin-bottom: 5px;'>📅 Billing Timeline</h4>", unsafe_allow_html=True)

if 'timeline_window' not in st.session_state:
    st.session_state.timeline_window = (min_date, max_date)

selected_dates = st.sidebar.date_input(
    "Timeline Window",
    value=st.session_state.timeline_window,
    label_visibility="collapsed"
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    capped_start = max(selected_dates[0], min_date)
    capped_end = min(selected_dates[1], max_date)
    
    if (capped_start, capped_end) != selected_dates:
        st.session_state.timeline_window = (capped_start, capped_end)
        st.rerun()
    else:
        st.session_state.timeline_window = selected_dates

date_range = st.session_state.timeline_window

# 2. Weekday/Weekend Filter
st.sidebar.markdown("<h4 style='color: {text_secondary}; font-size: 0.95rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 5px;'>⏳ Temporal Filter</h4>", unsafe_allow_html=True)
day_filter = st.sidebar.selectbox(
    "Filter by Days",
    options=["All Business Days", "Weekdays Only", "Weekends Only"],
    label_visibility="collapsed"
)

# 3. SKU Custom Selectors
st.sidebar.markdown("<h4 style='color: {text_secondary}; font-size: 0.95rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 5px;'>📦 Product Portfolios</h4>", unsafe_allow_html=True)
# Fetch Top 50 SKU IDs to keep dropdown extremely responsive
top_50_skus = cached_sku.sort_values(by='Total_Revenue', ascending=False).head(50)['SKU'].tolist()
selected_skus = st.sidebar.multiselect(
    "Target Top SKUs",
    options=top_50_skus,
    help="Select one or multiple SKUs from the top 50 revenue-generating products."
)

# 4. Custom exact SKU search
search_sku = st.sidebar.text_input(
    "Exact SKU Identifier Search",
    value="",
    placeholder="e.g. GC1010",
    help="Input an exact product SKU ID (case sensitive) to isolate its analytics."
).strip()

# 5. Price / Quantity Threshold Sliders
st.sidebar.markdown("<h4 style='color: {text_secondary}; font-size: 0.95rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 5px;'>💵 Pricing Brackets (₹)</h4>", unsafe_allow_html=True)
min_price = float(df_raw['Price'].min())
max_price = float(df_raw['Price'].max())
price_range = st.sidebar.slider(
    "Unit Price",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=50.0,
    label_visibility="collapsed"
)

st.sidebar.markdown("<h4 style='color: {text_secondary}; font-size: 0.95rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 5px;'>🛍️ Sales Volume / Bill</h4>", unsafe_allow_html=True)
min_qty = int(df_raw['Billing_Quantity'].min())
max_qty = int(df_raw['Billing_Quantity'].max())
qty_range = st.sidebar.slider(
    "Billing Quantity",
    min_value=min_qty,
    max_value=max_qty,
    value=(min_qty, max_qty),
    step=5,
    label_visibility="collapsed"
)


# Advanced Filters Additions
st.sidebar.markdown("<h4 style='color: {text_secondary}; font-size: 0.95rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 5px;'>🎯 Customer Segments</h4>", unsafe_allow_html=True)
available_segments = ['Champions', 'Loyal Customers', 'Big Spenders', 'At Risk', 'Lost Customers']
selected_segments = st.sidebar.multiselect("Filter by Segment", options=available_segments)

st.sidebar.markdown("<h4 style='color: {text_secondary}; font-size: 0.95rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 5px;'>💰 Rev Threshold / Tx (₹)</h4>", unsafe_allow_html=True)
rev_threshold = st.sidebar.slider("Min Revenue", min_value=0, max_value=int(df_raw['Revenue'].max()), value=0, step=100)

st.sidebar.markdown("<h4 style='color: {text_secondary}; font-size: 0.95rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 5px;'>📅 Quarter</h4>", unsafe_allow_html=True)
selected_quarters = st.sidebar.multiselect("Filter by Quarter", options=[1, 2, 3, 4], format_func=lambda x: f"Q{x}")

st.sidebar.markdown("<h4 style='color: {text_secondary}; font-size: 0.95rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 5px;'>📈 Chart Grouping</h4>", unsafe_allow_html=True)
grouping = st.sidebar.selectbox("Group By", options=["Day", "Week", "Month", "Quarter"])
yoy_comparison = st.sidebar.checkbox("Year-over-Year comparison", value=False)

# Clear Button in Sidebar
if st.sidebar.button("🔄 Reset Global Filters", use_container_width=True):
    st.rerun()

# ----------------------------------------------------
# 4. SLICING & DICING PROCESSOR
# ----------------------------------------------------
# Check if any filters are active
is_filtered = False
df_filtered = df_raw

# 1. Apply Date Filter
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0])
    end_dt = pd.Timestamp(date_range[1])
    if start_dt > pd.Timestamp(min_date) or end_dt < pd.Timestamp(max_date):
        is_filtered = True
        df_filtered = df_filtered[(df_filtered['Billing_Date'] >= start_dt) & (df_filtered['Billing_Date'] <= end_dt)]

# 2. Apply Weekday/Weekend Filter
if day_filter == "Weekdays Only":
    is_filtered = True
    df_filtered = df_filtered[df_filtered['Is_Weekend'] == False]
elif day_filter == "Weekends Only":
    is_filtered = True
    df_filtered = df_filtered[df_filtered['Is_Weekend'] == True]

# 3. Apply Multi-SKU Filter
if selected_skus:
    is_filtered = True
    df_filtered = df_filtered[df_filtered['SKU'].isin(selected_skus)]

# 4. Apply Exact SKU Search
if search_sku:
    is_filtered = True
    df_filtered = df_filtered[df_filtered['SKU'] == search_sku]

# 5. Apply Price Range Filter
if price_range[0] > min_price or price_range[1] < max_price:
    is_filtered = True
    df_filtered = df_filtered[(df_filtered['Price'] >= price_range[0]) & (df_filtered['Price'] <= price_range[1])]


# Apply Segment Filter (Note: applied to customer_agg post RFM or pre-filter if needed. Wait, segment filter requires joining segment back to txs)
if selected_segments:
    is_filtered = True
    # We must compute segments globally first, then filter txs
    valid_customers = cached_cust[cached_cust['Segment'].isin(selected_segments)]['Customer_ID']
    df_filtered = df_filtered[df_filtered['Customer_ID'].isin(valid_customers)]

if rev_threshold > 0:
    is_filtered = True
    df_filtered = df_filtered[df_filtered['Revenue'] >= rev_threshold]

if selected_quarters:
    is_filtered = True
    df_filtered = df_filtered[df_filtered['Quarter'].isin(selected_quarters)]

# 6. Apply Quantity Range Filter
if qty_range[0] > min_qty or qty_range[1] < max_qty:
    is_filtered = True
    df_filtered = df_filtered[(df_filtered['Billing_Quantity'] >= qty_range[0]) & (df_filtered['Billing_Quantity'] <= qty_range[1])]

# Calculate or fetch visual datasets depending on whether filters are actively subsetting rows
if is_filtered:
    # If filtered, compute aggregates on the subset (very fast due to pre-sliced Pandas dataframe)
    daily_ts = df_filtered.groupby('Billing_Date')[['Billing_Quantity', 'Revenue']].sum().reset_index().sort_values('Billing_Date')
    daily_ts['Cumulative_Revenue'] = daily_ts['Revenue'].cumsum()
    
    sku_agg = df_filtered.groupby('SKU', observed=False).agg(
        Total_Revenue=('Revenue', 'sum'),
        Total_Quantity=('Billing_Quantity', 'sum'),
        Avg_Price=('Price', 'mean'),
        Tx_Count=('Billing_Date', 'count')
    ).reset_index()
    sku_agg = sku_agg[sku_agg['Total_Quantity'] > 0]
    
    customer_agg = df_filtered.groupby('Customer_ID', observed=False).agg(
        Total_Revenue=('Revenue', 'sum'),
        Total_Quantity=('Billing_Quantity', 'sum'),
        Tx_Count=('Billing_Date', 'count'),
        Last_Purchase=('Billing_Date', 'max')
    ).reset_index()
    customer_agg = customer_agg[customer_agg['Total_Revenue'] > 0]
    customer_agg = compute_rfm(df_filtered, customer_agg)
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_agg = df_filtered.groupby('Weekday', observed=False)[['Billing_Quantity', 'Revenue']].sum().reset_index()
    weekday_agg['Weekday'] = pd.Categorical(weekday_agg['Weekday'], categories=days_order, ordered=True)
    weekday_agg = weekday_agg.sort_values('Weekday')
    
    cal_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_agg = df_filtered.groupby('Month_Name', observed=False)[['Billing_Quantity', 'Revenue']].sum().reset_index()
    month_agg['Month_Name'] = pd.Categorical(month_agg['Month_Name'], categories=cal_months, ordered=True)
    month_agg = month_agg.sort_values('Month_Name')
    
    heatmap_matrix = df_filtered.groupby(['Month_Name', 'Weekday'], observed=False)['Billing_Quantity'].sum().unstack().fillna(0)
    heatmap_matrix = heatmap_matrix.reindex(index=cal_months, columns=days_order)
else:
    # Use global fast-cache aggregations (instantaneous)
    daily_ts = cached_daily
    sku_agg = cached_sku
    customer_agg = cached_cust
    weekday_agg = cached_weekday
    month_agg = cached_month
    heatmap_matrix = cached_heatmap

# Compute active KPIs based on filtered context
kpi_revenue = float(df_filtered['Revenue'].sum())
kpi_quantity = int(df_filtered['Billing_Quantity'].sum())
kpi_customers = int(df_filtered['Customer_ID'].nunique())
kpi_skus = int(df_filtered['SKU'].nunique())
kpi_avg_price = float(df_filtered['Price'].mean()) if len(df_filtered) > 0 else 0.0

# Calculate previous period stats
data_min_date = pd.Timestamp(df_raw['Billing_Date'].min().date())
data_max_date = pd.Timestamp(df_raw['Billing_Date'].max().date())

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0])
    end_dt = pd.Timestamp(date_range[1])
    days_delta = (end_dt - start_dt).days + 1
    past_end_dt = start_dt - pd.Timedelta(days=1)
    past_start_dt = past_end_dt - pd.Timedelta(days=days_delta - 1)

    # Format period label for display
    period_label = f"{past_start_dt.strftime('%d %b %Y')} – {past_end_dt.strftime('%d %b %Y')}"

    # Only compute comparison if previous period overlaps with actual data range
    prev_period_has_data = (past_end_dt >= data_min_date) and (past_start_dt <= data_max_date)

    if prev_period_has_data:
        df_past = df_raw[
            (df_raw['Billing_Date'] >= past_start_dt) &
            (df_raw['Billing_Date'] <= past_end_dt)
        ]
        past_revenue = float(df_past['Revenue'].sum())
        past_quantity = int(df_past['Billing_Quantity'].sum())
        past_customers = int(df_past['Customer_ID'].nunique())
        past_skus = int(df_past['SKU'].nunique())

        def get_diff_html(current, past, label):
            if past == 0:
                return f"<span style='color: #a1a1aa;'>-- vs {label}</span>"
            diff = ((current - past) / past) * 100
            if diff > 0:
                return f"<span style='color: #4ade80;'>▲ +{diff:.1f}%</span> <span style='color: #a1a1aa;'>vs {label}</span>"
            elif diff < 0:
                return f"<span style='color: #f87171;'>▼ {diff:.1f}%</span> <span style='color: #a1a1aa;'>vs {label}</span>"
            else:
                return f"<span style='color: #a1a1aa;'>±0.0% vs {label}</span>"

        diff_rev  = get_diff_html(kpi_revenue,   past_revenue,   period_label)
        diff_qty  = get_diff_html(kpi_quantity,   past_quantity,  period_label)
        diff_cust = get_diff_html(kpi_customers,  past_customers, period_label)
        diff_skus = get_diff_html(kpi_skus,       past_skus,      period_label)
    else:
        # Previous period is before the dataset starts — no comparison available
        no_data_msg = f"<span style='color: #a1a1aa;'>No data before {data_min_date.strftime('%d %b %Y')}</span>"
        diff_rev  = no_data_msg
        diff_qty  = no_data_msg
        diff_cust = no_data_msg
        diff_skus = no_data_msg
else:
    diff_rev  = "<span style='color: #a1a1aa;'>-- vs previous period</span>"
    diff_qty  = "<span style='color: #a1a1aa;'>-- vs previous period</span>"
    diff_cust = "<span style='color: #a1a1aa;'>-- vs previous period</span>"
    diff_skus = "<span style='color: #a1a1aa;'>-- vs previous period</span>"


# ----------------------------------------------------
# 5. DASHBOARD MAIN INTERFACE RENDERER
# ----------------------------------------------------
# Render Title Block
st.markdown(f"""
<div class='dashboard-header'>
    <div style='display: flex; align-items: center; gap: 15px;'>
        <div style='background: linear-gradient(135deg, #00ADB5, #00ADB5); width: 50px; height: 50px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);'>📈</div>
        <div>
            <h1 style='color: {text_primary}; font-weight: 700; margin: 0; font-size: 1.85rem; letter-spacing: -0.01em;'>Billing & Transaction Analytics</h1>
            <p style='color: {text_muted}; margin: 2px 0 0 0; font-size: 0.9rem; font-weight: 500;'>Exploratory Data Analysis Dashboard • Real-time Transaction Intelligence System</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Render Custom Metric Cards
st.markdown(f"""
<div class='kpi-container'>
    <div class='kpi-card'>
        <div class='kpi-title'>Total Revenue</div>
        <div class='kpi-value'>{format_inr(kpi_revenue)}</div>
        <div class='kpi-subtext'>
            {diff_rev}
        </div>
        <div class='kpi-glow' style='background: #00ADB5;'></div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-title'>Total Quantity Sold</div>
        <div class='kpi-value'>{format_units(kpi_quantity)}</div>
        <div class='kpi-subtext'>
            {diff_qty}
        </div>
        <div class='kpi-glow' style='background: #00ADB5;'></div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-title'>Active SKU Catalog</div>
        <div class='kpi-value'>{kpi_skus:,}</div>
        <div class='kpi-subtext'>
            {diff_skus}
        </div>
        <div class='kpi-glow' style='background: #00ADB5;'></div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-title'>Active Customer Base</div>
        <div class='kpi-value'>{kpi_customers:,}</div>
        <div class='kpi-subtext'>
            {diff_cust}
        </div>
        <div class='kpi-glow' style='background: #00ADB5;'></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Executive Summary",
    "📈 Temporal Seasonality",
    "📦 SKU Deep-Dive",
    "👥 Client Insights",
    "🏷️ ABC Inventory",
    "🔁 Cohort Retention",
    "🛒 Market Basket"
])

# ----------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# ----------------------------------------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h3 style='color: {text_primary}; margin-bottom: 15px;'>📈 Revenue Progression & Growth Timeline</h3>", unsafe_allow_html=True)
        # Check if daily time series has data
        if len(daily_ts) > 0:
            # Handle Grouping
            if grouping == 'Day':
                chart_data = df_filtered.groupby('Billing_Date')['Revenue'].sum().reset_index()
                x_col = 'Billing_Date'
            elif grouping == 'Week':
                chart_data = df_filtered.groupby(df_filtered['Billing_Date'].dt.to_period('W').apply(lambda r: r.start_time))['Revenue'].sum().reset_index()
                x_col = 'Billing_Date'
            elif grouping == 'Month':
                chart_data = df_filtered.groupby(df_filtered['Billing_Date'].dt.to_period('M').apply(lambda r: r.start_time))['Revenue'].sum().reset_index()
                x_col = 'Billing_Date'
            elif grouping == 'Quarter':
                chart_data = df_filtered.groupby(df_filtered['Billing_Date'].dt.to_period('Q').apply(lambda r: r.start_time))['Revenue'].sum().reset_index()
                x_col = 'Billing_Date'
            
            if yoy_comparison:
                chart_data['Year'] = chart_data['Billing_Date'].dt.year.astype(str)
                chart_data['DayOfYear'] = chart_data['Billing_Date'].dt.dayofyear
                fig = px.line(chart_data, x='DayOfYear', y='Revenue', color='Year', title=f"Revenue by {grouping} (YoY)")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#EEEEEE')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.area_chart(chart_data.set_index(x_col)[['Revenue']], color="#00ADB5", use_container_width=True)
        else:
            st.info("No matching transaction history in the selected timeline window.")
            
    with col2:
        st.markdown("<h3 style='color: {text_primary}; margin-bottom: 15px;'>⚖️ Weekday vs Weekend Sales</h3>", unsafe_allow_html=True)
        # Compute weekend split
        weekend_rev = df_filtered.groupby('Is_Weekend')['Revenue'].sum().reset_index()
        weekend_rev['DayType'] = weekend_rev['Is_Weekend'].map({True: 'Weekends', False: 'Weekdays'})
        
        # Horizontal bar chart using streamlit native components
        st.bar_chart(
            weekend_rev.set_index('DayType')['Revenue'], 
            horizontal=True, 
            color="#00ADB5", 
            use_container_width=True
        )
        
        st.markdown("<h3 style='color: {text_primary}; margin-top: 1.5rem; margin-bottom: 15px;'>🏆 Top 3 SKU Portfolios</h3>", unsafe_allow_html=True)
        top_skus = sku_agg.sort_values(by='Total_Revenue', ascending=False).head(3)
        for idx, row in top_skus.iterrows():
            st.markdown(f"""
            <div style='background: {tab_bg}; border: 1px solid {border_color}; border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='font-weight: 700; color: {text_primary};'>SKU: {row['SKU']}</span>
                    <span style='color: #00ADB5; font-weight: 600;'>{format_inr(row['Total_Revenue'])}</span>
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.75rem; color: {text_muted}; margin-top: 4px;'>
                    <span>Qty: {int(row['Total_Quantity']):,} units</span>
                    <span>Avg Price: ₹{row['Avg_Price']:,.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 2: TEMPORAL SEASONALITY
# ----------------------------------------------------
with tab2:
    st.markdown("<h3 style='color: {text_primary}; margin-bottom: 10px;'>📊 Sales Seasonality Heatmap (Month vs Weekday)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: {text_muted}; font-size: 0.85rem; margin-top: -8px; margin-bottom: 20px;'>Evaluates peak volumes (quantity sold) across the calendar year to isolate ordering behaviors.</p>", unsafe_allow_html=True)
    
    if heatmap_matrix.sum().sum() > 0:
        # Create a professional customized Seaborn heatmap
        fig, ax = plt.subplots(figsize=(12, 6.5))
        fig.patch.set_facecolor(plot_bg)
        ax.set_facecolor(plot_bg)
        
        # Style heatmap with custom color palette
        sns.heatmap(
            heatmap_matrix / 1e3,  # represent in thousands
            annot=True,
            fmt=".1f",
            cmap="viridis",
            cbar=True,
            cbar_kws={'label': 'Billing Units Sold (Thousands)'},
            linewidths=0.5,
            linecolor=(238/255, 238/255, 238/255, 0.1),
            ax=ax
        )
        
        # Typography and ticks styling
        ax.set_title("Order Volumetrics (Thousands of Units Purchased)", color=text_primary, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel("Weekday", color=text_primary, fontsize=11, labelpad=10)
        ax.set_ylabel("Month of Calendar Year", color=text_primary, fontsize=11, labelpad=10)
        
        ax.tick_params(colors=text_muted, labelsize=10)
        ax.figure.axes[-1].yaxis.label.set_color(text_primary)
        ax.figure.axes[-1].tick_params(colors=text_muted)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Insufficient data range to compute temporal heatmap matrices.")
        
    # Additional breakdowns
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: {text_primary}; margin-bottom: 15px;'>📅 Weekly Seasonality Pattern</h3>", unsafe_allow_html=True)
        # Total Quantity by Weekday
        st.bar_chart(
            weekday_agg.set_index('Weekday')['Billing_Quantity'],
            color="#00ADB5",
            use_container_width=True
        )
        
    with col2:
        st.markdown("<h3 style='color: {text_primary}; margin-bottom: 15px;'>📆 Monthly Seasonal Outliers</h3>", unsafe_allow_html=True)
        # Total Quantity by Month Name
        st.bar_chart(
            month_agg.set_index('Month_Name')['Billing_Quantity'],
            color="#00ADB5",
            use_container_width=True
        )

# ----------------------------------------------------
# TAB 3: SKU DEEP-DIVE
# ----------------------------------------------------
with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 style='color: {text_primary}; margin-bottom: 15px;'>📦 Top 10 SKUs by Sales Revenue</h3>", unsafe_allow_html=True)
        top10_revenue = sku_agg.sort_values(by='Total_Revenue', ascending=False).head(10)
        st.bar_chart(
            top10_revenue.set_index('SKU')['Total_Revenue'],
            color="#00ADB5",
            use_container_width=True
        )
        
    with col2:
        st.markdown("<h3 style='color: {text_primary}; margin-bottom: 15px;'>📦 Top 10 SKUs by Billing Quantities</h3>", unsafe_allow_html=True)
        top10_quantity = sku_agg.sort_values(by='Total_Quantity', ascending=False).head(10)
        st.bar_chart(
            top10_quantity.set_index('SKU')['Total_Quantity'],
            color="#00ADB5",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: {text_primary}; margin-bottom: 10px;'>📊 Price vs. Quantity Elasticity (Product Distribution Mapping)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: {text_muted}; font-size: 0.85rem; margin-top: -8px; margin-bottom: 20px;'>Evaluates the volumes of units sold relative to average price. Large bubbles indicate higher revenue products.</p>", unsafe_allow_html=True)

    if len(sku_agg) > 0:
        # Plotly is not installed, so we utilize high-performance interactive native streamlit scatter charts!
        # Select the top 100 SKUs by Transaction count to make the scatter interactive and not cluttered
        scatter_data = sku_agg.sort_values(by='Total_Revenue', ascending=False).head(200).copy()
        
        # Display scatter chart
        st.scatter_chart(
            data=scatter_data,
            x='Avg_Price',
            y='Total_Quantity',
            size='Total_Revenue',
            color='Avg_Price',
            use_container_width=True
        )
    else:
        st.info("No matching SKU profiles within selected filtering metrics.")

# ----------------------------------------------------
# TAB 4: CLIENT INSIGHTS
# ----------------------------------------------------
with tab4:
    st.markdown("<h3 style='color: {text_primary}; margin-bottom: 15px;'>🎯 RFM Customer Segmentation</h3>", unsafe_allow_html=True)
    if len(customer_agg) > 0 and 'Segment' in customer_agg.columns:
        col1, col2 = st.columns([1, 1])
        
        # Pie Chart
        with col1:
            seg_counts = customer_agg['Segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            fig_pie = px.pie(seg_counts, values='Count', names='Segment', title="Segment Distribution", hole=0.4, 
                             color_discrete_sequence=px.colors.sequential.Teal)
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#EEEEEE')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Treemap
        with col2:
            fig_tree = px.treemap(seg_counts, path=['Segment'], values='Count', title="Segment Treemap",
                                  color='Count', color_continuous_scale='Teal')
            fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#EEEEEE')
            st.plotly_chart(fig_tree, use_container_width=True)
            
        col3, col4 = st.columns([1, 1])
        # Segment Revenue
        with col3:
            seg_rev = customer_agg.groupby('Segment')['Total_Revenue'].sum().reset_index()
            fig_bar = px.bar(seg_rev, x='Segment', y='Total_Revenue', title="Revenue by Segment", color='Segment',
                             color_discrete_sequence=px.colors.sequential.Teal)
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#EEEEEE')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # Segment Table
        with col4:
            st.markdown("##### Segment Summary Table")
            summary_table = customer_agg.groupby('Segment').agg(
                Customers=('Customer_ID', 'count'),
                Avg_Recency=('Recency', 'mean'),
                Avg_Freq=('Tx_Count', 'mean'),
                Total_Revenue=('Total_Revenue', 'sum')
            ).reset_index()
            st.dataframe(summary_table.style.format({
                'Total_Revenue': lambda x: format_inr(x),
                'Avg_Recency': '{:.1f}',
                'Avg_Freq': '{:.1f}'
            }), use_container_width=True, hide_index=True)
            
    else:
        st.info("No RFM data available.")


# ----------------------------------------------------
# TAB 5: ABC INVENTORY ANALYSIS
# ----------------------------------------------------
with tab5:
    st.markdown(f"<h3 style='color: {text_primary}; margin-bottom: 5px;'>🏷️ ABC Inventory Analysis</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {text_muted}; font-size: 0.85rem; margin-bottom: 20px;'>Categorises SKUs into Class A (top 70% revenue), Class B (next 20%), and Class C (bottom 10%) using cumulative Pareto logic.</p>", unsafe_allow_html=True)

    abc_df = compute_abc(sku_agg)

    if len(abc_df) > 0:
        # ── Pareto Chart ──
        top_abc = abc_df.head(60)  # limit for readability
        fig_pareto = go.Figure()
        # Bar: individual SKU revenue
        fig_pareto.add_trace(go.Bar(
            x=top_abc['SKU'].astype(str),
            y=top_abc['Total_Revenue'],
            name='Revenue',
            marker_color=top_abc['ABC_Class'].map({'A': '#00ADB5', 'B': '#F6C90E', 'C': '#F87171'}),
            hovertemplate='<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>'
        ))
        # Line: cumulative %
        fig_pareto.add_trace(go.Scatter(
            x=top_abc['SKU'].astype(str),
            y=top_abc['Cum_Pct'],
            name='Cumulative %',
            yaxis='y2',
            line=dict(color='#EEEEEE', width=2, dash='dot'),
            hovertemplate='%{y:.1f}%<extra></extra>'
        ))
        fig_pareto.add_hline(y=70, line_dash='dash', line_color='#00ADB5',
                             annotation_text='70% — Class A cutoff', yref='y2',
                             annotation_font_color='#00ADB5')
        fig_pareto.add_hline(y=90, line_dash='dash', line_color='#F6C90E',
                             annotation_text='90% — Class B cutoff', yref='y2',
                             annotation_font_color='#F6C90E')
        fig_pareto.update_layout(
            title='Pareto Chart — SKU Revenue with Cumulative % (top 60 SKUs)',
            yaxis=dict(title='Revenue (₹)', color=text_primary),
            yaxis2=dict(title='Cumulative Revenue %', overlaying='y', side='right',
                        range=[0, 105], color=text_primary),
            xaxis=dict(showticklabels=False, title='SKUs (sorted by revenue)'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(57,62,70,0.15)',
            font_color=text_primary, legend=dict(orientation='h', y=1.08)
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        # ── ABC Class Summary ──
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"<h4 style='color: {text_primary};'>📊 Class Summary</h4>", unsafe_allow_html=True)
            abc_summary = abc_df.groupby('ABC_Class').agg(
                SKU_Count=('SKU', 'count'),
                Total_Revenue=('Total_Revenue', 'sum'),
                Total_Qty=('Total_Quantity', 'sum')
            ).reindex(['A', 'B', 'C']).reset_index()
            abc_summary['Revenue_Share_%'] = (abc_summary['Total_Revenue'] / abc_summary['Total_Revenue'].sum() * 100).round(1)
            abc_summary['SKU_Share_%'] = (abc_summary['SKU_Count'] / abc_summary['SKU_Count'].sum() * 100).round(1)
            st.dataframe(abc_summary.style.format({
                'Total_Revenue': lambda x: format_inr(x),
                'Total_Qty': '{:,}',
                'Revenue_Share_%': '{:.1f}%',
                'SKU_Share_%': '{:.1f}%'
            }), use_container_width=True, hide_index=True)
        with col2:
            st.markdown(f"<h4 style='color: {text_primary};'>🏷️ Class Distribution</h4>", unsafe_allow_html=True)
            fig_abc_pie = px.pie(abc_summary, values='SKU_Count', names='ABC_Class',
                                 color='ABC_Class',
                                 color_discrete_map={'A': '#00ADB5', 'B': '#F6C90E', 'C': '#F87171'},
                                 hole=0.45, title='SKUs by ABC Class')
            fig_abc_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=text_primary)
            st.plotly_chart(fig_abc_pie, use_container_width=True)

        # ── SKU table filterable by class ──
        st.markdown(f"<h4 style='color: {text_primary}; margin-top: 10px;'>🗂️ Full SKU Classification Table</h4>", unsafe_allow_html=True)
        selected_class = st.selectbox("Filter by class", ['All', 'A', 'B', 'C'], key='abc_class_filter')
        view_df = abc_df if selected_class == 'All' else abc_df[abc_df['ABC_Class'] == selected_class]
        st.dataframe(
            view_df[['SKU', 'Total_Revenue', 'Total_Quantity', 'Tx_Count', 'Cum_Pct', 'ABC_Class']]
            .rename(columns={'Total_Revenue': 'Revenue (₹)', 'Total_Quantity': 'Qty',
                             'Tx_Count': 'Transactions', 'Cum_Pct': 'Cumulative %', 'ABC_Class': 'Class'})
            .style.format({'Revenue (₹)': lambda x: format_inr(x), 'Qty': '{:,}',
                           'Transactions': '{:,}', 'Cumulative %': '{:.1f}%'}),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No SKU data available for ABC analysis.")


# ----------------------------------------------------
# TAB 6: COHORT RETENTION ANALYSIS
# ----------------------------------------------------
with tab6:
    st.markdown(f"<h3 style='color: {text_primary}; margin-bottom: 5px;'>🔁 Cohort Retention Analysis</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {text_muted}; font-size: 0.85rem; margin-bottom: 20px;'>Groups customers by their first purchase month and tracks what % returned in each subsequent month. Reveals loyalty and churn patterns.</p>", unsafe_allow_html=True)

    with st.spinner('Computing cohort matrix...'):
        retention = compute_cohort(df_filtered)

    if retention is not None and len(retention) > 0:
        # Format index for display
        retention.index = retention.index.astype(str)
        retention.columns = [f'M+{c}' for c in retention.columns]

        fig_cohort, ax_cohort = plt.subplots(figsize=(14, max(5, len(retention) * 0.55)))
        fig_cohort.patch.set_facecolor(plot_bg)
        ax_cohort.set_facecolor(plot_bg)

        sns.heatmap(
            retention,
            annot=True, fmt='.0f',
            cmap='YlGnBu',
            linewidths=0.4,
            linecolor=(1.0, 1.0, 1.0, 0.05),
            cbar_kws={'label': 'Retention %'},
            ax=ax_cohort,
            vmin=0, vmax=100
        )
        ax_cohort.set_title('Customer Retention by Cohort Month (%)', color=text_primary,
                            fontsize=13, fontweight='bold', pad=15)
        ax_cohort.set_xlabel('Months Since First Purchase', color=text_primary, fontsize=10)
        ax_cohort.set_ylabel('Cohort (First Purchase Month)', color=text_primary, fontsize=10)
        ax_cohort.tick_params(colors=text_muted, labelsize=8)
        ax_cohort.figure.axes[-1].yaxis.label.set_color(text_primary)
        ax_cohort.figure.axes[-1].tick_params(colors=text_muted)
        plt.tight_layout()
        st.pyplot(fig_cohort)
        plt.close(fig_cohort)

        # Key insights
        st.markdown(f"<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        m1_col = 'M+1' if 'M+1' in retention.columns else None
        m3_col = 'M+3' if 'M+3' in retention.columns else None
        m6_col = 'M+6' if 'M+6' in retention.columns else None

        with col1:
            if m1_col:
                val = retention[m1_col].mean()
                st.markdown(f"""
                <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:12px; padding:18px; text-align:center;'>
                    <div style='color:{text_muted}; font-size:0.8rem; font-weight:600; text-transform:uppercase;'>Avg M+1 Retention</div>
                    <div style='color:{text_primary}; font-size:1.8rem; font-weight:700;'>{val:.1f}%</div>
                </div>""", unsafe_allow_html=True)
        with col2:
            if m3_col:
                val = retention[m3_col].mean()
                st.markdown(f"""
                <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:12px; padding:18px; text-align:center;'>
                    <div style='color:{text_muted}; font-size:0.8rem; font-weight:600; text-transform:uppercase;'>Avg M+3 Retention</div>
                    <div style='color:{text_primary}; font-size:1.8rem; font-weight:700;'>{val:.1f}%</div>
                </div>""", unsafe_allow_html=True)
        with col3:
            if m6_col:
                val = retention[m6_col].mean()
                st.markdown(f"""
                <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:12px; padding:18px; text-align:center;'>
                    <div style='color:{text_muted}; font-size:0.8rem; font-weight:600; text-transform:uppercase;'>Avg M+6 Retention</div>
                    <div style='color:{text_primary}; font-size:1.8rem; font-weight:700;'>{val:.1f}%</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Insufficient data to compute cohort retention.")


# ----------------------------------------------------
# TAB 7: MARKET BASKET ANALYSIS
# ----------------------------------------------------
with tab7:
    st.markdown(f"<h3 style='color: {text_primary}; margin-bottom: 5px;'>🛒 Market Basket Analysis</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {text_muted}; font-size: 0.85rem; margin-bottom: 20px;'>Identifies SKU pairs that are frequently bought together. Confidence A→B means: <i>'X% of customers who bought A also bought B on the same day.'</i></p>", unsafe_allow_html=True)

    top_n_pairs = st.slider("Number of SKU pairs to display", min_value=10, max_value=100, value=30, step=5, key='mba_topn')

    with st.spinner('Analysing basket co-occurrences...'):
        mba_df = compute_market_basket(df_filtered, top_n=top_n_pairs)

    if mba_df is not None and len(mba_df) > 0:
        col1, col2 = st.columns([1.6, 1])

        with col1:
            st.markdown(f"<h4 style='color: {text_primary};'>📋 Top SKU Pairs by Co-Occurrence</h4>", unsafe_allow_html=True)
            # Display styled pair rules
            for _, row in mba_df.head(20).iterrows():
                conf = max(row['Confidence_AtoB_%'], row['Confidence_BtoA_%'])
                color = '#4ade80' if conf >= 60 else ('#F6C90E' if conf >= 30 else '#F87171')
                st.markdown(f"""
                <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:10px;
                            padding:12px 18px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='color:{text_primary}; font-weight:700; font-size:0.95rem;'>{row['SKU_A']}</span>
                        <span style='color:{text_muted};'> + </span>
                        <span style='color:{text_primary}; font-weight:700; font-size:0.95rem;'>{row['SKU_B']}</span>
                    </div>
                    <div style='text-align:right;'>
                        <span style='color:{color}; font-weight:700; font-size:1rem;'>{conf:.0f}% confidence</span>
                        <br><span style='color:{text_muted}; font-size:0.75rem;'>{int(row['Co_Occurrences'])} co-occurrences · {row['Support_%']:.2f}% support</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<h4 style='color: {text_primary};'>📊 Full Pairs Table</h4>", unsafe_allow_html=True)
            st.dataframe(
                mba_df.rename(columns={
                    'SKU_A': 'SKU A', 'SKU_B': 'SKU B',
                    'Co_Occurrences': 'Co-Occur',
                    'Support_%': 'Support%',
                    'Confidence_AtoB_%': 'Conf A→B%',
                    'Confidence_BtoA_%': 'Conf B→A%'
                }),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("Not enough multi-item baskets found. Try removing SKU or Customer filters.")


# ----------------------------------------------------
# PDF EXPORT — Sidebar Download Button
# ----------------------------------------------------
st.sidebar.markdown("<hr style='border-color: rgba(57,62,70,0.5); margin: 1.5rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h4 style='color: {text_secondary}; font-size: 0.95rem; font-weight: 600; margin-bottom: 5px;'>📄 Executive Report</h4>", unsafe_allow_html=True)

if st.sidebar.button("📥 Download PDF Report", use_container_width=True, type='primary'):
    with st.spinner('Generating executive PDF report...'):
        top_skus_export = sku_agg.sort_values('Total_Revenue', ascending=False).head(10)
        top_cust_export = customer_agg.sort_values('Total_Revenue', ascending=False).head(10)

        rfm_export = None
        if 'Segment' in customer_agg.columns:
            rfm_export = customer_agg.groupby('Segment').agg(
                Customers=('Customer_ID', 'count'),
                Avg_Recency=('Recency', 'mean'),
                Avg_Freq=('Tx_Count', 'mean'),
                Total_Revenue=('Total_Revenue', 'sum')
            ).reset_index()

        abc_export = compute_abc(sku_agg)

        dr = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])) if isinstance(date_range, tuple) else None

        pdf_buffer = generate_pdf_report(
            kpi_rev=format_inr(kpi_revenue),
            kpi_qty=kpi_quantity,
            kpi_cust=kpi_customers,
            kpi_skus=kpi_skus,
            top_skus_df=top_skus_export,
            top_cust_df=top_cust_export,
            rfm_summary_df=rfm_export,
            abc_summary_df=abc_export,
            date_range=dr
        )

    st.sidebar.download_button(
        label="⬇️ Click to Download",
        data=pdf_buffer,
        file_name=f"BTAS_Report_{datetime.date.today().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
