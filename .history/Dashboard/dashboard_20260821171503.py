import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="E-Commerce Public Dataset Dashboard",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = os.path.join(
        BASE_DIR,
        '..',
        'E_Commerce_Public_Dataset'
    )

    orders_df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            'orders_dataset.csv'
        )
    )

    order_items_df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            'order_items_dataset.csv'
        )
    )

    products_df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            'products_dataset.csv'
        )
    )

    reviews_df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            'order_reviews_dataset.csv'
        )
    )

    # Konversi kolom tanggal
    date_columns = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for column in date_columns:
        orders_df[column] = pd.to_datetime(
            orders_df[column],
            errors='coerce'
        )

    return (
        orders_df,
        order_items_df,
        products_df,
        reviews_df
    )


orders_df, order_items_df, products_df, reviews_df = load_data()


# ============================================================
# DATA PROCESSING — PERTANYAAN 1
# ============================================================

q1_df = pd.merge(
    orders_df,
    order_items_df,
    on='order_id',
    how='inner'
)

q1_df = pd.merge(
    q1_df,
    products_df[
        [
            'product_id',
            'product_category_name'
        ]
    ],
    on='product_id',
    how='inner'
)


# Periode analisis
q1_df = q1_df[
    (q1_df['order_purchase_timestamp'] >= '2017-01-01') &
    (q1_df['order_purchase_timestamp'] < '2018-09-01')
].copy()


# Hapus kategori produk yang kosong
q1_df = q1_df[
    q1_df['product_category_name'].notna()
].copy()


# Nilai transaksi
q1_df['total_transaction'] = (
    q1_df['price'] +
    q1_df['freight_value']
)


# Bulan transaksi
q1_df['month'] = (
    q1_df['order_purchase_timestamp']
    .dt.to_period('M')
    .astype(str)
)


# ============================================================
# DATA PROCESSING — PERTANYAAN 2
# ============================================================

q2_orders_df = orders_df[
    orders_df['order_delivered_customer_date'].notna()
].copy()


q2_df = pd.merge(
    q2_orders_df,
    reviews_df[
        [
            'order_id',
            'review_score'
        ]
    ],
    on='order_id',
    how='inner'
)


# Periode analisis
q2_df = q2_df[
    (q2_df['order_purchase_timestamp'] >= '2017-01-01') &
    (q2_df['order_purchase_timestamp'] < '2018-09-01')
].copy()


# Selisih aktual dan estimasi pengiriman
q2_df['delivery_difference_days'] = (
    q2_df['order_delivered_customer_date']
    -
    q2_df['order_estimated_delivery_date']
).dt.total_seconds() / (24 * 60 * 60)


# Status pengiriman
q2_df['delivery_status'] = q2_df[
    'delivery_difference_days'
].apply(
    lambda x:
        'Tepat waktu / lebih cepat'
        if x <= 0
        else 'Terlambat'
)


# Kategori keterlambatan
q2_df['delay_category'] = pd.cut(
    q2_df['delivery_difference_days'],
    bins=[
        -float('inf'),
        0,
        3,
        7,
        14,
        float('inf')
    ],
    labels=[
        'Tepat waktu / lebih cepat',
        'Terlambat 1-3 hari',
        'Terlambat 4-7 hari',
        'Terlambat 8-14 hari',
        'Terlambat >14 hari'
    ]
)


# ============================================================
# DATA PROCESSING — RFM ANALYSIS
# ============================================================

rfm_df = pd.merge(
    orders_df,
    order_items_df,
    on='order_id',
    how='inner'
)


# Periode analisis
rfm_df = rfm_df[
    (rfm_df['order_purchase_timestamp'] >= '2017-01-01') &
    (rfm_df['order_purchase_timestamp'] < '2018-09-01')
].copy()


# Nilai transaksi
rfm_df['total_transaction'] = (
    rfm_df['price'] +
    rfm_df['freight_value']
)


# Tanggal acuan
reference_date = pd.Timestamp('2018-09-01')


# Menghitung RFM
rfm = rfm_df.groupby('customer_id').agg(

    Recency=(
        'order_purchase_timestamp',
        lambda x: (
            reference_date - x.max()
        ).days
    ),

    Frequency=(
        'order_id',
        'nunique'
    ),

    Monetary=(
        'total_transaction',
        'sum'
    )

).reset_index()


# ============================================================
# RFM SCORING
# ============================================================

rfm['R_score'] = pd.qcut(
    rfm['Recency'],
    q=5,
    labels=[5, 4, 3, 2, 1],
    duplicates='drop'
).astype(int)


rfm['F_score'] = pd.qcut(
    rfm['Frequency'].rank(method='first'),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)


rfm['M_score'] = pd.qcut(
    rfm['Monetary'].rank(method='first'),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

def rfm_segment(row):

    if (
        row['R_score'] >= 4 and
        row['F_score'] >= 4 and
        row['M_score'] >= 4
    ):
        return 'Best Customers'

    elif (
        row['F_score'] >= 4 and
        row['M_score'] >= 3
    ):
        return 'Loyal Customers'

    elif (
        row['R_score'] >= 4 and
        row['F_score'] <= 3
    ):
        return 'Potential Customers'

    elif row['R_score'] <= 2:
        return 'At Risk / Lost Customers'

    else:
        return 'Regular Customers'


rfm['customer_segment'] = rfm.apply(
    rfm_segment,
    axis=1
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛒 E-Commerce Dashboard")

st.sidebar.write(
    """
    Dashboard analisis E-Commerce Public Dataset
    periode Januari 2017 hingga Agustus 2018.
    """
)

st.sidebar.divider()

st.sidebar.subheader("Filter Periode")


min_date = q1_df[
    'order_purchase_timestamp'
].min().date()


max_date = q1_df[
    'order_purchase_timestamp'
].max().date()


date_range = st.sidebar.date_input(
    "Pilih periode transaksi:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


if len(date_range) == 2:

    start_date = pd.Timestamp(
        date_range[0]
    )

    end_date = (
        pd.Timestamp(date_range[1])
        +
        pd.Timedelta(days=1)
    )

    # Filter Q1
    filtered_q1 = q1_df[
        (q1_df['order_purchase_timestamp'] >= start_date) &
        (q1_df['order_purchase_timestamp'] < end_date)
    ].copy()

    # Filter Q2
    filtered_q2 = q2_df[
        (q2_df['order_purchase_timestamp'] >= start_date) &
        (q2_df['order_purchase_timestamp'] < end_date)
    ].copy()

else:

    filtered_q1 = q1_df.copy()
    filtered_q2 = q2_df.copy()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛒 E-Commerce Public Dataset Dashboard"
)

st.markdown(
    """
    Dashboard interaktif untuk menganalisis:

    - **Tren nilai transaksi**
    - **Kategori produk dengan kontribusi terbesar**
    - **Hubungan ketepatan waktu pengiriman dengan skor ulasan**
    - **Hubungan harga produk dengan ongkos kirim**
    - **Segmentasi pelanggan menggunakan RFM Analysis**
    """
)

st.divider()


# ============================================================
# KPI
# ============================================================

total_transaction = filtered_q1[
    'total_transaction'
].sum()


total_orders = filtered_q1[
    'order_id'
].nunique()


if total_orders > 0:

    average_transaction = (
        filtered_q1
        .groupby('order_id')
        ['total_transaction']
        .sum()
        .mean()
    )

else:

    average_transaction = 0


if not filtered_q1.empty:

    top_category = (
        filtered_q1
        .groupby('product_category_name')
        ['total_transaction']
        .sum()
        .idxmax()
    )

else:

    top_category = "-"


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Nilai Transaksi",
        f"R$ {total_transaction:,.0f}"
    )


with col2:

    st.metric(
        "Total Order",
        f"{total_orders:,}"
    )


with col3:

    st.metric(
        "Rata-rata Nilai / Order",
        f"R$ {average_transaction:,.0f}"
    )


with col4:

    st.metric(
        "Kategori Terbesar",
        top_category
    )


# ============================================================
# PERTANYAAN 1
# ============================================================

st.divider()

st.header(
    "📈 Analisis Nilai Transaksi"
)


# ------------------------------------------------------------
# Tren transaksi
# ------------------------------------------------------------

st.subheader(
    "Tren Total Nilai Transaksi Bulanan"
)


monthly_transaction = (
    filtered_q1
    .groupby('month')
    ['total_transaction']
    .sum()
    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(12, 5)
)


ax.plot(
    monthly_transaction['month'],
    monthly_transaction['total_transaction'],
    marker='o'
)


ax.set_xlabel(
    "Bulan"
)


ax.set_ylabel(
    "Nilai Transaksi (R$)"
)


ax.tick_params(
    axis='x',
    rotation=45
)


plt.tight_layout()

st.pyplot(fig)


# ------------------------------------------------------------
# Top kategori
# ------------------------------------------------------------

st.subheader(
    "🏆 10 Kategori Produk dengan Nilai Transaksi Terbesar"
)


category_transaction = (
    filtered_q1
    .groupby('product_category_name')
    ['total_transaction']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.barh(
    category_transaction.index,
    category_transaction.values
)


ax.set_xlabel(
    "Nilai Transaksi (R$)"
)


ax.set_ylabel(
    "Kategori Produk"
)


plt.tight_layout()

st.pyplot(fig)


# ------------------------------------------------------------
# Hubungan price dan freight
# ------------------------------------------------------------

st.subheader(
    "💰 Hubungan Harga Produk dan Ongkos Kirim"
)


fig, ax = plt.subplots(
    figsize=(10, 5)
)


ax.scatter(
    filtered_q1['price'],
    filtered_q1['freight_value'],
    alpha=0.3
)


ax.set_xlabel(
    "Harga Produk (R$)"
)


ax.set_ylabel(
    "Ongkos Kirim (R$)"
)


ax.set_title(
    "Hubungan Harga Produk dengan Ongkos Kirim"
)


plt.tight_layout()

st.pyplot(fig)


if len(filtered_q1) > 1:

    correlation = (
        filtered_q1['price']
        .corr(filtered_q1['freight_value'])
    )

    st.info(
        f"""
        Koefisien korelasi Pearson antara harga produk
        dan ongkos kirim adalah **{correlation:.2f}**.
        
        Nilai tersebut menunjukkan adanya hubungan positif
        dengan kekuatan sedang.
        """
    )


# ============================================================
# PERTANYAAN 2
# ============================================================

st.divider()

st.header(
    "🚚 Analisis Pengiriman & Kepuasan Pelanggan"
)


delivery_review = (
    filtered_q2
    .groupby('delivery_status')
    ['review_score']
    .agg(
        average_review='mean',
        total_reviews='count'
    )
    .reset_index()
)


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Ketepatan pengiriman
# ------------------------------------------------------------

with col1:

    st.subheader(
        "⭐ Review Berdasarkan Ketepatan Pengiriman"
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.bar(
        delivery_review['delivery_status'],
        delivery_review['average_review']
    )


    ax.set_ylabel(
        "Rata-rata Review"
    )


    ax.set_ylim(
        0,
        5
    )


    plt.xticks(
        rotation=15
    )


    plt.tight_layout()

    st.pyplot(fig)


# ------------------------------------------------------------
# Tingkat keterlambatan
# ------------------------------------------------------------

with col2:

    st.subheader(
        "⏱️ Review Berdasarkan Tingkat Keterlambatan"
    )


    delay_review = (
        filtered_q2
        .groupby(
            'delay_category',
            observed=False
        )['review_score']
        .mean()
        .reset_index()
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.bar(
        delay_review['delay_category'].astype(str),
        delay_review['review_score']
    )


    ax.set_ylabel(
        "Rata-rata Review"
    )


    ax.set_ylim(
        0,
        5
    )


    plt.xticks(
        rotation=25
    )


    plt.tight_layout()

    st.pyplot(fig)


# ------------------------------------------------------------
# Tabel ringkasan delivery
# ------------------------------------------------------------

st.subheader(
    "📋 Ringkasan Pengiriman dan Review"
)


delivery_summary = (
    filtered_q2
    .groupby(
        'delay_category',
        observed=False
    )
    .agg(
        average_review=(
            'review_score',
            'mean'
        ),
        total_reviews=(
            'review_score',
            'count'
        )
    )
    .reset_index()
)


delivery_summary.columns = [
    'Kategori Pengiriman',
    'Rata-rata Review',
    'Jumlah Review'
]


st.dataframe(
    delivery_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ANALISIS LANJUTAN — RFM
# ============================================================

st.divider()

st.header(
    "👥 Customer Segmentation — RFM Analysis"
)


st.markdown(
    """
    RFM Analysis digunakan untuk mengelompokkan pelanggan berdasarkan
    **Recency**, **Frequency**, dan **Monetary** selama periode analisis.
    """
)


# ------------------------------------------------------------
# KPI RFM
# ------------------------------------------------------------

segment_counts = (
    rfm['customer_segment']
    .value_counts()
)


col1, col2, col3, col4, col5 = st.columns(5)


segments = [
    'Best Customers',
    'Loyal Customers',
    'Potential Customers',
    'Regular Customers',
    'At Risk / Lost Customers'
]


columns = [
    col1,
    col2,
    col3,
    col4,
    col5
]


for column, segment in zip(
    columns,
    segments
):

    with column:

        st.metric(
            segment,
            f"{segment_counts.get(segment, 0):,}"
        )


# ------------------------------------------------------------
# Distribusi pelanggan
# ------------------------------------------------------------

st.subheader(
    "Distribusi Segmen Pelanggan"
)


segment_summary = (
    rfm['customer_segment']
    .value_counts()
    .reset_index()
)


segment_summary.columns = [
    'customer_segment',
    'customer_count'
]


fig, ax = plt.subplots(
    figsize=(10, 5)
)


ax.bar(
    segment_summary['customer_segment'],
    segment_summary['customer_count']
)


ax.set_xlabel(
    "Segmen Pelanggan"
)


ax.set_ylabel(
    "Jumlah Pelanggan"
)


plt.xticks(
    rotation=20
)


plt.tight_layout()

st.pyplot(fig)


# ------------------------------------------------------------
# Karakteristik setiap segmen
# ------------------------------------------------------------

st.subheader(
    "Karakteristik Segmen Pelanggan"
)


rfm_summary = (
    rfm
    .groupby('customer_segment')
    .agg(
        customer_count=(
            'customer_id',
            'count'
        ),

        avg_recency=(
            'Recency',
            'mean'
        ),

        avg_frequency=(
            'Frequency',
            'mean'
        ),

        avg_monetary=(
            'Monetary',
            'mean'
        )
    )
    .sort_values(
        'avg_monetary',
        ascending=False
    )
    .reset_index()
)


rfm_summary.columns = [
    'Segmen Pelanggan',
    'Jumlah Pelanggan',
    'Rata-rata Recency (hari)',
    'Rata-rata Frequency',
    'Rata-rata Monetary (R$)'
]


st.dataframe(
    rfm_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RINGKASAN KATEGORI
# ============================================================

st.divider()

st.subheader(
    "📊 Ringkasan Kategori Produk"
)


category_summary = (
    filtered_q1
    .groupby('product_category_name')
    ['total_transaction']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)


category_summary.columns = [
    'Kategori Produk',
    'Total Nilai Transaksi'
]


st.dataframe(
    category_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "E-Commerce Public Dataset | Muhammad Nur Fattah | Dicoding"
)