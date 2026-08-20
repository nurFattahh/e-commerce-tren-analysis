import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


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

    orders_df = pd.read_csv(
        '../E_Commerce_Public_Dataset/orders_dataset.csv'
    )

    order_items_df = pd.read_csv(
        '../E_Commerce_Public_Dataset/order_items_dataset.csv'
    )

    products_df = pd.read_csv(
        '../E_Commerce_Public_Dataset/products_dataset.csv'
    )

    reviews_df = pd.read_csv(
        '../E_Commerce_Public_Dataset/order_reviews_dataset.csv'
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
    products_df[['product_id', 'product_category_name']],
    on='product_id',
    how='inner'
)

# Periode analisis:
# Januari 2017 - Agustus 2018

q1_df = q1_df[
    (q1_df['order_purchase_timestamp'] >= '2017-01-01') &
    (q1_df['order_purchase_timestamp'] < '2018-09-01')
].copy()

# Nilai transaksi = price + freight_value
q1_df['total_transaction'] = (
    q1_df['price'] +
    q1_df['freight_value']
)

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
    reviews_df[['order_id', 'review_score']],
    on='order_id',
    how='inner'
)

# Pastikan periode sama dengan pertanyaan bisnis
q2_df = q2_df[
    (q2_df['order_purchase_timestamp'] >= '2017-01-01') &
    (q2_df['order_purchase_timestamp'] < '2018-09-01')
].copy()

# Menghitung selisih tanggal aktual dengan estimasi
q2_df['delivery_difference_days'] = (
    q2_df['order_delivered_customer_date']
    - q2_df['order_estimated_delivery_date']
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
    bins=[-float('inf'), 0, 3, 7, 14, float('inf')],
    labels=[
        'Tepat waktu / lebih cepat',
        'Terlambat 1-3 hari',
        'Terlambat 4-7 hari',
        'Terlambat 8-14 hari',
        'Terlambat >14 hari'
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛒 E-Commerce Dashboard")

st.sidebar.write(
    "Dashboard analisis E-Commerce Public Dataset "
    "periode Januari 2017 hingga Agustus 2018."
)

st.sidebar.divider()

st.sidebar.subheader("Filter Periode")

min_date = q1_df['order_purchase_timestamp'].min().date()
max_date = q1_df['order_purchase_timestamp'].max().date()

date_range = st.sidebar.date_input(
    "Pilih periode transaksi:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:

    start_date = pd.Timestamp(date_range[0])
    end_date = (
        pd.Timestamp(date_range[1])
        + pd.Timedelta(days=1)
    )

    filtered_q1 = q1_df[
        (q1_df['order_purchase_timestamp'] >= start_date) &
        (q1_df['order_purchase_timestamp'] < end_date)
    ].copy()

else:

    filtered_q1 = q1_df.copy()


# ============================================================
# HEADER
# ============================================================

st.title("🛒 E-Commerce Public Dataset Dashboard")

st.markdown(
    """
    Dashboard interaktif untuk menganalisis **nilai transaksi,
    kategori produk, serta hubungan ketepatan waktu pengiriman
    dengan skor ulasan pelanggan**.
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

average_transaction = (
    filtered_q1
    .groupby('order_id')['total_transaction']
    .sum()
    .mean()
)

top_category = (
    filtered_q1
    .groupby('product_category_name')
    ['total_transaction']
    .sum()
    .idxmax()
)


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
# TREN NILAI TRANSAKSI
# ============================================================

st.subheader("📈 Tren Total Nilai Transaksi Bulanan")

monthly_transaction = (
    filtered_q1
    .groupby('month')['total_transaction']
    .sum()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    monthly_transaction['month'],
    monthly_transaction['total_transaction'],
    marker='o'
)

ax.set_xlabel("Bulan")
ax.set_ylabel("Nilai Transaksi (R$)")
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# TOP 10 KATEGORI
# ============================================================

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

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    category_transaction.index,
    category_transaction.values
)

ax.set_xlabel("Nilai Transaksi (R$)")
ax.set_ylabel("Kategori Produk")

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# PENGIRIMAN & REVIEW
# ============================================================

st.divider()

st.header("🚚 Pengiriman & Kepuasan Pelanggan")

delivery_review = (
    q2_df
    .groupby('delivery_status')['review_score']
    .agg(
        average_review='mean',
        total_reviews='count'
    )
    .reset_index()
)


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Tepat waktu vs terlambat
# ------------------------------------------------------------

with col1:

    st.subheader(
        "⭐ Review Berdasarkan Ketepatan Pengiriman"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        delivery_review['delivery_status'],
        delivery_review['average_review']
    )

    ax.set_ylabel("Rata-rata Review")
    ax.set_ylim(0, 5)

    plt.xticks(rotation=15)

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
        q2_df
        .groupby(
            'delay_category',
            observed=False
        )['review_score']
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        delay_review['delay_category'].astype(str),
        delay_review['review_score']
    )

    ax.set_ylabel("Rata-rata Review")
    ax.set_ylim(0, 5)

    plt.xticks(rotation=25)

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# TABEL RINGKASAN
# ============================================================

st.divider()

st.subheader("📊 Ringkasan Kategori Produk")

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