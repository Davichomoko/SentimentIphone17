# app.py
# ========================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
try:
    from wordcloud import WordCloud, STOPWORDS
except Exception:
    # If wordcloud is not installed on the host, avoid import-time crash.
    WordCloud = None
    STOPWORDS = set()
from collections import Counter

# ========================================================
# Konstanta dan fungsi tambahan

DEFAULT_SAVE_DIR = os.getenv("SAVE_DIR", "data/")
DEFAULT_DATA_PREP = os.path.join("data", "data_preprocessing.csv")
DEFAULT_DATA_LABEL = os.path.join("data", "data_labeling.csv")

_stopwords_streamlit = set(STOPWORDS)

def load_dataframe(path):
    """Fungsi aman untuk membaca file CSV dengan fallback encoding."""
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception as e:
            st.error(f"Gagal memuat file {path}: {e}")
            return pd.DataFrame()


# ========================================================
# Konfigurasi halaman

icon = Image.open("assets/logo.png")

st.set_page_config(
    page_title="Dashboard Sentimen IPhone 17",
    page_icon=icon,
    layout="wide"
)


# ========================================================
# UI & Main
col1, col2 = st.columns([0.08, 0.9])

with col1:
    st.image(icon, width=100)

with col2:
    st.markdown(
        "<h1 style='margin-top: 0.3em; padding-top: 0;'>Dashboard Sentimen IPhone 17</h1>",
        unsafe_allow_html=True
    )

# Tabs
_tab1, _tab2, _tab3 = st.tabs(["Home", "Eksplorasi Dataset", "About"])

# ========================================================
# ---------------- Tab 1: Home ----------------
with _tab1:
    st.markdown(r"""
        Dashboard ini dibuat untuk menampilkan hasil analisis sentimen publik terhadap peluncuran iPhone 17 berdasarkan komentar dari kanal YouTube GadgetIn. 
        Tujuannya adalah untuk mengetahui bagaimana tanggapan masyarakat Indonesia terhadap produk terbaru Apple melalui pengelompokan komentar menjadi tiga kategori sentimen, yaitu positif, netral, dan negatif.
        
        Analisis dilakukan menggunakan pendekatan Natural Language Processing (NLP) dengan membandingkan dua model utama: model tradisional berbasis Logistic Regression dengan TF-IDF, dan fine-tuning IndoBERT yang merupakan pengembangan dari arsitektur transformer untuk Bahasa Indonesia.
        
        Melalui dashboard ini, pengguna dapat mengunggah data komentar, menjalankan prediksi sentimen, melatih ulang model, serta melihat visualisasi hasil analisis seperti distribusi label dan kata-kata yang paling berpengaruh. 
        Proyek ini bertujuan untuk memberikan gambaran umum mengenai opini publik dan mendukung pengambilan keputusan berdasarkan data.
        """)
   
    st.subheader("**Sumber Data**")
    youtube_url = "https://youtu.be/NhNpGUtcdf8?si=cXWY1Qp1kDx0n94R"
    try:
        st.video(youtube_url)
    except Exception:
        st.markdown(f"**Video sumber analisis:** [{youtube_url}]({youtube_url})")
        
    st.markdown("---")
    st.subheader("Perbandingan Akurasi Model (Before vs After SMOTE)")

    accuracy_data = {
        "Logistic Regression": {"Before SMOTE": 0.71, "After SMOTE": 0.78},
        "Decision Tree": {"Before SMOTE": 0.80, "After SMOTE": 0.78},
        "Random Forest": {"Before SMOTE": 0.75, "After SMOTE": 0.76},
        "Support Vector Machine": {"Before SMOTE": 0.71, "After SMOTE": 0.72},
        "K-Nearest Neighbors": {"Before SMOTE": 0.53, "After SMOTE": 0.57},
        "Naive Bayes": {"Before SMOTE": 0.59, "After SMOTE": 0.71}
    }

    df_acc = pd.DataFrame(accuracy_data).T

    classifiers = list(accuracy_data.keys())
    before = [v["Before SMOTE"] for v in accuracy_data.values()]
    after = [v["After SMOTE"] for v in accuracy_data.values()]

    x = np.arange(len(classifiers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, before, width, label='Before SMOTE')
    bars2 = ax.bar(x + width/2, after, width, label='After SMOTE')

    for bar1, bar2 in zip(bars1, bars2):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                f'{bar1.get_height():.2f}', ha='center', fontsize=9, color='black')
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                f'{bar2.get_height():.2f}', ha='center', fontsize=9, color='black')

    ax.set_xlabel("Classifier")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classifier Accuracy: Before vs After SMOTE")
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()

    st.pyplot(fig)
    
    st.markdown(r"""
        Berdasarkan grafik perbandingan di atas menunjukan bahwa model Logistic Regression menjadi salah satu akurasi tertinggi
        setelah dilakukan SMOTE (membuat setiap data pada label sama banyaknya) dibanding dengan model Machine Learning lainnya.
        """)
    
    st.markdown("---")
    st.subheader("Akurasi Model IndoBERT vs Logistic Regression")

    comp_df = pd.DataFrame({
        "Model": ["IndoBERT", "Logistic Regression"],
        "Macro-F1": [0.64, 0.71]
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(comp_df["Model"], comp_df["Macro-F1"], color=["#2b8cbe", "#f03b20"], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.2f}', xy=(b.get_x() + b.get_width() / 2, h),
                   xytext=(0, 6), textcoords="offset points",
                   ha='center', va='bottom')
    
    ax.set_title("Macro-F1 Comparison")
    fig.tight_layout()
    st.pyplot(fig)
    
    st.markdown(r"""
        Berdasarkan grafik perbandingan di atas menunjukan bahwa model Logistic Regression memiliki akurasi lebih tinggi
        (71%) dibandingkan dengan model fine-tuning IndoBERT (64%).
        """)


# ========================================================
# ---------------- Tab 2: Eksplorasi Dataset ----------------
with _tab2:
    st.subheader("Eksplorasi Dataset")
    src = st.radio(
        "Sumber data untuk eksplorasi:",
        ["data_preprocessing.csv", "data_labelingIndoBERT.csv", "data_labelingTradisional.csv", "Unggah CSV"],
        horizontal=True,
    )

    if src == "Unggah CSV":
        up = st.file_uploader("Unggah file CSV", type=["csv"])
        if up is not None:
            df_x = load_dataframe(up)
        else:
            df_x = pd.DataFrame()
    else:
        file_map = {
            "data_preprocessing.csv": os.path.join("data", "data_preprocessing.csv"),
            "data_labelingIndoBERT.csv": os.path.join("data", "data_labelingIndoBERT.csv"),
            "data_labelingTradisional.csv": os.path.join("data", "data_labelingTradisional.csv"),
        }
        path = file_map.get(src)
        df_x = load_dataframe(path)

    if df_x.empty:
        st.info("Tidak ada data untuk ditampilkan. Pastikan file ada atau unggah CSV.")
    else:
        st.dataframe(df_x.head(200), use_container_width=True)
        cols = df_x.columns.tolist()
        default_idx = 0
        for pref in ["final_clean", "steming_data", "cleaning", "Comment", "comment", "processed", "text"]:
            if pref in cols:
                default_idx = cols.index(pref)
                break
        text_col = st.selectbox("Pilih kolom teks", options=cols, index=default_idx)

        show_text_exploration = src not in ("data_labelingIndoBERT.csv", "data_labelingTradisional.csv")

        all_text = ""
        if show_text_exploration:
            with st.expander("WordCloud"):
                all_text = " ".join(df_x[text_col].astype(str).tolist()).strip()
                if not all_text:
                    st.info("Tidak ada teks untuk dibuat WordCloud.")
                else:
                    if WordCloud is None:
                        st.warning("Paket 'wordcloud' tidak tersedia — WordCloud tidak dapat dibuat. Tambahkan 'wordcloud' ke requirements.txt lalu deploy ulang.")
                    else:
                        try:
                            wc_obj = WordCloud(stopwords=_stopwords_streamlit, background_color="white", width=1000, height=400).generate(all_text)
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.imshow(wc_obj, interpolation="bilinear")
                            ax.axis("off")
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Gagal membuat WordCloud: {e}")

            with st.expander("Frekuensi Kata Terbanyak"):
                tokens = [w for w in all_text.split() if w not in _stopwords_streamlit]
                cnt = Counter(tokens)
                topk = st.slider("Top-K", 5, 40, 15)
                items = cnt.most_common(topk)
                if items:
                    words, counts = zip(*items)
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    ax2.bar(words, counts)
                    ax2.set_xticklabels(words, rotation=45, ha="right")
                    ax2.set_title("Frekuensi Kata")
                    st.pyplot(fig2)
                    plt.close(fig2)
        
        if "label" in df_x.columns:
            with st.expander("Distribusi Label"):
                vc = df_x["label"].value_counts()
                fig3, ax3 = plt.subplots(figsize=(5, 5))
                ax3.pie(vc.values, labels=vc.index, autopct="%1.1f%%", startangle=90)
                ax3.axis("equal")
                st.pyplot(fig3)
                plt.close(fig3)

# ========================================================
# ---------------- Tab 3: About ----------------
with _tab3:
    st.markdown(r"""
    Aplikasi ini adalah bagian dari proyek **Forum Group Discussion (FGD)** yang dikerjakan oleh Asisten Mata Kuliah Praktikum Unggulan (DGX) Universitas Gunadarma pada Semester Ganjil Tahun Akademik 2025/2026.            

    ### Tim Pengembang:
    **Divisi Riset**
    - Irsyad Nur Candra Putra  
    - Muhammad Fitrah Pratama  
    - Faizah Rizki Auliawati

    **Divisi Kompetisi**
    - Inanda Kalsa Ratnamaya  
    - Davicho Wiediatmoko  
    - Nazwa Akilla Zahra

    **Divisi Desain Kreatif**
    - Almirah Laksita Chandrawati  
    - Muhammad Naufal

    **Pembimbing**
    - Prof. Dr. Detty Purnamasari, S.Kom., MMSI., M.I.Kom.  
    - Ulfa Hidayati, S.T., MMSI.  
    - Milda Safrila Oktiana, S.Kom., MMSI.  
    - Fanka Arie Reza, S.Kom.  
    - Mario Mora Siregar, S.Kom.  
    - Felix Hardyan, S.Kom.
    """)
