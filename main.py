import streamlit as st
import pandas as pd
import pickle
import requests

# --- 1. PENGATURAN HALAMAN & KUNCI API ---
st.set_page_config(layout="wide")

# Ambil API key dari Streamlit Secrets
try:
    API_KEY = st.secrets["tmdb_api_key"]
except FileNotFoundError:
    st.error("File .streamlit/secrets.toml tidak ditemukan.")
    API_KEY = "" 
except KeyError:
    st.error("Pastikan 'tmdb_api_key' ada di file secrets.toml Anda.")
    API_KEY = ""

# --- 2. FUNGSI LOAD DATA (VERSI CEPAT / PICKLE) ---
@st.cache_data
def load_data_complete():
    """Memuat data lengkap untuk halaman Info."""
    try:
        return pd.read_pickle('movies_complete.pkl')
    except FileNotFoundError:
        st.error("File 'movies_complete.pkl' tidak ditemukan. Jalankan setup_data.py dulu!")
        return None

@st.cache_data
def load_data_recommender():
    """Memuat data ringkas dan similarity matrix untuk rekomendasi."""
    try:
        new_df = pd.read_pickle('movies_clean.pkl')
        with open('similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
        return new_df, similarity
    except FileNotFoundError:
        st.error("File pickle tidak ditemukan. Jalankan setup_data.py dulu!")
        return None, None

# --- 3. FUNGSI LOGIKA ---

@st.cache_data
def fetch_poster(movie_id):
    """Mengambil URL poster dari API TMDB."""
    if not API_KEY:
        return "https://via.placeholder.com/500x750.png?text=API+Key+Error"
        
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750.png?text=No+Poster"
    except requests.exceptions.RequestException:
        return "https://via.placeholder.com/500x750.png?text=API+Fetch+Error"

@st.cache_data
def recommend(movie_title, new_df, similarity):
    """Memberikan rekomendasi film."""
    if movie_title not in new_df['title'].values:
        return [], []
    
    movie_index = new_df[new_df['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])
    
    recommended_movies = []
    recommended_posters = []
    
    # Ambil 10 film (index 1-11)
    for i in distances[1:11]:
        movie_id = new_df.iloc[i[0]].movie_id
        recommended_movies.append(new_df.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
        
    return recommended_movies, recommended_posters
# --- 3. MEMUAT DATA UTAMA ---
# Load data langsung dari file yang sudah diproses
movies_data = load_data_complete()
new_df, similarity = load_data_recommender()


# --- 4. MEMBUAT SIDEBAR & KONTEN ---
# ... (Sisa kode ke bawah tetap sama) ...
# --- 4. MEMBUAT SIDEBAR ---
st.sidebar.title("Navigasi Aplikasi") 

pilihan_halaman = st.sidebar.radio(
    "Pilih Halaman:",
    (
        "👋 Home page",
        "🔭 General info",
        "🎬 Rekomendasi Film",
        "🎨 Filters",
        "📈 characteristics",
        "👁️ FOV",
        "🌌 Galaxy",
        "✨ Galaxy field",
        "🪞 Mirror",
        "🗺️ Survey footprint"
    ),
    index=0  # Mulai dari Home page
)

# --- 5. KONTEN HALAMAN (BERDASARKAN PILIHAN SIDEBAR) ---

if pilihan_halaman == "👋 Home page":
    st.title("Selamat Datang di Halaman Utama 👋")
    st.write("Home page Aditya, Reza dan Rifda.")
    st.write("Silakan pilih halaman lain dari sidebar di sebelah kiri.")

elif pilihan_halaman == "🔭 General info":
    st.title("Informasi Umum Film 🔭")
    st.write("Cari detail film, sinopsis, dan lihat posternya di sini.")
    
    # Cek apakah data sudah siap
    if movies_data is not None:
        
        movie_list = movies_data['title'].values
        selected_title_info = st.selectbox(
            "Pilih judul film untuk melihat detailnya:",
            movie_list
        )
        
        if selected_title_info:
            # Ambil baris data film
            selected_row = movies_data.loc[movies_data['title'] == selected_title_info].iloc[0]
            
            # Tampilkan Poster dan Judul
            col_info_1, col_info_2 = st.columns([1, 3]) # Membagi layout jadi 2 kolom
            
            with col_info_1:
                st.image(fetch_poster(selected_row['movie_id']), width=200)
            
            with col_info_2:
                st.subheader(selected_title_info)
                st.markdown(f"**Sinopsis:**")
                st.write(selected_row['overview'])

                # --- PERBAIKAN DISINI ---
                # Karena data dari pickle sudah berbentuk List, kita langsung join saja.
                # Tidak perlu memanggil fungsi convert() lagi.
                
                # Genre
                genres = selected_row['genres'] 
                if isinstance(genres, list):
                    st.write(f"**Genre:** {', '.join(genres)}")
                else:
                    st.write(f"**Genre:** {genres}")

                # Cast (Pemeran)
                cast = selected_row['cast']
                if isinstance(cast, list):
                    st.write(f"**Pemeran:** {', '.join(cast[:3])}") # Ambil 3 nama pertama
                
                # Director (Sutradara)
                # Di setup_data, kolom 'crew' sudah kita ubah isinya jadi nama sutradara
                director = selected_row['crew']
                if isinstance(director, list):
                     st.write(f"**Sutradara:** {', '.join(director)}")

    else:
        st.error("Data film tidak berhasil dimuat. Pastikan setup_data.py sudah dijalankan.")

elif pilihan_halaman == "🎬 Rekomendasi Film":
    st.title('Rekomendasi Film 🎬')
    
    # Cek apakah data berhasil dimuat
    if new_df is not None and similarity is not None:
        st.write('Pilih film yang Anda suka, dan kami akan merekomendasikan 5 film yang mirip!')

        # Buat daftar judul film untuk dropdown
        movie_list = new_df['title'].values
        
        selected_movie = st.selectbox(
            'Pilih film:',
            movie_list
        )

        if st.button('Cari Rekomendasi'):
            st.spinner("Mencari film yang mirip...")
            # Panggil fungsi recommend dengan data yang sudah siap
            names, posters = recommend(selected_movie, new_df, similarity)
            
            if names:
                st.subheader(f"Rekomendasi untuk '{selected_movie}':")
                
                # Tampilkan 5 rekomendasi dalam 5 kolom
                cols_baris_1 = st.columns(5,gap="large")
                
                # Gunakan loop untuk mengisi 5 kolom pertama
                for i in range(5):
                    with cols_baris_1[i]:
                        st.image(posters[i])
                        st.caption(names[i])
                
                # --- BARIS KEDUA (Film 6-10) ---
                # Buat 5 kolom baru untuk baris kedua
                cols_baris_2 = st.columns(5,gap="large")
                
                # Gunakan loop untuk mengisi 5 kolom berikutnya
                for i in range(5):
                    with cols_baris_2[i]:
                        st.image(posters[i + 5])  # <-- Perhatikan: posters[5], posters[6], dst
                        st.caption(names[i + 5])
            else:
                st.error("Film tidak ditemukan dalam data kami.")
    else:
        st.error("Data film tidak berhasil dimuat. Periksa apakah file CSV Anda ada di folder yang benar.")

elif pilihan_halaman == "🎨 Filters":
    st.title("Halaman Filters 🎨")
    st.write("Ini adalah konten untuk halaman 'Filters'.")
    st.info("Anda sedang melihat halaman ini, persis seperti di contoh gambar!")
    st.slider("Contoh slider filter", 0, 100, 50)
    st.multiselect("Contoh filter multi-pilih", ["A", "B", "C"])

elif pilihan_halaman == "📈 characteristics":
    st.title("Halaman Characteristics 📈")
    st.write("Ini adalah konten untuk 'characteristics'.")

# ... dan seterusnya untuk setiap halaman ...

else:
    # Halaman default jika Anda belum membuat 'if'-nya
    st.title(pilihan_halaman)
    st.write(f"Konten untuk {pilihan_halaman} belum dibuat.")