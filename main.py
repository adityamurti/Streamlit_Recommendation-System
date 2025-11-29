import streamlit as st
import pandas as pd
import pickle
import requests

# --- 1. PENGATURAN HALAMAN & KUNCI API ---
st.set_page_config(layout="wide")

# Ambil API key dari Streamlit Secrets
API_KEY = st.secrets["tmdb_api_key"]


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

@st.cache_data
def load_knn_data(): # <--- 1. Nama fungsi diganti
    """Memuat data lengkap untuk KNN Filtering."""
    
    # PASTIKAN ANDA SUDAH MENGUBAH NAMA FILE DI FOLDER ANDA MENJADI 'knn_data.pkl'
    with open('knn_data.pkl', 'rb') as f: # <--- 2. Nama file diganti
        data = pickle.load(f)
    return data 
    

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

# --- MASUKKAN DI BAGIAN FUNGSI LOGIKA ---
# --- FUNGSI LOGIKA (VERSI PERBAIKAN) ---

def get_similar_movies(movie_id, data, n_recommendations=5):
    """
    Mencari ID film mirip menggunakan Sparse Matrix & KNN.
    Input & Output adalah MOVIE ID (Integer), bukan Judul.
    """
    model = data['model_knn']
    sparse_mat = data['sparse_matrix']
    movie_to_idx = data['movie_to_idx'] # Mapping ID -> Index
    idx_to_movie = data['idx_to_movie'] # Mapping Index -> ID

    # Cek menggunakan ID, bukan Judul
    if movie_id not in movie_to_idx:
        return []

    movie_idx = movie_to_idx[movie_id]
    
    # Hitung jarak
    distances, indices = model.kneighbors(sparse_mat[movie_idx].reshape(1, -1), n_neighbors=n_recommendations+1)

    recommendations = []
    for i in range(1, len(distances.flatten())):
        idx = indices.flatten()[i]
        try:
            # Kembalikan ID film (Integer)
            rec_id = idx_to_movie[idx]
            recommendations.append(rec_id)
        except KeyError:
            continue
            
    return recommendations

@st.cache_data(show_spinner=False)
def recommend_for_user(user_id, _knn_data, _movie_data):
    """
    Rekomendasi User dengan konversi ID -> Judul.
    _movie_data diperlukan untuk mengambil judul film dari ID.
    """
    df_ratings = _knn_data['df_ratings_users']
    
    # 1. Filter User
    user_history = df_ratings[df_ratings['user'] == user_id]
    
    if user_history.empty:
        return [], []

    # 2. Ambil ID film dengan rating tertinggi
    # PERBAIKAN: Gunakan kolom 'final_rating' dan 'id'
    top_liked_ids = user_history.sort_values(by='final_rating', ascending=False)['id'].head(5).tolist()
    
    if not top_liked_ids:
        return [], []
    
    watched_ids = set(user_history['id'].values)
    candidates = {} 
    
    # 3. Cari rekomendasi (berbasis ID)
    for m_id in top_liked_ids:
        recs = get_similar_movies(m_id,_knn_data, n_recommendations=5)
        for rec_id in recs:
            if rec_id not in watched_ids:
                candidates[rec_id] = candidates.get(rec_id, 0) + 1
    
    # 4. Urutkan berdasarkan skor
    sorted_recs = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    top_rec_ids = [x[0] for x in sorted_recs[:10]]
    
    # 5. KONVERSI ID -> JUDUL (Agar UI tidak error)
    # Buat mapping cepat dari movie_id ke title menggunakan _movie_data
    id_to_title = dict(zip(_movie_data['movie_id'], _movie_data['title']))
    
    # Terjemahkan ID ke Judul. Jika tidak ketemu, tampilkan ID-nya saja.
    top_rec_titles = [id_to_title.get(i, str(i)) for i in top_rec_ids]
    top_liked_titles = [id_to_title.get(i, str(i)) for i in top_liked_ids]
    
    return top_rec_titles, top_liked_titles
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
        "👤 Rekomendasi User",
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
        st.write('Pilih film yang Anda suka, dan kami akan merekomendasikan 10 film yang mirip!')

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

elif pilihan_halaman == "👤 Rekomendasi User":
    st.title("Rekomendasi Spesifik User 👤")
    
    # --- Load Data sebagai Dictionary ---
    knn_data = load_knn_data() # <--- 6. Panggil fungsi baru & simpan ke variabel baru
    
    # Pastikan data berhasil dimuat
    if knn_data is not None:
        
        # Ambil dataframe ratings dari dalam dictionary
        raw_ratings = knn_data['df_ratings_users']
        
        # --- LOGIKA PENCARIAN & FILTERING ---
        # 1. Ambil semua user unik
        all_users = raw_ratings['user'].unique()
        
        st.write("Cari User ID atau pilih dari daftar Top Reviewers:")
        
        # 2. Input Search
        search_query = st.text_input("🔍 Cari ID User (Ketik minimal 3 huruf):", "")
        
        options_to_show = []
        
        # 3. Logika Filter
        if search_query and len(search_query) >= 3:
            # Mengubah ke string untuk keamanan pencarian
            options_to_show = [str(u) for u in all_users if search_query.lower() in str(u).lower()]
            
            if not options_to_show:
                st.warning("User ID tidak ditemukan.")
        else:
            # Tampilkan Top 20 user paling aktif
            top_users = raw_ratings['user'].value_counts().head(20).index.tolist()
            options_to_show = top_users
            
            if not search_query:
                st.info("💡 Menampilkan 20 user paling aktif secara default.")

        # 4. Dropdown Selection
        selected_user = st.selectbox(
            "Pilih User dari hasil pencarian:",
            options_to_show
        )
        
        # --- TOMBOL EKSEKUSI ---
        if st.button(f"Analisa User: {selected_user}", disabled=not selected_user):
            
            with st.spinner(f"Sedang menganalisis selera {selected_user}..."):
                
                # --- PERBAIKAN 2: Panggil fungsi baru dengan 2 argumen saja ---
                recs, liked_movies = recommend_for_user(selected_user, knn_data, new_df)
                
                # --- TAMPILAN HASIL ---
                st.write("---")
                
                # History Tontonan
                with st.expander(f"📚 Lihat History Tontonan ({len(liked_movies)} film favorit)"):
                    st.write(", ".join(liked_movies))

                st.subheader(f"🎁 Rekomendasi Khusus:")
                
                if recs:
                    cols = st.columns(5)
                    for i, title in enumerate(recs[:5]): # Ambil 5 teratas
                        with cols[i]:
                            try:
                                # Mencari ID film dari judul untuk ambil poster
                                # Pastikan 'new_df' ada (sudah di-load di bagian atas script)
                                movie_matches = new_df[new_df['title'] == title]
                                
                                if not movie_matches.empty:
                                    m_id = movie_matches['movie_id'].values[0]
                                    st.image(fetch_poster(m_id))
                                    st.caption(title)
                                else:
                                    # Fallback jika film ada di collaborative tapi tidak ada di database poster
                                    st.image("https://via.placeholder.com/500x750.png?text=No+Image")
                                    st.caption(f"{title} (No Info)")
                            except Exception as e:
                                st.warning(f"Error loading {title}")
                else:
                    st.warning("Tidak dapat memberikan rekomendasi (User mungkin baru atau datanya unik).")
                
    else:
        st.error("Data rating user tidak ditemukan atau file corrupt.")

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