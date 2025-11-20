# 🎬 Sistem Rekomendasi Film

Aplikasi rekomendasi film menggunakan Content-Based Filtering (TF-IDF & Cosine Similarity).

## 🛠️ Cara Install & Menjalankan (Untuk Team)

Karena file model (`.pkl`) terlalu besar untuk di-upload ke GitHub, silakan ikuti langkah ini untuk membuatnya sendiri di laptop kalian.

### 1. Clone Repository
git clone https://github.com/USERNAME_KAMU/NAMA_REPO.git
cd NAMA_REPO

### 2. Siapkan Data
Pastikan kalian memiliki file dataset berikut di dalam folder project:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`
*(Download disini : [dari Kaggle atau minta link GDrive ke saya](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata))*

### 3. Install Library
pip install pandas scikit-learn streamlit requests

### 4. Generate Model (PENTING!)
Jalankan script ini **satu kali saja** untuk membuat file `similarity.pkl` dan `movies_clean.pkl`:
python setup_data.py

Tunggu sampai muncul tulisan "Selesai!".

### 5. Jalankan Aplikasi
streamlit run main.py
