# simpan ini dengan nama setup_data.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import pickle 
print("Sedang memuat data...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]    
movies.dropna(inplace=True)

# --- Fungsi Helper ---
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

print("memproses data ...")
# Proses Cleaning
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

movies['tags'] = movies['overview'].apply(lambda x: [x] if isinstance(x, str) else []) + \
                 movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Dataframe Akhir
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

print("Sedang menghitung similarity matrix...")
# Hitung Similarity
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# --- SIMPAN HASILNYA ---
print("Menyimpan file...")
# 1. Simpan DataFrame bersih ke CSV (atau Pickle agar tipe data terjaga)
new_df.to_pickle('movies_clean.pkl') 
# atau new_df.to_csv('movies_clean.csv', index=False) -> Tapi pickle lebih cepat load-nya

# 2. Simpan Similarity Matrix ke Pickle (karena ini array besar)
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# 3. (Opsional) Simpan data mentah yang sudah di-merge untuk keperluan "General Info"
# Agar halaman General Info tidak perlu merge ulang

movies.to_csv('movies_complete.csv', index=False)
#movies.to_pickle('movies_complete.pkl')

print("Selesai! Sekarang file 'movies_clean.pkl', 'movies_complete.pkl', dan 'similarity.pkl' sudah siap.")