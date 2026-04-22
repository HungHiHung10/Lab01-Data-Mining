

# Lab01-Data-Mining

## Đồ án 1: Tiền xử lý dữ liệu

**Phân tích thống kê — Thiết kế Pipeline — Phân tích và Đánh giá tác động**

---

## Thông tin nhóm

| STT | Họ và Tên | MSSV | Nhiệm vụ |
|-----|-----------|------|----------|
| 1 | Nguyễn Hữu Khánh Hưng | 23120271 | EDA chuyên sâu phần Ảnh và Văn bản. Thực hiện 1 câu nâng cao phần Ảnh, 1 câu nâng cao phần Văn bản |
| 2 | Y Nguyên Mlô | 23120299 | Các kỹ thuật tiền xử lý dữ liệu Hình ảnh (a, b, f) và Văn bản (d, e) |
| 3 | Phạm Quốc Khánh | 23120283 | Các kỹ thuật tiền xử lý dữ liệu Hình ảnh (c, d, f) và Văn bản (a, b, c) |
| 4 | Vũ Trần Phúc | 23120333 | Lựa chọn tập dữ liệu dạng Bảng. EDA chuyên sâu, các kỹ thuật tiền xử lý và đánh giá định lượng |
| 5 | Trần Nguyễn Minh Quân | 23120342 | Các kỹ thuật tiền xử lý và đánh giá định lượng dạng Bảng. Tổng hợp và thực hiện bài báo cáo |

---

## Mục lục

- [Mô tả Dự án](#mô-tả-dự-án)
- [Mô tả Tập dữ liệu](#mô-tả-tập-dữ-liệu)
- [Cấu trúc Dự án](#cấu-trúc-dự-án)
- [Yêu cầu và Cài đặt Môi trường](#yêu-cầu-và-cài-đặt-môi-trường)
- [Hướng dẫn Chạy](#hướng-dẫn-chạy)
- [Tóm tắt Kết quả](#tóm-tắt-kết-quả)
- [Output Files](#output-files)
- [Tài liệu Tham khảo](#tài-liệu-tham-khảo)

---

## Mô tả Dự án

Dự án thực hiện tiền xử lý dữ liệu toàn diện cho 3 loại dữ liệu khác nhau, mỗi loại gồm 2 notebooks (EDA + Preprocessing), tổng cộng 6 notebooks:

- **Dữ liệu Hình ảnh:** Phân tích thống kê pixel, chuẩn hoá, augmentation, trích xuất đặc trưng cạnh.
- **Dữ liệu Dạng Bảng:** Xử lý dữ liệu Amazon Products — imputation, outlier handling, scaling, encoding, feature selection, xử lý mất cân bằng lớp.
- **Dữ liệu Văn bản:** Xử lý IMDB Reviews — cleaning, tokenization, stemming/lemmatization, vectorization, classification và clustering.

Pipeline được thiết kế theo nguyên tắc thực nghiệm có kiểm soát (ablation-driven): mỗi bước triển khai nhiều phương án, đánh giá bằng metric định lượng, và chọn phương án tối ưu dựa trên bằng chứng thống kê.

---


## Mô tả Tập dữ liệu

| | Intel Image Classification | Amazon Products Sales | IMDB Movie Reviews |
|---|---|---|---|
| **Nguồn** | [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) | [Kaggle](https://www.kaggle.com/datasets) | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Loại dữ liệu** | Hình ảnh (RGB) | Bảng (CSV) | Văn bản (tiếng Anh) |
| **Quy mô** | ~25,000 ảnh, 150×150 pixel | 42,675 bản ghi, 18 thuộc tính | 50,000 bài đánh giá phim |
| **Mô tả** | Tập ảnh cảnh quan tự nhiên và đô thị, chia sẵn train/test/prediction. 6 lớp cân bằng tốt (IR max = 1.15). Định dạng JPEG/PNG | Dữ liệu sản phẩm Amazon gồm thông tin giá cả, đánh giá, lượt mua, trạng thái bán hàng và nhãn bền vững. Chứa cả biến số, biến phân loại và biến định danh. Tỷ lệ thiếu dữ liệu từ 2.4% đến 92% tuỳ thuộc tính | Bộ sưu tập đánh giá phim trên IMDB, cân bằng hoàn hảo 25K/25K giữa hai lớp. Văn bản tiếng Anh có độ dài trung bình ~230 từ, chứa nhiễu HTML và URL từ quá trình thu thập dữ liệu |

---

## Cấu trúc Dự án

```
Lab01-Data-Mining/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── amazon_products_sales_data_cleaned.csv
│   │   ├── image.txt
│   │   └── IMDB Dataset.csv
│   └── processed/
│       ├── dense.npy
│       ├── sparse.npy
│       ├── image_features.npy
│       └── images_dense.npy
│
└── notebooks/
    ├── 01_EDA_image.ipynb
    ├── 02_preprocessing_image.ipynb
    ├── 03_EDA_tabular.ipynb
    ├── 04_preprocessing_tabular.ipynb
    ├── 05_EDA_text.ipynb
    └── 06_Preprocessing_text.ipynb
```

---

## Yêu cầu và Cài đặt Môi trường

### Yêu cầu Hệ thống

- Python 3.8+
- pip hoặc conda
- Jupyter Notebook hoặc JupyterLab
- (Khuyến nghị) GPU với CUDA để tăng tốc PyTorch

### Cài đặt

```bash
# Sao chép dự án
git clone https://github.com/HungHiHung10/Lab01-Data-Mining.git
cd Lab01-Data-Mining

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

### Tải tài nguyên NLP (bắt buộc cho phần Văn bản)

```python
import nltk
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)
```

---

## Hướng dẫn Chạy

### Chuẩn bị dữ liệu

Đảm bảo các file nằm đúng vị trí trong `data/raw/`:

```
data/raw/
├── amazon_products_sales_data_cleaned.csv
├── image.txt
└── IMDB Dataset.csv
```

### Thứ tự thực thi

**Giai đoạn 1 — EDA:**

1. `01_EDA_image.ipynb`
2. `03_EDA_tabular.ipynb`
3. `05_EDA_text.ipynb`

**Giai đoạn 2 — Preprocessing:**

4. `02_preprocessing_image.ipynb`
5. `04_preprocessing_tabular.ipynb`
6. `06_Preprocessing_text.ipynb`

Các notebook trong cùng giai đoạn có thể chạy độc lập. EDA nên chạy trước Preprocessing tương ứng vì kết quả EDA định hướng các quyết định tiền xử lý.

---

## Tóm tắt Kết quả

### Dữ liệu Hình ảnh

**EDA:** Dữ liệu cân bằng tốt (IR max = 1.15). Phát hiện 0.43% ảnh trùng lặp qua pHash, loại bỏ để tránh data leakage. Phân phối độ sáng lệch trái và không đồng đều giữa các lớp (glacier sáng nhất, forest tối nhất). Nhóm Đô thị có tương phản sắc nét hơn nhóm Thiên nhiên — gợi ý edge/texture là đặc trưng phân biệt chính.

**Preprocessing:** Pipeline ablation-driven so sánh nhiều phương án cho từng bước: Resize (32×32 vs 64×64 vs 128×128 → chọn 32×32), Không gian màu (RGB vs Grayscale vs HSV vs LAB → chọn HSV), Chuẩn hoá (MinMax vs Z-score toàn tập vs Z-score per-channel → chọn Z-score per-channel), Augmentation (KS test xác nhận dịch chuyển phân phối có kiểm soát), Giảm chiều (PCA xác nhận dữ liệu dư thừa cao, t-SNE cho thấy glacier–mountain–sea chồng lấp), Đặc trưng cạnh (Sobel, Prewitt, Canny — ANOVA có ý nghĩa nhưng đơn lẻ chưa đủ). Cấu hình chốt: Resize 32×32 | HSV | Z-score per-channel | Augmentation.

---

### Dữ liệu Bảng

**EDA:** Kiểm định D'Agostino-Pearson bác bỏ phân phối chuẩn cho 6/6 biến số (p ≈ 0), skewness từ -1.84 đến 24.34. Đa cộng tuyến nghiêm trọng giữa original\_price và discounted\_price (r = 0.97–0.99). Chênh lệch Pearson vs Spearman chứng minh dữ liệu chứa outlier và quan hệ phi tuyến. Little's MCAR test bác bỏ MCAR (chi² = 6,519) — dữ liệu thiếu theo MAR kết hợp MNAR.

**Preprocessing:** Xử lý giá trị thiếu — mô phỏng 10% MCAR, so sánh 5 chiến lược bằng RMSE; Hybrid Imputation: k-NN cho biến hành vi, MICE cho biến giá (cải thiện 10.5–58.1% so với đơn biến). Phát hiện ngoại lai — 5 phương pháp (IQR, Z-Score, IF, LOF, DBSCAN), đối chiếu Jaccard + KS Test; chiến lược kép: majority voting + Winsorization P1-P99 (loại 1,115 dòng, skewness 12.32 → 6.04). Chuẩn hoá — so sánh 4 scaler qua Levene + Violin Plot; RobustScaler phù hợp nhất cho biến lệch mạnh. Mã hoá — so sánh 4 phương pháp qua VIF; Frequency Encoding tối ưu (Mean VIF 4.977). Lựa chọn đặc trưng — 3 tầng (Filter, Model-based, PCA/t-SNE); mọi phương pháp hội tụ R² ≈ 0.23 với LR, RF đạt R² ≈ 0.96 trên cùng features. Mất cân bằng lớp — imbalance 1:8; No Resampling + RF đạt F1-macro cao nhất (0.9059); AUC-ROC ≈ 0.98–0.99 cho mọi chiến lược.

---

### Dữ liệu Văn bản

**EDA:** Dữ liệu cân bằng hoàn hảo (25K mỗi lớp). Độ dài văn bản không phân biệt cảm xúc (Mann-Whitney U: Word Count p=0.029 không thực tiễn, Char Count p=0.173 không ý nghĩa). Nghịch lý từ phủ định ở cả hai lớp chứng minh cần N-grams. Zipf (R²=0.98) phát hiện 62% Hapax Legomena. 4 vấn đề cốt lõi: nhiễu HTML/URL, phân mảnh từ vựng, domain stopwords, curse of dimensionality (214K từ).

**Preprocessing:** Regex clean → NLTK tokenize → WordNet Lemmatize → TF-IDF (min\_df=3, max\_df=0.95, bi-gram) cắt vocabulary từ ~100K xuống ~40–50K. So sánh Word-level vs BPE; WordNet Lemmatizer an toàn hơn Snowball Stemmer. TF-IDF + LinearSVC đạt Accuracy ~89.5%, vượt SBERT + LinearSVC (~82.3%) do linear bottleneck trên không gian phi tuyến. SBERT cần mô hình phi tuyến để phát huy tiềm năng.

---

## Output Files

| File | Định dạng | Mô tả |
|------|-----------|-------|
| `dense.npy` | numpy array | Ma trận dữ liệu dạng dense (text) |
| `sparse.npy` | sparse matrix | Ma trận TF-IDF dạng sparse (text) |
| `image_features.npy` | numpy array | Feature matrix trích xuất từ ảnh |
| `images_dense.npy` | numpy array | Ảnh đã tiền xử lý |

---

## Tài liệu Tham khảo

- Scikit-learn Documentation — https://scikit-learn.org/stable/
- Pandas Documentation — https://pandas.pydata.org/docs/
- SciPy Statistical Tests — https://docs.scipy.org/doc/scipy/reference/stats.html
- OpenCV Documentation — https://docs.opencv.org/
- NLTK Book — https://www.nltk.org/book/
- spaCy Documentation — https://spacy.io/
- imbalanced-learn Documentation — https://imbalanced-learn.org/stable/
- PyTorch Documentation — https://pytorch.org/docs/stable/
- Gensim Word2Vec — https://radimrehurek.com/gensim/models/word2vec.html
- Sentence Transformers — https://www.sbert.net/