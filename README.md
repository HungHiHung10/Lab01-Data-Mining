

# Lab01-Data-Mining

## Đồ án 1: Tiền xử lý dữ liệu

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

- [Lab01-Data-Mining](#lab01-data-mining)
  - [Đồ án 1: Tiền xử lý dữ liệu](#đồ-án-1-tiền-xử-lý-dữ-liệu)
  - [Thông tin nhóm](#thông-tin-nhóm)
  - [Mục lục](#mục-lục)
  - [Mô tả Dự án](#mô-tả-dự-án)
  - [Mô tả Tập dữ liệu](#mô-tả-tập-dữ-liệu)
  - [Cấu trúc Dự án](#cấu-trúc-dự-án)
  - [Yêu cầu và Cài đặt Môi trường](#yêu-cầu-và-cài-đặt-môi-trường)
    - [Yêu cầu Hệ thống](#yêu-cầu-hệ-thống)
    - [Cài đặt](#cài-đặt)
    - [Tải tài nguyên NLP (bắt buộc cho phần Văn bản)](#tải-tài-nguyên-nlp-bắt-buộc-cho-phần-văn-bản)
  - [Hướng dẫn Chạy](#hướng-dẫn-chạy)
    - [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
    - [Thứ tự thực thi](#thứ-tự-thực-thi)
  - [Tóm tắt kết quả](#tóm-tắt-kết-quả)
    - [Dữ liệu Ảnh](#dữ-liệu-ảnh)
    - [Dữ liệu Bảng](#dữ-liệu-bảng)
    - [Dữ liệu Văn bản](#dữ-liệu-văn-bản)
  - [Output Files](#output-files)
  - [Tài liệu Tham khảo](#tài-liệu-tham-khảo)

---

## Mô tả Dự án

Dự án thực hiện tiền xử lý dữ liệu toàn diện cho 3 loại dữ liệu khác nhau, mỗi loại gồm 2 notebooks (EDA + Preprocessing), tổng cộng 6 notebooks:

- **Dữ liệu Hình ảnh:** Phân tích thống kê pixel, chuẩn hoá, augmentation, trích xuất đặc trưng cạnh.
- **Dữ liệu Dạng Bảng:** Xử lý dữ liệu Amazon Products — imputation, outlier handling, scaling, encoding, feature selection, xử lý mất cân bằng lớp.
- **Dữ liệu Văn bản:** Xử lý IMDB Reviews — cleaning, tokenization, stemming/lemmatization, vectorization, classification và clustering.

Pipeline được thiết kế theo nguyên tắc thực nghiệm có kiểm soát: mỗi bước triển khai nhiều phương án, đánh giá bằng metric định lượng, và chọn phương án tối ưu dựa trên bằng chứng thống kê.

---


## Mô tả Tập dữ liệu

| | Intel Image Classification | Amazon Products Sales | IMDB Movie Reviews |
|---|---|---|---|
| **Nguồn** | [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) | [Kaggle](https://www.kaggle.com/datasets/ikramshah512/amazon-products-sales-dataset-42k-items-2025) | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Loại dữ liệu** | Hình ảnh (RGB) | Bảng (CSV) | Văn bản (tiếng Anh) |
| **Quy mô** | ~25,000 ảnh, 150×150 pixel | 42,675 bản ghi, 17 thuộc tính | ~50,000 bài đánh giá phim |
| **Mô tả** | Tập ảnh cảnh quan tự nhiên và đô thị, chia sẵn train/test/prediction. 6 lớp cân bằng tốt (IR max = 1.15). Định dạng JPEG/PNG | Dữ liệu sản phẩm Amazon gồm thông tin giá cả, đánh giá, lượt mua, trạng thái bán hàng và nhãn bền vững. Chứa cả biến số, biến phân loại và biến định danh. Tỷ lệ thiếu dữ liệu từ 2.4% đến 92% tuỳ thuộc tính | Bộ sưu tập đánh giá phim trên IMDB, cân bằng hoàn hảo 25K/25K giữa hai lớp. Văn bản tiếng Anh có độ dài trung bình ~230 từ, chứa nhiễu HTML và URL từ quá trình thu thập dữ liệu |


<u>**Lưu ý:**</u> Bắt buộc tải tất cả các file dữ liệu từ Google Drive:

**[Link Google Drive](https://drive.google.com/drive/folders/1uTFaLdK2uq7u_jMF_ao1ffoF-5k838El?usp=sharing)**

Đảm bảo các folder `raw/` và `processed/` nằm đúng vị trí trong `data/`:



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

Trong phần tiền xử lý văn bản, sử dụng NLTK để tải các tài nguyên cần thiết như tokenizer, stopwords, và WordNet. Chạy đoạn code sau một lần trong Jupyter Notebook để tải về các tài nguyên này:
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
## Tóm tắt kết quả

### Dữ liệu Ảnh
---
**EDA — Phát hiện chính**

Dữ liệu cân bằng tốt với Imbalance Ratio tối đa chỉ 1.15, không cần resampling. Quá trình kiểm tra chất lượng phát hiện 0.43% ảnh trùng lặp tuyệt đối qua perceptual hashing (pHash), được loại bỏ để tránh data leakage. Phân tích đặc trưng thị giác cho thấy hai điểm đáng chú ý: phân phối độ sáng lệch trái và không đồng đều giữa các lớp (glacier sáng nhất, forest tối nhất), đồng thời nhóm Đô thị (street, buildings) có tương phản sắc nét hơn hẳn nhóm Thiên nhiên (glacier, sea) — gợi ý rằng đặc trưng cạnh (edge) và kết cấu (texture) sẽ là chìa khoá phân biệt chính.

**Preprocessing**

Pipeline được thiết kế theo hướng thực nghiệm có kiểm soát, mỗi bước đều so sánh nhiều phương án và chọn dựa trên bằng chứng định lượng:

- **Resize:** So sánh 32×32, 64×64, 128×128 → chọn 32×32 (cân bằng tốt giữa tốc độ tính toán và mức giữ thông tin).
- **Không gian màu:** So sánh RGB, Grayscale, HSV, LAB → chọn HSV (kênh Hue/Saturation tách tốt các lớp có đặc trưng màu mạnh như forest).
- **Chuẩn hoá:** So sánh Min-Max [0,1], Min-Max [-1,1], Z-score toàn tập, Z-score per-channel → chọn Z-score per-channel (ổn định nhất, giảm lệch thang đo giữa các kênh màu).
- **Augmentation:** Dịch chuyển phân phối có kiểm soát — KS test cho thấy phân phối pixel thay đổi có ý nghĩa nhưng cấu trúc cụm lớp vẫn được duy trì.
- **Giảm chiều:** PCA xác nhận dữ liệu dư thừa cao, nén mạnh được. t-SNE cho thấy ranh giới phi tuyến giữa các lớp, đặc biệt glacier–mountain–sea chồng lấp đáng kể.
- **Đặc trưng cạnh:** Sobel, Prewitt, Canny — edge density có ý nghĩa thống kê giữa các lớp (ANOVA p-value rất nhỏ), tuy nhiên đơn lẻ chưa đủ phân loại hoàn chỉnh.

**Cấu hình chốt:** Resize 32×32 | HSV | Z-score per-channel | Augmentation.

**Hạn chế và hướng phát triển:** Pipeline hiện thiên về mô hình cổ điển (LR, k-NN) với đặc trưng bậc thấp (pixel, màu, cạnh), augmentation chưa tối ưu theo từng lớp. Hướng tiếp theo là tích hợp dữ liệu đã tiền xử lý vào các kiến trúc Deep Learning (CNN/ViT) kết hợp cross-validation để bứt phá giới hạn hiệu năng.

---
### Dữ liệu Bảng 
---

**EDA — Phát hiện chính**

Kiểm định D'Agostino-Pearson bác bỏ phân phối chuẩn cho 6/6 biến số (p ≈ 0), mức độ lệch từ -1.84 (product\_rating) đến 24.34 (total\_reviews). Heatmap Pearson và Spearman phát hiện đa cộng tuyến nghiêm trọng giữa original\_price và discounted\_price (r = 0.97–0.99), đồng thời chênh lệch lớn giữa hai hệ số ở nhiều cặp biến (discounted\_price — purchased\_last\_month: Pearson = -0.10, Spearman = -0.69) chứng minh dữ liệu chứa outlier và quan hệ phi tuyến. Little's MCAR test bác bỏ cơ chế MCAR (chi² = 6,519, p ≈ 0) — dữ liệu thiếu theo MAR kết hợp MNAR, với sustainability\_tags thiếu 92%, nhóm biến giá thiếu đồng thời 4.8%, nhóm rating/reviews thiếu 2.4%. Đề xuất: RobustScaler cho biến lệch mạnh, loại bỏ original\_price, KNN Imputer cho biến số và hằng số domain cho biến phân loại.

**Preprocessing — Pipeline thực nghiệm có kiểm soát**

Mỗi bước triển khai nhiều phương án cạnh tranh, đánh giá bằng metric định lượng:

- **Xử lý giá trị thiếu:** Mô phỏng 10% MCAR trên Ground Truth, so sánh 5 chiến lược bằng RMSE. k-NN và MICE áp đảo phương pháp đơn biến (cải thiện 10.5–58.1%). Triển khai Hybrid Imputation: k-NN (k=10) cho biến hành vi, MICE cho biến giá. Biến phân loại: buy\_box\_availability → "Not Available", sustainability\_tags → "No Tag".
- **Phát hiện ngoại lai:** 5 phương pháp (IQR, Z-Score, Isolation Forest, LOF, DBSCAN), đối chiếu qua Jaccard và KS Test. IQR gộp đánh dấu 41% — quá khắt khe; Z-Score bị masking effect; phương pháp đa biến bảo toàn phân phối tốt hơn. Chiến lược kép: majority voting đa biến + Winsorization P1-P99. Loại 1,115 dòng → 41,560. Skewness target: 12.32 → 6.04.
- **Chuẩn hoá:** So sánh MinMax, Z-Score, RobustScaler, Quantile qua Levene Test và Violin Plot. RobustScaler phù hợp nhất cho biến lệch mạnh. StandardScaler vẫn khả dụng cho tree-based models.
- **Mã hoá:** So sánh One-Hot, Target, Binary, Frequency Encoding qua VIF. Frequency Encoding tối ưu: giữ 16 features, Mean VIF thấp nhất (4.977), không tạo đa cộng tuyến mới. Loại original\_price (r = 0.97) và cột định danh.
- **Lựa chọn đặc trưng:** 3 tầng — Filter (ANOVA, Chi-square, MI), Model-based (RF/GB importance, RFE-CV), Giảm chiều (PCA, t-SNE). Mọi phương pháp hội tụ R² ≈ 0.2348 với Linear Regression. total\_reviews chiếm 84–85% importance trong RF/GB. PCA cần 9/10 PCs cho 90% variance — không giảm chiều được. Vấn đề nằm ở model choice: cùng features, RF đạt R² ≈ 0.96 vs LR ≈ 0.23.
- **Mất cân bằng lớp:** Target rời rạc hoá thành is\_popular (top 10%), imbalance 1:8. No Resampling + RF đạt F1-macro cao nhất (0.9059). SMOTE cân bằng tốt nhất khi cần Recall cao (P=0.72, R=0.90). AUC-ROC ≈ 0.98–0.99 cho mọi chiến lược — resampling dịch chuyển decision boundary, không cải thiện discriminative power. Resampling trước split gây data leakage.

---
### Dữ liệu Văn bản 
---

**EDA — Phát hiện chính**

Dữ liệu cân bằng hoàn hảo (25,000 mẫu mỗi lớp), không cần resampling. Độ dài văn bản lệch phải mạnh (Mean ~230, Median ~173 từ) nhưng không phân biệt được cảm xúc (Mann-Whitney U: Word Count p=0.029 — ý nghĩa thống kê nhưng không thực tiễn, Character Count p=0.173 — không ý nghĩa). Nghịch lý từ phủ định xuất hiện ở cả hai lớp: "good" trong Negative ("not good"), "never" trong Positive ("never seen better") — chứng minh phân tích unigram không đủ, cần N-grams. Phân tích Zipf (R²=0.98) phát hiện 62% từ vựng chỉ xuất hiện 1 lần (Hapax Legomena). Tổng hợp 4 vấn đề cốt lõi: nhiễu HTML/URL, phân mảnh từ vựng (movie/movies), domain stopwords không phân biệt cảm xúc, và curse of dimensionality từ 214K từ vựng.

**Preprocessing — Pipeline NLP**

Pipeline giải quyết tuần tự 4 vấn đề từ EDA:

- **Làm sạch:** Regex loại HTML, URL, ký tự đặc biệt, lowercase. Stopwords mặc định cần tuỳ chỉnh để không xoá nhầm từ phủ định quan trọng.
- **Tokenization:** So sánh Word-level (NLTK) và Subword (BPE) — Word-level cho BoW/TF-IDF, BPE triệt tiêu OOV cho Transformer.
- **Lemmatization:** WordNet Lemmatizer được chọn (an toàn ngữ nghĩa hơn Snowball Stemmer), giải quyết phân mảnh từ vựng.
- **Vector hoá:** TF-IDF (min\_df=3, max\_df=0.95, bi-gram) cắt vocabulary từ ~100K xuống ~40–50K. So sánh với Word2Vec và SBERT.

**Kết quả:** TF-IDF + LinearSVC đạt Accuracy ~89.5%, vượt SBERT + LinearSVC (~82.3%) do SBERT nén ngữ nghĩa vào không gian phi tuyến mà bộ phân loại tuyến tính không bóc tách hiệu quả. Pipeline chốt: Regex clean → NLTK tokenize → WordNet Lemmatize → TF-IDF (bi-gram) + LinearSVC. SBERT cần mô hình phi tuyến để phát huy tiềm năng.

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
