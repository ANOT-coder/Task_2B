# 🇮🇩 IndoBERT Emotion Classifier Based  NLP Text Classification
A deep-learning-based NLP system fine-tuned on the **IndoBERT** architecture to classify Indonesian social media text into five emotional categories: **Anger, Fear, Happy, Love, and Sadness.**

## 📊 Dataset Background
The model was trained using the **Indonesian Emotion Twitter Dataset**. Social media data in Indonesia is uniquely challenging due to its "code-switching" nature (mixing formal Indonesian with local dialects and English) and the heavy use of "Bahasa Gaul" (slang).
* **Dataset Source:** [Indonesian Twitter Emotion Dataset](https://www.kaggle.com/datasets/dennisherdi/indonesian-twitter-emotion)


---

## 🛠 Deep-Dive: Data Cleaning & Preprocessing
To achieve a high accuracy of 76%, the raw data underwent a rigorous cleaning pipeline to reduce noise and help the transformer focus on emotional "tokens":

1.  **Regex-Based Scrubbing:** Removal of non-essential Twitter metadata including URLs (`http\S+`), user mentions (`@username`), and hashtags (`#`).
2.  **Noise Reduction:** Stripping of punctuation and numerical characters that do not carry emotional weight.
3.  **Case Folding:** Standardizing all text to lowercase to ensure the model treats "MARAH" and "marah" as the same semantic unit.
4.  **Tokenization (WordPiece):** Using the IndoBERT tokenizer, sentences were broken into sub-word units. This is critical for Indonesian because it handles complex prefixes and suffixes (e.g., *kebahagiaan* → *ke* + *bahagia* + *an*).

---

## 🧠 Why IndoBERT? (Technical Architecture)
The project utilizes `indobenchmark/indobert-base-p2`. Unlike standard BERT (trained on English), IndoBERT was pre-trained on a **23GB corpus of 4 billion Indonesian words** from Wikipedia, news articles, and WebCrawler data.

### 🔬 The Bidirectional Advantage
Standard NLP models (like LSTMs) read text from left-to-right. BERT uses a **Bidirectional** approach, meaning it looks at the words both before and after a specific word simultaneously to understand context.
* **Contextual Example:** In the sentence *"Aku sangat marah tapi tetap sayang,"* the model understands "marah" is softened by "sayang" because it views the whole sentence structure at once.

 token used for classification

---

## 📐 Mathematical Foundations
The model's "learning" is driven by minimizing the **Cross-Entropy Loss**. Since we are dealing with 5 distinct emotions, we use the **Multiclass Cross-Entropy Loss** formula:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{i,j} \log(p_{i,j})$$

Where:
* $N$ is the number of samples.
* $C$ is the number of classes (5).
* $y_{i,j}$ is a binary indicator (1 if class $j$ is the correct label for sample $i$).
* $p_{i,j}$ is the probability predicted by the model for class $j$.

The model applies a **Softmax function** at the final layer to turn raw "logits" into a probability distribution that sums to 1.0 (100%).

---

## 📈 Final Evaluation Results
The model was evaluated on a held-out test set (20% of the total data).

### Classification Report:
| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Anger** | 0.78 | 0.74 | 0.76 | 221 |
| **Fear** | 0.81 | 0.85 | 0.83 | 129 |
| **Happy** | 0.74 | 0.76 | 0.75 | 203 |
| **Love** | 0.79 | 0.77 | 0.78 | 127 |
| **Sadness** | 0.72 | 0.73 | 0.72 | 196 |
| **Accuracy** | | | **0.76** | **876** |



---

## 📂 Model Repository Checklist
The following 6 files must be present in the `model_folder/` for the system to function:
1.  **`model.safetensors`** (474.9 MB) – The fine-tuned weights.
2.  **`config.json`** – The model architecture configuration.
3.  **`tokenizer.json`** – The vocabulary dictionary.
4.  **`label_encoder.joblib`** – The mapping for emotion labels.
5.  **`tokenizer_config.json`** – Tokenizer settings.
6.  **`training_args.bin`** – Training hyperparameter metadata.

---

## 🔍 Result Analysis

### Why 76% Accuracy?
While the model performs exceptionally well, several linguistic factors influenced the 76% ceiling:
* **Sentiment Overlap:** Indonesian tweets often use similar vocabulary for "Anger" and "Sadness." Without emojis or extended context, the model occasionally mislabels high-intensity negative emotions.
* **Informal Slang:** The use of "Bahasa Gaul" (slang) and abbreviated words requires highly specific tokenization; while IndoBERT is excellent at this, some extremely localized slang remains challenging.
* **Short-Form Constraints:** Tweets with fewer than 5 words often lack the contextual "tokens" needed for the transformer to reach 90%+ confidence.

### 💡 Future Improvements
To move toward 85%+ accuracy, the following steps are recommended:
* **Data Augmentation:** Increasing the sample size for "Fear" and "Love" through back-translation.
* **Sarcasm Detection:** Implementing a secondary layer specifically to detect sarcastic nuances in Indonesian social media.
* **Longer Training:** Increasing epochs with a smaller learning rate scheduler.
---