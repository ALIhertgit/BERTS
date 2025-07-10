# Impact-Level Classification of News Headlines with BERT

This repository contains a Jupyter Notebook that demonstrates a complete workflow for building a sophisticated text classifier to determine the "impact level" (Low, Medium, or High) of a news headline.

The project begins by pragmatically creating a labeled dataset from keywords, then establishes a baseline with a traditional Bag-of-Words (BoW) model. It culminates in fine-tuning a robust, context-aware BERT model, using a novel trigger-word masking technique to prevent overfitting and improve generalization.

---

### Table of Contents
1.  [Introduction](#1-introduction)
2.  [Data and Methodology](#2-data-and-methodology)
3.  [Model Implementation](#3-model-implementation)
4.  [Experimental Results](#4-experimental-results)
5.  [Analysis and Conclusion](#5-analysis-and-conclusion)
6.  [Technologies Used](#6-technologies-used)
7.  [How to Run](#7-how-to-run)

---

### 1. Introduction
Efficient prioritisation of news articles based on impact levels is critical in news aggregation and information dissemination. This study investigates the classification of news headlines into three distinct categories—Low, Medium, and High impact—using advanced machine learning methods. We compare the performance of a traditional Bag-of-Words (BoW) approach to a state-of-the-art BERT model, fine-tuned with a novel trigger-word masking technique designed to encourage contextual understanding rather than keyword memorisation.

### 2. Data and Methodology
**Dataset**

The AG-News dataset, comprising a random subset of 10,000 news headlines, was used. Headlines were categorised based on predefined keywords associated with impact levels. To address a significant class imbalance, the dataset was balanced to **3,000 entries** per impact level through upsampling of minority classes.

**Preprocessing**

Two distinct preprocessing pipelines were used:
*   **For the BERT model**, a trigger-word masking strategy was applied during tokenisation. This technique randomly masked 50% of predefined trigger words with the `[MASK]` token, forcing the model to learn from the surrounding context.
*   **For the Bag-of-Words baseline**, stop-words were removed before constructing a fixed vocabulary of 600 words (300 most common and 300 least common).

### 3. Model Implementation
**Bag-of-Words Baseline**

A baseline model using the fixed 600-word vocabulary was implemented. A multinomial Logistic Regression model with L2 regularisation (C=1, max_iter=1000) was used.

**BERT Model**

The pre-trained `bert-base-uncased` model was fine-tuned for two epochs using a batch size of 16 on a Tesla T4 GPU. The trigger-word masking strategy was integrated directly into the tokenisation function to enhance the model's generalisation capabilities.

### 4. Experimental Results
The final masked BERT model demonstrated a dramatic improvement over the Bag-of-Words baseline, especially in identifying the rare "High" impact class.

| Model           | Accuracy | F1-Low | F1-Medium | F1-High | F1-macro |
| :-------------- | :------- | :----- | :-------- | :------ | :------- |
| Bag-of-Words    | 0.913    | 0.954  | 0.806     | 0.059   | 0.606    |
| **BERT (masked)**   | **0.981**    | **0.982**  | **0.972**     | **0.988**   | **0.981**    |

<br>

**Figure 1: Confusion Matrix for the Bag-of-Words Baseline Model**
*(This matrix highlights the baseline's failure to correctly classify "High" impact headlines.)*

![BoW Confusion Matrix](Picture1.png)

<br>

**Figure 2: Per-class F1-score Comparison: BERT vs. Bag-of-Words**
*(This chart clearly illustrates the superior performance of the BERT model across all classes.)*

![F1 Score Comparison](Picture2.png)

<br>

**Figure 3: Training Loss Curve for the Final Masked BERT Model**
*(This curve shows the model's successful convergence during training.)*

![BERT Loss Curve](Picture3.png)

### 5. Analysis and Conclusion
The Bag-of-Words baseline demonstrated limited capability in classifying rare High-impact headlines due to its reliance on explicit keyword matching, achieving a mere 0.06 F1-score for that class. Conversely, the masked BERT model exhibited superior performance, effectively leveraging contextual information beyond mere keyword occurrences. The trigger-word masking strategy proved crucial in preventing the model from simply memorizing keywords, forcing it to generalize and achieve a robust F1-macro score of 0.981.

The masked BERT model significantly outperformed the traditional Bag-of-Words approach, underscoring the effectiveness of contextual embedding techniques in news classification tasks. Future research should explore human-labelled datasets, enhanced masking techniques, and considerations for real-time deployment.

### 6. Technologies Used
*   **PyTorch**
*   **Hugging Face Transformers**: For the BERT model and training infrastructure.
*   **Hugging Face Datasets**: For efficient data handling and preprocessing.
*   **Scikit-Learn**: For the baseline model, metrics, and data splitting.
*   **Pandas**: For data manipulation.
*   **Seaborn & Matplotlib**: For data visualization.

### 7. How to Run
1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Set up Environment:**
    It is highly recommended to run this notebook in an environment with a GPU. Google Colab is an excellent choice.
    ```bash
    # It is recommended to use a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    The core dependencies can be installed by running the first code cell in the `BRETA.ipynb` notebook.
    ```python
    !pip install -q transformers datasets scikit-learn
    ```
4.  **Run the Notebook:**
    Open `BRETA.ipynb` in a Jupyter environment (like JupyterLab, VS Code, or Google Colab) and run the cells sequentially to reproduce the results.
