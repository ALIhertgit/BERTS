This repository contains a Jupyter Notebook (BRETA.ipynb) that walks through the process of building a sophisticated text classifier. The goal is to determine the "impact level" (Low, Medium, or High) of a news headline.
The project demonstrates a complete workflow, starting from a simple keyword-based labeling system and progressively building up to a robust, context-aware BERT model. It highlights common challenges in NLP projects, such as class imbalance and model overfitting on trivial keywords, and provides effective solutions.
The Narrative: A Journey in 5 Phases
The notebook tells a story of iterative model improvement:
Phase 1: Keyword-Based Labeling: We begin without a pre-labeled dataset. Instead, we create our own labels using a dictionary of "trigger words." This pragmatic approach quickly gets us a labeled dataset but introduces significant class imbalance and a heavy reliance on simple keywords.
Phase 2: The Baseline - Bag-of-Words (BoW): To establish a performance baseline, a classic BoW model with a Logistic Regression classifier is trained. It performs reasonably well on the majority class but fails catastrophically on the minority "High" impact class, achieving an F1-score of just 0.06. This clearly demonstrates the need for a more advanced approach.
Phase 3: Initial BERT Model (Imbalanced Data): Our first attempt with bert-base-uncased shows a massive improvement over the BoW baseline. However, while overall accuracy is high (99.2%), the F1-score for the minority class is still lagging, and the model is likely just memorizing the trigger words.
Phase 4: Tackling Imbalance with BERT: We address the class imbalance by creating a balanced dataset (through oversampling minority classes). Training BERT on this new dataset yields near-perfect results (0.999 F1-score), showing the model can now effectively distinguish between all three classes.
Phase 5: Forcing Generalization with Masking: To ensure the model isn't just cheating by memorizing the trigger words, a clever data augmentation technique is introduced. During training, a random 50% of the trigger words in the input text are replaced with [MASK]. This forces the model to learn the surrounding context to make its prediction, leading to a more robust and generalizable classifier. This final model achieves an excellent F1-score of 0.981.
Key Features
Pragmatic Data Labeling: Demonstrates how to create a labeled dataset from scratch using a keyword-based system.
Class Imbalance Handling: Solves a critical machine learning problem using oversampling.
Advanced Model Training: Fine-tunes a bert-base-uncased model for sequence classification using the Hugging Face transformers and datasets libraries.
Robustness via Masking: Implements a trigger-word masking strategy to improve model generalization and prevent overfitting on simple cues.
Baseline Comparison: Provides a clear comparison against a scikit-learn Bag-of-Words model to highlight the power of transformers.
Comprehensive Evaluation: Uses classification reports and confusion matrices to analyze model performance across different stages.
Final Results
The final model, trained on a balanced and masked dataset, demonstrates strong performance and a good understanding of context beyond simple keywords.
Model Performance: BERT vs. Bag-of-Words
The bar chart clearly shows the superiority of the fine-tuned BERT model, especially for the under-represented "High" impact class where the BoW model failed.
Final Masked-BERT Model Performance
Class	Precision	Recall	F1-Score	Support
0 (Low)	0.989	0.976	0.982	619
1 (Med)	0.964	0.980	0.972	604
2 (High)	0.990	0.986	0.988	577
Macro Avg	0.981	0.981	0.981	1800
Training Loss
The training curve for the final model shows a steady decrease in loss, indicating successful learning.
Technologies Used
PyTorch
Hugging Face Transformers: For the BERT model and training infrastructure.
Hugging Face Datasets: For efficient data handling and preprocessing.
Scikit-Learn: For the baseline model, metrics, and data splitting.
Pandas: For data manipulation.
Seaborn & Matplotlib: For data visualization.
How to Run
Clone the Repository
Generated bash
git clone <repository-url>
cd <repository-directory>
Use code with caution.
Bash
Set up Environment
It is recommended to use a virtual environment.
Generated bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Use code with caution.
Bash
Install Dependencies
The core dependencies can be installed by running the first code cell in the notebook.
Generated python
!pip install -q transformers datasets scikit-learn
Use code with caution.
Python
Note: The notebook was created in a Google Colab environment with access to a GPU (Tesla T4).
Run the Notebook
Open BRETA.ipynb in a Jupyter environment (like Jupyter Lab or VS Code) and run the cells sequentially.
A Note on TrainingArguments
The notebook contains a TypeError in cells 5 and 19 related to the evaluation_strategy argument. This is a common issue when using a version of the transformers library where this argument was deprecated or renamed. The notebook correctly resolves this by removing the argument in the subsequent cells (8 and 20). If you are using a very recent version of transformers, you might need to use evaluation_strategy="epoch" and remove the tokenizer argument from the Trainer in favor of a DataCollator. The provided code is self-correcting and should run as-is.
