# spam_email_classification

This project is a simple yet powerful Spam Email Classification System built using Natural Language Processing (NLP) and Machine Learning techniques. It classifies emails into two categories: Spam or Ham (Not Spam) using algorithms like Naive Bayes, Logistic Regression, and Random Forest.

**Project Features :**

1. Preprocessing of email text (lowercasing, punctuation removal, etc.)
2. Feature extraction using TfidfVectorizer
3. Model training using multiple ML algorithms
4. Evaluation with classification report and confusion matrix
5. Save/load trained models using pickle
6. Test individual email predictions with confidence scores

**Dataset :**
The project uses the Spam Emails Dataset available on Kaggle.
You can download it using the KaggleHub library:
import kagglehub
path = kagglehub.dataset_download("venky73/spam-mails-dataset")

**Libraries Used :**
pandas
numpy
sklearn
pickle
string, re
kagglehub (for Kaggle dataset download)
matplotlib (optional, for visualizations)

Install them using:
pip install pandas numpy scikit-learn kagglehub

**Models Used :**
1. Naive Bayes (Best performance in many spam filtering tasks)
2. Logistic Regression
3. Random Forest

**How to Run (VS Code) :**
1. Clone the Repository
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier

2. Install Dependencies
pip install -r requirements.txt

3. Run the Script
python spam_email_classifier.py

4. Test Prediction
After training, the model will save as best_spam_classifier.pkl. You can test new emails by modifying the test block inside the script.

**Example Output :**
1. Email: Congratulations! You won a FREE iPhone!
Prediction: SPAM (Confidence: 98.7%)

2. Email: Let's meet for lunch at 1 PM today.
Prediction: HAM (Confidence: 95.4%)

**Future Improvements :**
1. Add GUI or web interface
2. Integrate email fetching from inbox
3. Use deep learning models (LSTM, BERT)
4. Add visual performance metrics (ROC curves)



