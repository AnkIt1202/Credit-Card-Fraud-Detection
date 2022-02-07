# Detecting-credit-card-fraud-transactions-using-ML-Classification-Models



Credit card fraud happens when someone — a fraudster or a thief — uses your stolen credit card or the information from that card to make unauthorized purchases in your name or take out cash advances using your account.

### **Problem Statement:**

Credit card companies such as **Citibank**, **HSBC**, and **American Express** need to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

### **Aim:**

In this demo, you have to build a classification model to identify fraudulent credit card transactions

### **Dataset Description**

The datasets contains transactions made by credit cards in September 2013 by european cardholders. 

Presents transactions that occurred in two days, where we have **492** frauds out of **284,807** transactions. 

- **Time** - Number of seconds elapsed between this transaction and the first transaction in the dataset
- **V1-V28** - Encrpted attributes (or columns) to protect user identities and sensitive features (v1-v28)
- **Amount** - Transaction Amount
- **Class** - **1** for fraudulent transactions, **0** otherwise

### **Tasks to be performed:**

- Install the required dependencies, import the required libraries and load the data set 
- Perform Exploratory Data Analysis (EDA) on the data set
  -  Generate a Data Report using Pandas Profiling and record your observations
  - Plot **Univariate Distributions**
    - What is the distribution of the **amount** & **class** columns in the data set?
    
- Pre-process that data set for modeling
  - Handle Missing values present in the data set
  - Scale the data set using **RobustScaler()**
  - Split the data into training and testing set using sklearn's **train_test_split** function
- Modelling
  - Build and evaluate a SVM Model
  - Build and evaluate a KNN Model
  - Build and evaluate a Naive Bayes Model

- Model Optimization: Implement **GridSearchCV**
- Model Boosting: Implement **Gradient Boosting** & **XGBoost**
- Dealing with Imbalanced Classes: Re-sampling the data set
- Model Interpertation: Interpret Fraud Detection Model With **Eli5**
- Use **PyCaret** to find the best model and perform Automatic Hyperparameter tuning 

  - Import PyCaret and load the data set
  - Initialize or setup the environment 
  - Compare Multiple Models and their Accuracy Metrics
  - Create the model
  - Tune the model
  - Evaluate the model
- Deploy the model using **Streamlit**
