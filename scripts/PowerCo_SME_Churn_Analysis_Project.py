# In[1]:

# --- Core Libraries ---
import numpy as np
import pandas as pd
import datetime as dt
import warnings

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Display Settings ---
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# --- Sklearn: Preprocessing, Splitting & Scaling ---
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

# --- Sklearn: Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# --- Sklearn: Metrics ---
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    recall_score, roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, average_precision_score, roc_curve
)

# --- Statistical Tests ---
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Warning Settings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[2]:

# Load dataset components
clientdf = pd.read_csv('client_data.csv')
pricedf = pd.read_csv('price_data.csv')


# In[3]:

# Preview dataset structures
print("=== Dataset Previews ===")
print("\nClient Data (First 5 rows):")
clientdf.head()


# In[4]:

print("\nPrice Data (First 5 rows):")
pricedf.head()


# In[5]:

# Summary of the datasets
print("\nclientdf Overview:")
print("Shape of data", clientdf.shape)
print("Unique value counts:")
print(clientdf.nunique())


# In[6]:

print("\npricedf Overview:")
print("Shape of data", pricedf.shape)
print("Unique value counts:")
print(pricedf.nunique())


# In[7]:

# Data types
clientdf.info()
pricedf.info()


# In[8]:

# Check for missing entries
missing = clientdf.isnull().mean().sort_values(ascending=False)
print(missing.head(27))


# In[9]:

# Statistical summaries
clientdf.describe()
clientdf.describe(include='object')


# In[10]:

pricedf.describe()
pricedf.describe(include='object')


# In[11]:

# Identify duplicates
dups = clientdf.duplicated()
print("Number of duplicate rows - clientdf: ", dups.sum())

dups = pricedf.duplicated()
print("Number of duplicate rows - pricedf: ", dups.sum())


# In[12]:

# Distribution of target variable
print(clientdf['churn'].value_counts())
clientdf['churn'].value_counts(normalize=True)*100


# In[13]:

# Data Wrangling
clientdf['id'] = clientdf['id'].str[:7]
pricedf['id'] = pricedf['id'].str[:7]


# In[14]:

# Convert date columns
clientdf["date_activ"] = pd.to_datetime(clientdf["date_activ"])
clientdf["date_end"] = pd.to_datetime(clientdf["date_end"])
clientdf["date_modif_prod"] = pd.to_datetime(clientdf["date_modif_prod"])
clientdf["date_renewal"] = pd.to_datetime(clientdf["date_renewal"])
pricedf["price_date"] = pd.to_datetime(pricedf["price_date"])


# In[15]:

# Clean categorical variables
clientdf['channel_sales'] = clientdf['channel_sales'].replace({
    'foosdfpfkusacimwkcsosbicdxkicaua': 'channel_sales_1',
    'MISSING': 'not_specified',
    'lmkebamcaaclubfxadlmueccxoimlema': 'channel_sales_2',
    'usilxuppasemubllopkaafesmlibmsdf': 'channel_sales_3',
    'ewpakwlliwisiwduibdlfmalxowmwpci': 'channel_sales_4',
    'sddiedcslfslkckwlfkdpoeeailfpeds': 'channel_sales_5',
    'epumfxlbckeskwekxbiuasklxalciiuu': 'channel_sales_6',
    'fixdbufsefwooaasfcxdxadsiekoceaa': 'channel_sales_7'
})

clientdf['has_gas'] = clientdf['has_gas'].replace({'f': 'false', 't': 'true'})

clientdf['origin_up'] = clientdf['origin_up'].replace({
    'lxidpiddsbxsbosboudacockeimpuepw': 'code_1',
    'kamkkxfxxuwbdslkwifmmcsiusiuosws': 'code_2',
    'ldkssxwpmemidmecebumciepifcamkci': 'code_3',
    'MISSING': 'not_specified',
    'usapbepcfoloekilkwsdiboslwaxobdp': 'code_4',
    'ewxeelcelemmiwuafmddpobolfuxioce': 'code_5'
})


# In[16]:

# Add tenure column
clientdf['tenure'] = clientdf['date_end'].dt.year - clientdf['date_activ'].dt.year


# In[17]:

# Encode target variable
clientdf['churn'] = clientdf['churn'].replace({0: 'retention', 1: 'churn'})


# In[18]:

# Aggregate price data
pricedf = pricedf.groupby('id', as_index=False)[
    ['price_off_peak_var', 'price_peak_var', 'price_mid_peak_var',
     'price_off_peak_fix', 'price_peak_fix', 'price_mid_peak_fix']
].mean()


# In[19]:

# Merge datasets
df = clientdf.merge(pricedf, on="id", how="inner", validate='one_to_one')
print(f"Merged dataset shape: {df.shape}")
df.head()


# In[20]:

# EDA - Distribution analysis
print("="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

print("\n1. Channel Sales Distribution:")
print(df['channel_sales'].value_counts())

print("\n2. Has Gas Distribution:")
print(df['has_gas'].value_counts())

print("\n3. Churn Distribution:")
print(df['churn'].value_counts())


# In[21]:

# Comparative analysis
print("\nForecast Energy Consumption by Churn Status:")
forecast_comparison = df.groupby('churn')[
    ["forecast_cons_12m", "forecast_cons_year"]
].mean().round(2)
print(forecast_comparison)


# In[22]:

# Data Visualization
print("\nGenerating visualizations...")

plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")

# Figure 1: Subscribed Power Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['pow_max'], bins=20, kde=True)
plt.title('Subscribed Power Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Subscribed Power')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[23]:

# Figure 2: Churn by Tenure
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='tenure', hue='churn')
plt.title('Churn Prevalence by Tenure', fontsize=14, fontweight='bold')
plt.xlabel('Tenure')
plt.ylabel('Count')
plt.legend(title='Churn Status')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[24]:

# Figure 3: Churn by Has Gas
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='has_gas', hue='churn')
plt.title('Churn Prevalence by Gas Subscription', fontsize=14, fontweight='bold')
plt.xlabel('Has Gas')
plt.ylabel('Count')
plt.legend(title='Churn Status')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[25]:

# Figure 4: Correlation Heatmap
plt.figure(figsize=(12, 10))
imputed_numeric_features = df.select_dtypes(['float64','int64','int32']).columns
correlation_matrix = df[imputed_numeric_features].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# In[26]:

# Figure 5: Net Margin by Churn Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='net_margin', data=df)
plt.title('Net Margin Distribution by Churn Status', fontsize=14, fontweight='bold')
plt.xlabel('Churn Status')
plt.ylabel('Net Margin')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[27]:

# Remove redundant columns and encode categorical variables
cols = ['id','date_activ','date_end', 'date_modif_prod', 'date_renewal']
df = df.drop(columns=cols)

le = LabelEncoder()
df['channel_sales'] = le.fit_transform(df['channel_sales'])
df['has_gas'] = le.fit_transform(df['has_gas'])
df['origin_up'] = le.fit_transform(df['origin_up'])


# In[28]:

# Feature selection using VIF
imputed_numeric_features = df.select_dtypes(['float64','int64','int32']).columns
vif = df[imputed_numeric_features]

# Calculate VIF and remove high VIF features
X = vif
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("Initial VIF values:")
print(vif_data)

# Remove high VIF features sequentially
columns_to_remove = ['margin_gross_pow_ele', 'price_off_peak_fix', 'forecast_price_energy_off_peak',
                    'forecast_price_pow_off_peak', 'price_mid_peak_var', 'price_peak_var',
                    'tenure', 'price_peak_fix', 'forecast_cons_year', 'cons_last_month',
                    'price_off_peak_var']

for col in columns_to_remove:
    if col in vif.columns:
        vif = vif.drop(columns=col)
        print(f"Removed {col}")

# Final VIF
X = vif
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nFinal VIF values:")
print(vif_data)


# In[29]:

# Join back target variable
df = vif.join(df['churn'])


# In[30]:

# Prepare features and target
X = df.drop(columns=('churn'))
y = df["churn"]
y = y.map({'churn': 1, 'retention': 0})


# In[31]:

# Split data and scale features
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
rescaledValidationX = scaler.transform(X_test)


# In[32]:

# Evaluate baseline models
num_folds = 3
seed = 42
scoring = 'average_precision'
shuffle = True

models = [
    ('LR', LogisticRegression(class_weight="balanced", random_state=seed)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
    ('NB', GaussianNB()),
    ('SVM', SVC(class_weight="balanced", random_state=seed))
]

print("Baseline Model Performance:")
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=seed)
    cv_results = cross_val_score(model, rescaledX, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[33]:

# Evaluate ensemble models
ensembles = [
    ('AB', AdaBoostClassifier(random_state=seed)),
    ('GBM', GradientBoostingClassifier(random_state=seed)),
    ('RF', RandomForestClassifier(class_weight="balanced", random_state=seed)),
    ('ET', ExtraTreesClassifier(class_weight="balanced", random_state=seed)),
    ('XGBC', XGBClassifier(random_state=seed)),
    ('HGBC', HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)),
    ('LGBC', LGBMClassifier(class_weight="balanced", verbose=-1, random_state=seed))
]

print("\nEnsemble Model Performance:")
results = []
names = []
for name, model in ensembles:
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=seed)
    cv_results = cross_val_score(model, rescaledX, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[34]:

# # Hyperparameter tuning for Random Forest
# n_estimators = [100, 200, 500, 1000]
# max_depth = [None, 1, 3, 5, 7]
# param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)

# model = RandomForestClassifier(random_state=seed, class_weight="balanced")
# kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
# grid_result = grid.fit(rescaledX, y_train)

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[35]:

# Train final model
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)
rf.fit(rescaledX, y_train)

# Predict probabilities
y_proba = rf.predict_proba(rescaledValidationX)[:, 1]

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
ap_score = average_precision_score(y_test, y_proba)

f1_scores = 2 * (precisions[1:] * recalls[1:]) / (precisions[1:] + recalls[1:] + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print("Average Precision (AUC-PR):", round(ap_score, 3))
print(f"Best Threshold: {best_threshold:.3f}")
print(f"Precision: {precisions[best_idx+1]:.3f}, Recall: {recalls[best_idx+1]:.3f}, F1 Score: {f1_scores[best_idx]:.3f}")


# In[36]:

# Plot Precision-Recall curve
plt.figure(figsize=(7, 6))
plt.plot(recalls, precisions, label=f'PR curve (AP={ap_score:.3f})')
plt.scatter(recalls[best_idx+1], precisions[best_idx+1], color='red', label=f'Best thr={best_threshold:.3f}')
plt.axvline(x=recalls[best_idx+1], color='red', linestyle='--')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid(True)
plt.show()


# In[37]:

# Evaluate with optimal threshold
y_pred_best = (y_proba >= best_threshold).astype(int)
print(f"\n=== Best Threshold = {best_threshold:.3f} ===")
print(classification_report(y_test, y_pred_best, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))


# In[38]:

# Performance comparison function
def evaluate_threshold(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recalls, precisions)
    f1 = f1_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)

    metrics = {
        "Balanced Accuracy": balanced_acc,
        "Sensitivity (Churn)": sensitivity,
        "Specificity (No Churn)": specificity,
        "Precision": precision_val,
        "Recall": recall_val,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }
    return metrics

# Compare thresholds
metrics_05 = evaluate_threshold(y_test, y_proba, threshold=0.5)
metrics_best = evaluate_threshold(y_test, y_proba, threshold=best_threshold)

comparison = pd.DataFrame({
    "Threshold = 0.5": metrics_05,
    f"Threshold = {best_threshold:.3f}": metrics_best
})

print("\nCLASSIFICATION PERFORMANCE COMPARISON")
print(comparison.round(4).to_string())