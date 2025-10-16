# ⚡ PowerCo Client Churn & Price Analysis (Python + Streamlit)

This project explores why clients leave **PowerCo**, a European energy supplier, and how pricing shapes retention.  
Built with **Python** and **Streamlit**, it blends data analysis, machine learning, and interactive dashboards to turn raw data into clear business insights.

The app lets users explore churn trends, test pricing effects, and view live model predictions — all in one place.  
Adapted from the **Forage PowerCo Churn Analysis**, this version brings the work to life through automation, visualization, and real-time exploration.



---

## 📁 Project Directory Structure — PowerCo Energy & Gas Churn Analysis

The project is organized to make each stage of the churn analysis — from data sourcing and model training to visualization and deployment — clear and traceable.  
It follows a logical flow that connects data preparation, modeling, and insights through both static and interactive outputs.

---

| Folder / File | Description |
|----------------|--------------|
| **`artifacts/`** | Stores serialized machine learning artifacts used for prediction and deployment. |
| ├── `powerco_churn_model.pkl` | Trained machine learning model for churn prediction. |
| ├── `scaler.pkl` | Scaler object used for feature normalization during preprocessing. |
| └── `README.md` | Notes describing model details and artifact usage. |

---

| **`data/`** | Contains both raw and processed datasets used in the churn analysis. |
| ├── **`raw/`** | Original unmodified datasets. |
| │ ├── `client_data.csv` | Client-level dataset including company, energy, gas usage, and churn info. |
| │ ├── `price_data.csv` | Contract pricing and margin details for each client. |
| │ └── `README.md` | Notes describing data sources and variables. |
| ├── **`processed/`** | Cleaned and structured datasets used for analysis and modeling. |
| │ ├── `PowerCo_analysis.csv` | Cleaned dataset used for descriptive analysis and visualization. |
| │ ├── `PowerCo_ml.csv` | Processed dataset prepared for machine learning model training. |
| │ └── `README.md` | Overview of transformation and preprocessing steps. |

---

| **`docs/`** | Supporting documentation for the PowerCo project. |
| ├── `Data Description.pdf` | Complete data dictionary explaining variables and dataset design. |
| └── `README.md` | Summary of included documents and their purpose. |

---

| **`notebook/`** | Jupyter notebook environment used for analysis and exploration. |
| ├── `PowerCo SME Churn Analysis Project.ipynb` | Full notebook containing exploratory analysis, feature engineering, and model development. |
| └── `README.md` | Notes on notebook workflow and purpose. |

---

| **`scripts/`** | Python scripts supporting app logic, model building, and prediction pipelines. |
| ├── `app.py` | Main Streamlit app for interactive churn prediction and dashboard visualization. |
| ├── `model.py` | Script handling model training, evaluation, and persistence. |
| ├── `PowerCo_SME_Churn_Analysis_Project.py` | Core Python workflow combining data loading, model training, and evaluation. |
| └── `README.md` | Description of each script and its role in the workflow. |

---

| **`visualization/`** | Contains static charts and UI screenshots from the analysis and Streamlit app. |
| ├── **`charts/`** | Analytical visuals and performance metrics. |
| │ ├── `Churn Prevalence by Gas Subscription.png` | Comparison of churn rates based on gas service subscription. |
| │ ├── `Churn Prevalence By Tenure.png` | Visualization showing churn trends by customer tenure group. |
| │ ├── `Net Margin Distribution by Churn Status.png` | Distribution of client net margins segmented by churn status. |
| │ ├── `Precision-Recall Curve.png` | Model evaluation curve showing tradeoff between precision and recall. |
| │ ├── `Subscribed_Power_Distribution.png` | Breakdown of subscribed power across different churn categories. |
| │ └── `README.md` | Summary of generated charts and insights. |
| ├── **`screenshots/`** | Interface previews from the Streamlit app and dashboard. |
| │ ├── `about.png` | About section of the Streamlit app. |
| │ ├── `batch prediction.png` | Batch prediction interface for multiple clients. |
| │ ├── `dashboard.png` | Main dashboard displaying churn insights and KPIs. |
| │ ├── `single customer.png` | Single customer prediction interface. |
| │ └── `README.md` | Notes describing the screenshots and their context. |

---

| **`requirements.txt`** | List of Python dependencies required to run the project and Streamlit app. |
| **`README.md`** | Main project file summarizing the objectives, methods, and key results. |

---

## 📚 Table of Contents  

A quick guide through the PowerCo Energy & Gas Churn Analysis journey — from raw data to interactive insights.  
The layout is designed for clarity, helping readers navigate data sources, analysis stages, and key business findings with ease.  
It’s straightforward, transparent, and built to tell the full story behind customer churn.

- [⚡ PowerCo Energy & Gas Churn Analysis](#-powerco-energy--gas-churn-analysis)  
- [📁 Project Directory Structure — PowerCo Energy & Gas Churn Analysis](#-project-directory-structure--powerco-energy--gas-churn-analysis)  
- [✴️ Summary](#✴️-summary)  
- [📊 Project Overview](#-project-overview)  
- [🎯 Executive Summary – PowerCo SME Churn Analysis](#-executive-summary--powerco-sme-churn-analysis)  
- [⚙️ Data Preparation & Methodology](#⚙️-data-preparation--methodology)  
  - [🔧 Data Integration & Cleaning](#🔧-data-integration--cleaning)  
- [📊 Project Workflow](#-project-workflow)  
- [📈 Dashboard Previews](#-dashboard-previews)  
- [📊 Excel Workbook Structure](#-excel-workbook-structure)  
  - [📄 Download the Full Excel Project](#📄-download-the-full-excel-project)  
- [📈 Key Findings](#-key-findings)  
  - [💼 Client Base Health](#💼-client-base-health)  
  - [📈 Sales Channel Performance](#📈-sales-channel-performance)  
  - [🔀 Service Mix](#🔀-service-mix)  
- [🔍 Churn & Tenure Insights](#🔍-churn--tenure-insights)  
- [💶 Pricing & Margin Analysis](#💶-pricing--margin-analysis)  
- [🚀 Strategic Recommendations](#🚀-strategic-recommendations)  
  - [⚡ Short-Term Actions](#⚡-short-term-actions)  
  - [🏗️ Long-Term Initiatives](#🏗️-long-term-initiatives)  
- [💼 Business Impact Projection](#💼-business-impact-projection)  
- [⚡ PowerCo Churn Dashboard — Summary & Visualization Report](#⚡-powerco-churn-dashboard--summary--visualization-report)  
  - [📊 Visual Charts (Summary)](#📊-visual-charts-summary)  
  - [📈 Visual Charts (Details)](#📈-visual-charts-details)  
  - [💡 KPI Summary](#💡-kpi-summary)  
  - [✴️ Reflection](#✴️-reflection)  
- [🧠 Tools & Skills Applied](#🧠-tools--skills-applied)  
- [💡 Lessons Learned](#💡-lessons-learned)  
- [🔮 Next Steps](#🔮-next-steps)  
- [🧾 Project Summary](#🧾-project-summary)  
- [⚙️ Installation](#⚙️-installation)  
- [✴️ Project Impact](#✴️-project-impact)  
- [🙋‍♂️ Author](#🙋‍♂️-author)  


---

## 📊 Project Overview

PowerCo, a leading energy provider, sought to understand why some clients terminate their services (“churn”). 
This project analyzes customer and pricing data to identify the key drivers of churn and predict which clients are most at risk.  
The insights help PowerCo improve **customer retention**, **pricing strategies**, and **overall profitability**.  
The full analysis was developed in **Python** and visualized through an interactive **Streamlit dashboard**.


---

## 🎯 Problem Statement
PowerCo, a major energy provider, faces significant customer attrition in a competitive market. With only 10% churn rate but high-value clients at risk, 
the company needed data-driven insights to identify at-risk customers and implement proactive retention strategies to reduce revenue loss.

---

## 🐍 Tools & Technologies  
| **Category** | **Libraries/Frameworks** |
|---------------|---------------------------|
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Modeling | `scikit-learn` |
| Deployment | `Streamlit` |
| Utility | `os`, `warnings`, `datetime` |

---

## 📦 Files Included  
- `client_data.csv` — Client demographic and service details.  
- `price_data.csv` — Pricing, consumption, and margin data.  
- `PowerCo_Churn_Analysis.ipynb` — Main Python analysis notebook.  
- `model.pkl` — Saved trained Random Forest model.  
- `app.py` — Streamlit dashboard script.  
- `Visualizations/` — Folder containing all project charts and screenshots.

---

# PowerCo Churn Analysis Overview

## Overview

This project analyzes customer churn behavior for PowerCo, an energy and power provider. The goal is to identify key factors driving customer attrition and to build predictive models that help PowerCo proactively retain at-risk clients. The analysis focuses on energy consumption, pricing, and customer engagement patterns.

Data provided by PowerCo (synthetic data for analytical purposes). The analysis is conducted using Python — leveraging **pandas**, **scikit-learn**, **matplotlib**, and **RandomForest** for data cleaning, visualization, and predictive modeling.

This dataset originates from the **BCG Data Science Job Simulation on Forage (August 2025)**, where I obtained a completion certificate. I further refined and expanded the dataset to make the analysis more engaging and impactful.

---

##  🎯 Datasets

This section provides a summary of the datasets used in the **PowerCo SME Churn Analysis** project, as documented in the *Data Description.pdf* file.  
The analysis is based on two main datasets: **client_data.csv** and **price_data.csv**, both containing information about customer behavior, energy usage, and pricing.

Two datasets were used in this analysis:

> [📄 View Raw Dataset (CSV)](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/docs/data_description.pdf)

### 🧾 client_data.csv

This dataset contains one record per client and includes information about their energy usage, contract activity, and churn status.

**Main columns:**
- **id** – Unique client identifier  
- **channel_sales** – Sales channel code  
- **cons_12m / cons_gas_12m / cons_last_month** – Electricity and gas consumption metrics  
- **date_activ / date_end / date_modif_prod / date_renewal** – Contract start, end, modification, and renewal dates  
- **forecast_cons_12m / forecast_cons_year** – Forecasted electricity consumption for future periods  
- **forecast_discount_energy / forecast_meter_rent_12m** – Predicted discounts and meter rental costs  
- **forecast_price_energy_off_peak / forecast_price_energy_peak / forecast_price_pow_off_peak** – Forecasted energy and power prices  
- **has_gas** – Indicates if the client also uses gas services  
- **imp_cons** – Current paid consumption  
- **margin_gross_pow_ele / margin_net_pow_ele / net_margin** – Profit and margin-related metrics  
- **nb_prod_act** – Number of active products  
- **num_years_antig** – Years of customer relationship  
- **origin_up** – Code for the initial campaign the client joined from  
- **pow_max** – Subscribed maximum power  
- **churn** – Target variable (1 = churned, 0 = retained)

### 💰 price_data.csv

This dataset provides energy and power pricing by time period for each client.

**Main columns:**
- **id** – Client identifier (links to `client_data.csv`)  
- **price_date** – Reference date for pricing  
- **price_off_peak_var / price_peak_var / price_mid_peak_var** – Variable energy prices for different time periods  
- **price_off_peak_fix / price_peak_fix / price_mid_peak_fix** – Fixed power prices for different time periods  


Some fields contain hashed text strings to preserve client privacy while retaining predictive meaning.

- **Data Description.pdf** — Provides clear definitions and descriptions of all variables used in both the client and price datasets, ensuring consistent understanding throughout the analysis.  

> [📄 View Data Dictionary (PDF)](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/docs/data_description.pdf)


---

## 📊 Project Workflow  

**Data Pipeline**: Client Data + Price Data → Data Cleaning & Preparation → Feature Engineering & Selection → EDA & Visualization → Model Development & Threshold Optimization → Visualization & Deployment 

1. **Data Import & Integration**  
   - Loaded and merged the `client_data.csv` and `price_data.csv` files using pandas.  
   - Unified both datasets through a common client identifier for analysis.

2. **Data Cleaning & Preparation**  
   - Handled missing values and outliers.  
   - Normalized numerical columns for uniform scaling.  
   - Converted date fields to `datetime` format.  
   - Encoded categorical fields (`channel_sales`, `has_gas`, `origin_up`).  
   - Created derived features such as **tenure** and **average pricing**.

3. **Feature Engineering & Selection**  
   - Constructed metrics like net margin ratios and 12-month consumption averages.  
   - Used **Variance Inflation Factor (VIF)** and correlation heatmaps to remove redundant features.  

4. **Exploratory Data Analysis (EDA)**  
   - Visualized customer churn by tenure, gas subscription, and channel type.  
   - Analyzed relationships between power consumption, pricing, and churn.  
   - Identified key behavioral trends linked to customer retention.

5. **Predictive Modeling**  
   - Built **Random Forest** classifiers to predict churn.  
   - Balanced data classes using resampling techniques.  
   - Tuned hyperparameters with GridSearchCV for optimal model performance.  
   - Evaluated model accuracy, recall, and feature importance.

6. **Visualization & Deployment**  
   - Created interactive charts and churn dashboards with **Streamlit**.  
   - Summarized client risk profiles and retention recommendations.



> [📊 View processed data (CSV)](https://github.com/rotimi2020/Data-Analyst-Portfolio/tree/main/powerco_churn_analysis)

---
## 📈 Screenshot Previews  

| Churn Dashboard | Pivot Table | Summary Insights |
|------------------|-------------|------------------|
| ![Churn Dashboard](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_dashboard.png) | ![Pivot Table](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_pivot.png) | ![Summary Insights](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_summary.png) |

---

## 📈 Python Overview & Visualizations — PowerCo Churn Analysis  

This section highlights how Python was used for **data exploration, visualization, and modeling** in the PowerCo Churn Analysis project.  
It leverages key libraries — **pandas**, **NumPy**, **matplotlib**, **seaborn**, and **scikit-learn** — for efficient transformation, analysis, and churn prediction.  
Both **Random Forest** were explored for robust classification results.

---

### 🧹 Data Cleaning & Preparation
- **Data Integration:** Merged client demographic data (14,606 records) with aggregated pricing information.  
- **Feature Engineering:** Derived tenure from contract dates and encoded categorical variables.  
- **Multicollinearity Treatment:** Removed 10 high-VIF features through manual selection.  
- **Scaling:** Applied `StandardScaler` to ensure model training compatibility.  

**Key Transformations:**
- Label encoded: `channel_sales`, `has_gas`, `origin_up`  
- Derived date feature: tenure from contract start/end periods  
- Aggregated pricing metrics at the client level  


> [📄 View Processed Dataset (CSV)](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/docs/data_description.pdf)

---

### 📊 Visualizations

#### 1. Churn Prevalence by Gas Subscription
Clients **without gas service** churn more often, revealing strong cross-sell potential.  

```python
# Figure 1: Churn by Gas Subscription
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='has_gas', hue='churn')
plt.title('Churn Prevalence by Gas Subscription', fontsize=14, fontweight='bold')
plt.xlabel('Has Gas Service')
plt.ylabel('Client Count')
plt.legend(title='Churn Status')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

```

![Churn Prevalence by Gas Subscription](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_dashboard.png)

---

- **Churn Prevalence by Tenure.png** — The 3–5 year tenure group shows the highest churn rate, suggesting mid-term customers are more volatile.

```python
# Figure 2: churn by tenure
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='tenure', hue='churn')
plt.title('churn Prevalence by tenure', fontsize=14, fontweight='bold')
plt.xlabel('tenure')
plt.ylabel('Count')
plt.legend(title='churn Status')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

```

![Churn Prevalence by Tenure](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_dashboard.png)

---

- **Feature Correlation Matrix.png** — Displays correlations between **margin, tenure, and churn rate**. 

```python
# Figure 3: Correlation Heatmap
plt.figure(figsize=(12, 10))
imputed_numeric_features = df.select_dtypes(['float64','int64','int32']).columns
imputed_numeric_features = imputed_numeric_features
correlation_matrix = df[imputed_numeric_features].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

```

![Feature Correlation Matrix](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_dashboard.png)

---

- **Net Margin Distribution by Churn Status.png** — Box plot comparing **profit margins** of churned vs. retained clients, highlighting greater stability among loyal ones.  

```python
# Figure 4: Total Net Margin by Churn Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='net_margin', data=df)
plt.title('Total Net Margin Distribution by churn Status', fontsize=14, fontweight='bold')
plt.xlabel('churn Status')
plt.ylabel('Total Net Margin')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

```


![Net Margin Distribution by Churn Status](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_dashboard.png)

---

- **Precision-Recall Curve.png** — Visualizes the model’s trade-off between **accuracy and recall** in predicting churn.

```python
# Figure 5: Plot precision-recall curve for Random Forest
plt.figure(figsize=(7, 6))
plt.plot(recalls, precisions, label=f'PR curve (AP={ap_score:.3f})')
plt.scatter(recalls[best_idx+1], precisions[best_idx+1], color='red', label=f'Best thr={best_threshold:.3f}')
plt.axvline(x=recalls[best_idx+1], color='red', linestyle='--')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (with best threshold)")
plt.legend()
plt.grid(True)
plt.show()

```


![recision-Recall Curve](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_dashboard.png)

---

- **Subscribed_Power_Distribution.png** — Shows how **consistent power consumption** relates to higher retention.  

```python
# Figure 6: Subscribed Power Distribution
plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")
plt.figure(figsize=(10, 6))
sns.histplot(df['pow_max'], bins=20, kde=True)
plt.title(' Subscribed Power Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Subscribed Power')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

```

![Subscribed_Power_Distribution](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_dashboard.png)

---

📁 **Download all visualization charts here:**  
[🔗 Download Charts](https://github.com/rotimi2020/Data-Analyst-Portfolio/tree/main/powerco_churn_analysis/visuals)


---

## 📈 Visualization Summary

### Visual Types Deployed
- **Count Plots**: Categorical feature distributions (channels, gas subscription)
- **Box Plots**: Numerical feature spread by churn status (margins, consumption)
- **Correlation Heatmaps**: Feature relationship matrices
- **Histograms**: Consumption pattern distributions
- **Precision-Recall Curves**: Model performance optimization


**Model Insights Interface**:
- Feature importance rankings showing margin metrics dominance over pricing
- Precision-recall curve with optimal threshold identification
- Performance comparison across multiple classification thresholds
- Business impact visualization of false positive/negative trade-offs

### Key Visual Insights
- **Churn Distribution**: 10% class imbalance highlighting retention challenge
- **Financial Patterns**: Churned clients showed 15% higher average margins
- **Consumption Trends**: Heavy users (pow_max) more likely to churn
- **Channel Performance**: Sales channel 1 dominated but had highest attrition
- **Temporal Patterns**: 3-7 year tenure clients most volatile


---


## 🔧 Technical Implementation

### Python Ecosystem
**Core Libraries**:
- pandas, numpy (data manipulation)
- scikit-learn (ML pipeline & evaluation)
- matplotlib, seaborn (visualization)
- statsmodels (VIF analysis)

**Methodology**:
- Stratified train-test splits (80-20)
- 3-fold cross-validation
- GridSearch for hyperparameter tuning
- Precision-recall curve analysis

---
### Python Predictive Modeling
**Model Selection**: Random Forest Classifier (n_estimators=1000, class_weight='balanced')

**Performance Highlights**:
- **Optimal Threshold**: 0.162 (vs default 0.5)
- **Recall Improvement**: 4.6% → 38.7%
- **F1-score Increase**: 0.087 → 0.309875` - **Balanced Accuracy**: 63.2%
- **ROC-AUC**: 0.681

**Feature Importance Top 5**:
1. margin_net_pow_ele (0.138) - Electricity profit margin
2. cons_12m (0.133) - 12-month consumption history
3. forecast_meter_rent_12m (0.121) - Projected meter rental
4. net_margin (0.111) - Overall profitability
5. forecast_cons_12m (0.110) - Expected future consumption

---


## 🧩 Modeling Overview
This part of the project focuses on predicting which PowerCo clients are most likely to stop their services.  
Using the company’s historical consumption and pricing data, I trained a **Random Forest Classifier** to spot churn patterns.  
Before modeling, the data was scaled, encoded, and balanced to handle the uneven churn distribution.  
I also fine-tuned the decision threshold to make sure the model caught as many at-risk clients as possible without too many false alarms.


---

## 🎯 Objective
- Predict customer churn probability.  
- Improve model sensitivity and overall balance using optimal threshold selection.  
- Evaluate trade-offs between precision and recall for business insights.

---

## 🧠 Model Used
- **Algorithm:** Random Forest Classifier  
- **Data Split:** 80% training / 20% testing  
- **Evaluation Metric Focus:** F1-Score, Precision, Recall, ROC-AUC, PR-AUC  
- **Rescaling:** StandardScaler  
- **Label Encoding:** Applied to categorical columns  

---

## 📊 Classification Reports

### Threshold = 0.50 (Default)

          precision    recall  f1-score   support
       0      0.907     0.999     0.951      2638
       1      0.812     0.046     0.087       284

accuracy                          0.906      2922


Confusion Matrix:
[[2635 3]
[ 271 13]]

ROC AUC: 0.522



---

### Threshold = 0.162 (Best Threshold)



          precision    recall  f1-score   support
       0      0.930     0.876     0.902      2638
       1      0.252     0.387     0.306       284

accuracy                          0.829      2922


Confusion Matrix:
[[2312 326]
[ 174 110]]

ROC AUC: 0.632


---

## 🧾 Precision-Recall Trade-Off (Sample)
| Threshold | Precision | Recall | F1 Score |
|------------|------------|---------|-----------|
| 0.00 | 0.097 | 1.000 | 0.177 |
| 0.05 | 0.115 | 0.831 | 0.202 |
| 0.10 | 0.165 | 0.606 | 0.259 |
| 0.15 | 0.237 | 0.426 | 0.304 |
| 0.20 | 0.291 | 0.282 | 0.286 |
| 0.50 | 0.813 | 0.046 | 0.087 |
| 0.162 *(best)* | **0.254** | **0.387** | **0.307** |

**Average Precision (AUC-PR): 0.258**  
**Best Threshold: 0.162**

---

## 🔍 Classification Performance Comparison (Threshold 0.5 vs 0.162)
| Metric | Threshold = 0.5 | Threshold = 0.162 | Change (0.162 vs 0.5) |
|:--|:--:|:--:|:--:|
| **Balanced Accuracy** | 0.5223 | 0.6319 | ↑ 0.1096 |
| **Sensitivity (Churn)** | 0.0458 | 0.3873 | ↑ 0.3415 |
| **Specificity (No Churn)** | 0.9989 | 0.8764 | ↓ 0.1224 |
| **Precision** | 0.8125 | 0.2523 | ↓ 0.5602 |
| **Recall** | 0.0458 | 0.3873 | ↑ 0.3415 |
| **F1 Score** | 0.0867 | 0.3056 | ↑ 0.2189 |
| **ROC-AUC** | 0.6807 | 0.6807 | — |
| **PR-AUC** | 0.2572 | 0.2572 | — |
| **TP** | 13 | 110 | — |
| **FP** | 3 | 326 | — |
| **TN** | 2635 | 2312 | — |
| **FN** | 271 | 174 | — |

---

## 💡 Insights
- Default threshold (0.5) achieved high precision but **missed most churners** (Recall = 0.046).  
- Adjusting threshold to **0.162** significantly improved recall to **0.387**, detecting more at-risk clients.  
- Although precision dropped, the F1-score improved from **0.0867 → 0.3056**, balancing false positives and negatives.  
- ROC-AUC remained stable (≈0.68), confirming consistent ranking ability.  
- The model is better suited for **early churn detection**, where recall is more critical than precision.

---

## 🧭 Business Recommendation
- Use **threshold = 0.162** to flag potential churners for retention campaigns.  
- Target these customers with personalized incentives or rate adjustments.  
- Continue improving model with:
  - Additional behavioral or payment data.
  - Cost-sensitive learning or SMOTE resampling.
  - Model explainability tools (e.g., SHAP values).

---

## 🌐 Streamlit Dashboard — PowerCo Churn Analysis App

This project also includes an interactive **Streamlit dashboard** built with Python.  
The app combines both **visual insights** and **machine learning predictions**, giving users an easy way to explore customer churn trends and test predictions without touching code.

### 🔹 Features
- **Interactive Visualizations:**  
  Explore churn distribution by tenure, gas subscription, and margin levels.  
  Charts update dynamically based on user selections.
- **Customer Churn Prediction:**  
  Enter client data (e.g., consumption, margin, tenure) and get an instant churn probability based on the trained Random Forest model (`model.pkl`).
- **Intuitive Interface:**  
  Built with a clean layout for non-technical users to quickly interpret PowerCo’s churn drivers.
- **Future Deployment:**  
  The dashboard isn’t live yet, but it will be deployed soon on **Streamlit Community Cloud** for public access.
  
---

### 📸 Streamlit Dashboard Preview 

| Churn Dashboard | Pivot Table | Summary Insights |
|------------------|-------------|------------------|
| ![Churn Dashboard](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_dashboard.png) | ![Pivot Table](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_pivot.png) | ![Summary Insights](https://github.com/rotimi2020/Data-Analyst-Portfolio/blob/main/powerco_churn_analysis/report_screenshots/powerco_summary.png) |

### 🧩 File Reference
- `app.py` — Streamlit application script  
- `model.pkl` — Serialized Random Forest model used for live churn predictions  
- `Visualizations/` — Contains screenshots and static chart exports

> _The Streamlit dashboard turns the analysis into a hands-on tool — bridging the gap between data science and business decisions.

---

## 🏁 Final Thoughts
This part of the project helped PowerCo get better at spotting which customers might leave before they actually do.  
The model isn’t perfect yet—recall could still improve—but it’s a solid step toward using data to guide real retention efforts.  
The next move is to fold these predictions into PowerCo’s day-to-day strategy and see how early outreach impacts customer loyalty over time.


<h2 id="installation"> ⚙️ Installation </h2>

To set up the project environment on your local machine, follow these steps:

### ✅ Step : Clone the Repository

```bash
git clone https://github.com/rotimi2020/Data-Analyst-Portfolio.git
cd Data-Analyst-Portfolio/Diabetes_Analysis

```

---

### 🚀 Project Impact

This project demonstrates my ability to deliver end-to-end data solutions—transforming raw data into actionable business intelligence. Through machine learning and interactive dashboards, I've created practical tools that help businesses understand and reduce customer churn.

The clean, documented structure ensures technical work remains accessible and actionable for both technical teams and business stakeholders.

---

<h2 id="author"> 🙋‍♂️ Author </h2>

**Rotimi Sheriff Omosewo**  
📧 Email: [omoseworotimi@gmail.com](mailto:omoseworotimi@gmail.com)  
📞 Contact: +234 903 441 1444  
🔗 LinkedIn: [linkedin.com/in/rotimi-sheriff-omosewo-939a806b](https://www.linkedin.com/in/rotimi-sheriff-omosewo-939a806b)  
📁 Project GitHub: [github.com/rotimi2020/Data-Analyst-Portfolio](https://github.com/rotimi2020/Data-Analyst-Portfolio)  

---

*Thank you for reviewing my work. I welcome opportunities to discuss how I can bring this same analytical approach and business mindset to your team.*