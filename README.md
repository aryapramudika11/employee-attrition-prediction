# Employee Attrition Prediction — HR Analytics

An end-to-end data science project to predict employee *attrition* (resignation) using an
HR Analytics dataset, covering *data cleaning*, *exploratory data analysis* (EDA), *feature
engineering*, *modeling*, *hyperparameter tuning*, and business recommendations for the HR team.

## 📌 Problem Statement

Employee attrition drives up recruitment costs, onboarding costs, and lost productivity.
This project addresses the question:

> Can we predict whether an employee will resign (`Attrition`) based on their demographic
> attributes, compensation, job satisfaction, and career history — so the HR team can take
> earlier retention action?

## 📂 Dataset

- **Source**: [HR_Analytics-4.xlsx](<https://youtu.be/QcTeeBrL6EY?si=WBgDMQSM0wFUsGnT>)
- **Size**: 1,480 employee rows × 37 columns
- **Target**: `Attrition` (Yes/No) — **16.3% Yes vs 83.7% No** (imbalanced)
- Features include demographics (Age, Gender, MaritalStatus), compensation (MonthlyIncome,
  SalarySlab), job satisfaction (JobSatisfaction, WorkLifeBalance, EnvironmentSatisfaction),
  and career history (YearsatCompany, TotalExperience, NumCompaniesWorked, etc.).

## 📖 Data Dictionary

Explanation of each column in the dataset:

### Identity & Demographics
| Column | Description |
|---|---|
| `EmpID` | Unique employee ID |
| `Age` | Employee's age |
| `AgeGroup` | Age bracket (18-25, 26-35, 36-45, 46-55, 55+) |
| `Gender` | Gender |
| `MaritalStatus` | Marital status (Single/Married/Divorced) |
| `Over18` | Whether the employee is over 18 (constant "Y" — dropped during cleaning) |

### Employment Status
| Column | Description |
|---|---|
| `Attrition` | **Target/label** — whether the employee resigned (Yes/No) |
| `Department` | Department the employee works in (Sales, IT, Finance, etc.) |
| `JobRole` | Job title/role |
| `JobLevel` | Job level (scale 1–5) |
| `BusinessTravel` | Business travel frequency (Non-Travel, Travel_Rarely, Travel_Frequently) |
| `DistanceFromHome(KM)` | Distance from home to office (KM) |
| `EmployeeCount` | Constant column with value 1 (dropped during cleaning) |
| `EmployeeNumber` | Employee identification number (similar to EmpID, dropped during cleaning) |
| `StandardWorkingHours` | Standard working hours (constant 80 — dropped during cleaning) |

### Education
| Column | Description |
|---|---|
| `Education` | Education level (scale 1=Below College to 5=Doctor) |
| `EducationField` | Field of study |

### Compensation
| Column | Description |
|---|---|
| `DailyRate` | Daily rate |
| `HourlyRate` | Hourly rate |
| `MonthlyIncome` | Monthly income |
| `SalarySlab` | Salary bracket (0-3 LPA, 3-6 LPA, 6-10 LPA, 10+ LPA) |
| `SalaryHike %` | Salary increase percentage |

### Satisfaction & Engagement (scale 1–4 unless noted)
| Column | Description |
|---|---|
| `EnvironmentSatisfaction` | Satisfaction with the work environment |
| `JobInvolvement` | Level of engagement in the job |
| `JobSatisfaction` | Job satisfaction |
| `RelationshipSatisfaction` | Satisfaction with relationships with coworkers |
| `WorkLifeBalance` | Work-life balance |
| `PerformanceRating` | Employee performance rating |

### Experience & Work History
| Column | Description |
|---|---|
| `NumCompaniesWorked` | Number of companies previously worked at |
| `TotalExperience(Years)` | Total years of work experience |
| `YearsatCompany` | Years at the current company |
| `YearsinCurrentRole` | Years in the current role |
| `YearsSincePromotion` | Years since the last promotion |
| `YearsWithCurrManager` | Years with the current manager — had 61 missing values, filled with the median during cleaning |
| `TrainingsLastYear` | Number of trainings attended last year |

### Other
| Column | Description |
|---|---|
| `OverTime` | Whether the employee frequently works overtime (Yes/No) — **the #1 predictor of attrition** |
| `StockOptionLevel` | Stock option ownership level (scale 0–3) |

> 💡 The `Over18`, `EmployeeCount`, and `StandardWorkingHours` columns are constant across all
> rows, so they carry no information for the model and were dropped during *data cleaning*.
> `EmpID` and `EmployeeNumber` were also dropped since they are only identifiers, not
> predictive features.

## 🛠️ Methodology

1. **Data Cleaning**
   - Filled missing values in `YearsWithCurrManager` with the median.
   - Dropped constant/uninformative columns: `Over18`, `EmployeeCount`, `StandardWorkingHours`.
   - Dropped identifier columns: `EmpID`, `EmployeeNumber`.

2. **Exploratory Data Analysis (EDA)**
   - Target distribution, attrition rate by Department & JobRole.
   - Relationship between OverTime, WorkLifeBalance, JobSatisfaction, EnvironmentSatisfaction,
     JobInvolvement, StockOptionLevel and attrition.
   - Numeric distributions (Age, Income, Distance, Tenure) vs. attrition.
   - Correlation heatmap across numeric features.

3. **Feature Engineering**
   - Label encoding for binary features, one-hot encoding for nominal features.
   - 80/20 train/test split (stratified).
   - Standardization of numeric features.
   - **SMOTE** (Synthetic Minority Oversampling) applied to the training data to address class
     imbalance — chosen over random undersampling (risks losing data given the small dataset)
     or random oversampling (risks overfitting since it only duplicates existing rows).

4. **Modeling & Tuning**
   - Baselines: Logistic Regression, Random Forest, XGBoost.
   - Hyperparameter tuning with `GridSearchCV` (5-fold Stratified CV), optimized for
     **ROC-AUC** — a metric that is more robust to class imbalance than accuracy, since it
     measures a model's ability to separate the Yes/No classes across all thresholds, not just
     at a single cutoff point like F1-score.

## 📊 Results

| Model                     | F1-score (Yes) | ROC-AUC |
|----------------------------|:--------------:|:-------:|
| Logistic Regression         | 0.515          | 0.834   |
| Random Forest                | 0.394          | 0.845   |
| XGBoost                       | 0.442          | 0.835   |
| Random Forest (Tuned)         | 0.381          | **0.852** |
| XGBoost (Tuned)                | 0.459          | 0.840   |

**Best model: Random Forest (Tuned) with ROC-AUC 0.852**
(`n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1`)

### Top Drivers of Attrition (Feature Importance — XGBoost Tuned)

1. **OverTime** — the most dominant factor
2. **StockOptionLevel**
3. JobRole (Manufacturing Director)
4. **WorkLifeBalance**
5. Department (Operations)
6. JobLevel
7. JobRole (Sales Executive)
8. BusinessTravel (Frequent)

## 💡 Business Recommendations

1. **Manage overtime workload** — re-evaluate workload for roles/departments with high
   OverTime rates, since this is the #1 predictor of attrition.
2. **Review stock option and compensation policy** for levels most prone to resigning.
3. **Retention programs for new hires** — strengthen onboarding & mentoring in the first
   1–2 years.
4. **Improve Work-Life Balance** through flexible working arrangements and better workload
   management.
5. Use this model as an **early-warning system** — assigning an attrition-risk score per
   employee so HR can intervene proactively.

## 📁 Repository Structure

```
├── HR_Attrition_Prediction.ipynb   # Main notebook (end-to-end, already executed)
├── HR_Analytics-4.xlsx         # Raw dataset
└── README.md
```

## ⚙️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
jupyter notebook HR_Attrition_Prediction.ipynb
```

## 🔧 Tools & Libraries

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `XGBoost` ·
`imbalanced-learn (SMOTE)` · `Jupyter Notebook`

## 🚧 Limitations & Future Work

- The dataset is cross-sectional (a single point in time) — the analysis would be stronger
  with multi-period historical data.
- Recall for the "Yes" class is still relatively low (25–71% depending on the model) — this
  could be improved by adjusting the classification threshold or using cost-sensitive
  learning, depending on business priorities (catching more potential leavers vs. reducing
  false alarms).
- Further work: explore interaction features, compare other imbalance-handling techniques
  (undersampling vs. oversampling), and deploy the model via a Streamlit/Flask API with
  model drift monitoring.

---
*This project was built as a Data Scientist portfolio piece.*
