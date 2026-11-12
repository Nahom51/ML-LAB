import pandas as pd
import os 

# ---------- Task 1: Load and Inspect ----------
df = pd.read_csv("loan_applications.csv")
print(df.shape)
print(df.dtypes)
print(df.head()) 
# ANSWERS:
# The dataset has 305 rows and 8 columns.
# applicant_id, monthly_income_etb and loan_amount_etb are read as numeric.
# The other five columns (has_collateral, employment_type, application_channel,
# credit_history_flag, defaulted) are read as text.
# ---------- Task 2: Descriptive Statistics ----------
print("Mean:", df["loan_amount_etb"].mean())
print("Median:", df["loan_amount_etb"].median())
print("Std (population):", df["loan_amount_etb"].std(ddof=0))  # ddof=0 = divide by n
print("Mode:", df["credit_history_flag"].mode())

# ANSWERS (Task 2):
# Mean loan amount = 19,352.33 ETB
# Median loan amount = 18,070 ETB
# Population standard deviation = 9,424.59 ETB
# Mode of credit_history_flag = Good
# Mode makes more sense for credit_history_flag because it is a categorical
# column (Good/Fair/Poor), so you cannot average it, and the most frequent
# category is the useful summary. loan_amount_etb is continuous, so almost
# every value is different and a mode would tell us very little.


# ---------- Task 3: A Veracity Check ----------
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df_clean = df.drop_duplicates()
print(df_clean.shape)

# ANSWERS (Task 3):
# Missing values: monthly_income_etb has 13 and has_collateral has 6.
# Exact duplicate rows: 5 (after dropping them, 300 rows remain).
# Combined problem rate = (13 + 6 + 5) / 305 = 24 / 305 = about 7.87%.
# ---------- Task 4: The 5 Vs of This Dataset ----------
print("File size (bytes):", os.path.getsize("loan_applications.csv"))
print("Rows per year at 50/day:", 50 * 365)

# ANSWERS (Task 4):
# Volume: the file is only about 14 KB (14,203 bytes). This is not big data by
# itself, because it is tiny and fits easily in memory on one laptop.
# Velocity: 50 applications/day x 365 = 18,250 rows per year. That is still
# small for a spreadsheet, so you would need many years of accumulation
# (hundreds of thousands to about a million rows) before a spreadsheet
# starts to struggle.
# Variety: every column here is structured. A real bank loan file would also
# include unstructured data such as scanned ID cards, signed contracts, or
# photos of the collateral.
# Veracity: the data has 13 missing incomes, 6 missing collateral answers and
# 5 exact duplicate rows (about 7.87% of rows with problems), so statistics
# should be checked or cleaned before being trusted.
# Value: the bank could set a collateral policy, such as requiring collateral
# for riskier applicants, because the data shows applicants without collateral
# default more often (see Task 5).


# ---------- Task 5 (stretch): Conditional Probability ----------
p_default = (df["defaulted"] == "Yes").mean()
print("P(Default):", p_default)

rate_by_collateral = df.groupby("has_collateral")["defaulted"].apply(
    lambda x: (x == "Yes").mean()
)
print(rate_by_collateral)

# ANSWERS (Task 5):
# P(Default) = 0.193 (about 19.3%)
# P(Default | Has Collateral = Yes) = 0.121 (about 12.1%)
# P(Default | Has Collateral = No) = 0.260 (about 26.0%)
# Yes, the Unit 1 pattern holds: applicants without collateral default more
# often, by about 14 percentage points (roughly 2.2 times the rate).