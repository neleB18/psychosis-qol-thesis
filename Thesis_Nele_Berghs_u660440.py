#%%-----------------------------------------------------------------------------------------------------
#Importing the needed packages / libraries 
import hashlib
import pyreadr  
import os
import re
import gc,time
import math
import pandas as pd 
from nacl.secret import SecretBox
from collections import OrderedDict
from typing import Dict, Any
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  
from scipy.stats import pearsonr, stats
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
import missingno as msno 
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNet
from glob import glob
import glob as glob_module
from IPython.display import display
from sklearn.impute import SimpleImputer, IterativeImputer
import shap
import scipy.stats as sps
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from fairlearn.metrics import MetricFrame
from pathlib import Path
from scipy.stats import shapiro, skew, kurtosis, probplot
#%%---------------------------------------------------------------------------------------------------
#Making the base dir for all the files and images
BASE_DIR = Path(__file__).resolve().parent
def p(*args):
    return os.path.join(BASE_DIR, *args)
#%%-------------------------------------------------------------------------------
#Decrypting the file (not needed for the repository):
#The paths
cipher_path = p("WideEncrypted.csv")
nonce_path  = p("nonce.bin")
#The paths that need to be made:
out_rds = p("data_decrypted.rds")
out_csv = p("data_decrypted.csv")
#Decrypting the file (only once)
if not os.path.exists(out_rds):
    password = input("Enter password: ").encode("utf-8")
    key = hashlib.blake2b(password, digest_size=32).digest()
    box = SecretBox(key)
    with open(cipher_path, "rb") as f: cipher = f.read()
    with open(nonce_path, "rb") as f: nonce = f.read()   
    plaintext = box.decrypt(cipher, nonce)
    with open(out_rds, "wb") as f:
        f.write(plaintext)
#Load decrypted dataset and getting the first information
res: OrderedDict[str, pd.DataFrame] = pyreadr.read_r(out_rds)   
df: pd.DataFrame = next(iter(res.values()))     
#%%-------------------------------------------------------------------------------------------------------
#Preprocessing the data:
#Some columns have been swapped, we need to fix this with the old dataset of Altrecht
old_cipher_path = p("OldWideEncrypted.csv")
old_nonce_path  = p("Oldnonce.bin")
old_out_rds = p("old_data_decrypted.rds")
processed_data = p("processed_data.csv")
#Decrypting the old file (only once)
if not os.path.exists(old_out_rds):
     password = input("Enter password: ").encode("utf-8")
     key = hashlib.blake2b(password, digest_size=32).digest()
     box = SecretBox(key)
     with open(old_cipher_path, "rb") as f: cipher = f.read()
     with open(old_nonce_path, "rb") as f: nonce = f.read()   
     plaintext = box.decrypt(cipher, nonce)
     with open(old_out_rds, "wb") as f:
        f.write(plaintext)
#Loading the old dataset
old_res: OrderedDict[str, pd.DataFrame] = pyreadr.read_r(old_out_rds)    
old_df: pd.DataFrame = next(iter(old_res.values()))
#Selecting the colums that need to be switched
columns_to_be_switched = ["jaaringeplandgegevensafname","maandingeplandgegevensafname","jaarafgerondgegevensafname","maandafgerondgegevensafname"]
#Build full list of expected columns (1-6 per base column)
old_cols = (["Proefpersoonnummer"] + [f"{b}.{i}" for b in columns_to_be_switched for i in range(1, 7)])
#Setting the numbers to strings in both sets
df["Proefpersoonnummer"] = df["Proefpersoonnummer"].astype(str)
old_df["Proefpersoonnummer"] = old_df["Proefpersoonnummer"].astype(str)
#Keep only the old columns that actually exist in the old dataset
old_cols_existing = [c for c in old_cols if c in old_df.columns]
df_sel_old = old_df[old_cols_existing].copy()
#Merge old and new datasets to compare and correct columns
merged = df.merge(df_sel_old, on="Proefpersoonnummer", how="left", suffixes=(".x", ".y"))
#Replace incorrect columns with corrected values from old dataset
for b in columns_to_be_switched:
    for i in range(1, 7):
        x = f"{b}.{i}.x"  
        y = f"{b}.{i}.y"   
        if x in merged.columns and y in merged.columns:
            merged[f"{b}.{i}"] = merged[y].combine_first(merged[x])
#Remove temporary merge columns (.x and .y suffixes)
drop_cols = [c for c in merged.columns if c.endswith(".x") or c.endswith(".y")]
df_final = merged.drop(columns=drop_cols)
#%%----------------------------------------------------------------------------------------------------------------------------------------------------------
#Making sure the warning does not come up
pd.set_option('future.no_silent_downcasting', True)
#Change missing values to NaN (leaving out proefpersoonnummer)
cols = [c for c in df_final.columns if c != "Proefpersoonnummer"]
#Empty strings become NaN
df_final[cols] = df_final[cols].replace('', np.nan)
#99 and 999 become NaN
df_final[cols] = df_final[cols].replace(
    ["99", "999", 99, 999],
    np.nan)
#%%--------------------------------------------------------------------------------------------------------
#There are some binary values for which 1 = yes, 2 = no. I want these to change to 1 = yes, 0 = no
binary_columns = [
    "leefsituatie_steun.1","leefsituatie_steun.2","leefsituatie_steun.3",
    "leefsituatie_steun.4","leefsituatie_steun.5","leefsituatie_steun.6",
    "MANSA_PH_7.1","MANSA_PH_9.1","MANSA_PH_10.1","MANSA_PH_11.1",
    "Leeftijd1ePsyKl_a.1"]
for col in binary_columns:
    if col in df_final.columns:
        df_final[col] = (
            df_final[col]
            .replace({1: 1, 2: 0, "1": 1, "2": 0})
            .astype("Int64"))
#%%----------------------------------------------------------------------------------------------------------
#I want to recode the variable "leefsituatie" in 4 more logical categories; 1 = zelfstandig, 2 = samenwonend met derden, 3 = onzelfstandig, 4 = dakloos / anders
leefsituatie_recoded = {1:1 #zelfstandig alleen wonend
                        ,2:2, 3:2, 4:2, 5:2, 6:2, #samenwonend met derden
                        7:3, 8:3, 10:3, 11:3, #onzelfstandig
                        9:4, 12:4 #dakloos / anders
                        }
leefsituatie_cols = [
    f"leefsituatie.{i}" for i in range(1, 7)
    if f"leefsituatie.{i}" in df_final.columns]
for col in leefsituatie_cols:
    df_final[col] = (
        pd.to_numeric(df_final[col], errors="coerce")  
        .map(leefsituatie_recoded)
        .astype("Int64"))
#%%-----------------------------------------------------------------------------------------------------------------------------------------------------
#Making a new "opleiding_nieuw" variabele with the answers from opleiding.1 for better alignment and making it numeric for the one-hot-encoding
df_final = df_final.copy()
df_final["opleiding_nieuw"] = (
    pd.to_numeric(df_final["opleiding.1"], errors="coerce")
    .astype("Int64"))
#%%------------------------------------------------------------------------------------------------------------------------------------------------------
#Fix the variable opleiding_nieuw = 0: only 1 entry (prevent warning in nested CV)
df_final["opleiding_nieuw"] = pd.to_numeric(
    df_final["opleiding_nieuw"], errors="coerce"
)
df_final.loc[
    df_final["opleiding_nieuw"] == 0, "opleiding_nieuw"
] = np.nan
#%%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Recoding burgerlijke staat into a binary item with 0 = no partner, 1 = a partner
for i in range(1, 7):
    col = f"burgerlijkestaat.{i}"
    if col in df_final.columns:
        df_final[col] = (
            pd.to_numeric(df_final[col], errors="coerce")  # 0–4 + NA
            .apply(lambda x: 1 if x in [2,4] else (0 if pd.notna(x) else pd.NA))
            .astype("Int64"))
#%%--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Recoding dagbesteding in binary groups (0 = no, 1 = yes, NA = no info)
#Original codebook:
#dagbesteding_1_ : 0 = nee, 13 = geen dagbesteding
#dagbesteding_2_ : 0 = nee,  1 = DAC
#dagbesteding_3_ : 0 = nee,  2 = arbeidsmatige dagbesteding / werkleer
#dagbesteding_4_ : 0 = nee,  3 = opleiding
#dagbesteding_5_ : 0 = nee,  4 = vrijwilligerswerk
#dagbesteding_6_ : 0 = nee,  5 = betaald, beschermd
#dagbesteding_7_ : 0 = nee,  6 = werkervaringsplaats / proefplaatsing
#dagbesteding_8_ : 0 = nee,  7 = betaald met ondersteuning
#dagbesteding_9_ : 0 = nee,  8 = regulier betaald werk
#dagbesteding_10_: 0 = nee,  9 = huisvrouw/-man
#dagbesteding_11_: 0 = nee, 10 = huishouden
#dagbesteding_12_: 0 = nee, 11 = dagbehandeling
#dagbesteding_13_: 0 = nee, 12 = anders
#dagbesteding_14_: 0 = nee, 999 = onbekend  → treat as missings
dag_groups = {
    "dagbest_betaald":    ["dagbesteding_6_", "dagbesteding_7_", "dagbesteding_8_", "dagbesteding_9_"],
    "dagbest_opleiding":  ["dagbesteding_4_"],
    "dagbest_dagact":     ["dagbesteding_2_", "dagbesteding_3_", "dagbesteding_12_"],
    "dagbest_vrijwillig": ["dagbesteding_5_"],
    "dagbest_huishouden": ["dagbesteding_10_", "dagbesteding_11_"],
    "dagbest_overig":     ["dagbesteding_13_"],
    "dagbest_geen":       ["dagbesteding_1_"],
    #no dummy for 14 --> "onbekend" is missing
}
for i in range(1, 7):
    for newname, bases in dag_groups.items():
        cols = [f"{b}.{i}" for b in bases if f"{b}.{i}" in df_final.columns]
        if not cols:
            continue
        #To numeric. 0 = nee, >0 = yes, NaN = non info
        block = df_final[cols].apply(pd.to_numeric, errors="coerce")
        any_pos = block.fillna(0).ne(0).any(axis=1)
        all_na = block.isna().all(axis=1)
        out = any_pos.astype("Int64")
        out[all_na] = pd.NA
        df_final[f"{newname}.{i}"] = out
#Remove original dagbesteding columns
dag_cols_original = [c for c in df_final.columns if c.startswith("dagbesteding_")]
df_final = df_final.drop(columns=dag_cols_original, errors="ignore")
#%%-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Recoding inkomsten into 0 = no, 1 = yes instead of the numbers. So making proper binary variables
for i in range(1, 7):  
    inkomsten_cols = [c for c in df_final.columns
                      if c.startswith("inkomsten_") and c.endswith(f".{i}")]
    for col in inkomsten_cols:
        s = pd.to_numeric(df_final[col], errors="coerce")  
        s = s.where(s.isna() | (s == 0), 1)
        df_final[col] = s.astype("Int64")
#%%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Adding columns for date finished, intervals between timepoints and months since first timepoint 
#Making the columns for date finished for every timepoint
new_cols = {}
for i in range(1, 7):
    year_col  = f"jaarafgerondgegevensafname.{i}"
    month_col = f"maandafgerondgegevensafname.{i}"
    out_col   = f"date_finished.{i}"
    y = pd.to_numeric(df_final[year_col],  errors="coerce").astype(float) 
    m = pd.to_numeric(df_final[month_col], errors="coerce").astype(float) 
    new_cols[out_col] = pd.to_datetime({"year": y, "month": m, "day": 1}, errors="coerce")
df_final = pd.concat([df_final, pd.DataFrame(new_cols, index=df_final.index)], axis=1).copy()
for i in range(1, 7):
    col = f"date_finished.{i}"
    df_final[col] = df_final[col].dt.date
#Columns for the intervals between timestamps
new_cols_months = {}
for i in range(1, 6):
    first_range = f"date_finished.{i}"
    second_range = f"date_finished.{i + 1}"
    outcome_col = f"month_diff_{i}_and_{i + 1}"
    a = pd.to_datetime(df_final[first_range], errors="coerce")
    b = pd.to_datetime(df_final[second_range], errors="coerce")
    diff_months = (b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)
    diff_months[a.isna() | b.isna()] = np.nan  # handle missing values
    new_cols_months[outcome_col] = diff_months.astype(float)
df_final = pd.concat([df_final, pd.DataFrame(new_cols_months, index=df_final.index)], axis=1).copy()
#Columns for months since first timepoint 
df_final["month_diff_1_and_3"] = df_final["month_diff_1_and_2"] + df_final["month_diff_2_and_3"]
df_final["month_diff_1_and_4"] = df_final["month_diff_1_and_2"] + df_final["month_diff_2_and_3"] + df_final["month_diff_3_and_4"]
df_final["month_diff_1_and_5"] = df_final["month_diff_1_and_2"] + df_final["month_diff_2_and_3"] + df_final["month_diff_3_and_4"] + df_final["month_diff_4_and_5"]
df_final["month_diff_1_and_6"] = df_final["month_diff_1_and_2"] + df_final["month_diff_2_and_3"] + df_final["month_diff_3_and_4"] + df_final["month_diff_4_and_5"] + df_final["month_diff_5_and_6"]
#Remove the unnecesarry columns 
columns_to_remove = ["Date_finished_1", "Date_finished_2", "Date_finished_3", "Date_finished_4", "Date_finished_5", "Date_finished_6", "Diff_finished_1_2_Months", "Diff_finished_2_3_Months", "Diff_finished_3_4_Months", "Diff_finished_4_5_Months", "Diff_finished_5_6_Months"]
df_final = df_final.drop(columns=columns_to_remove, errors="ignore")
#%%---------------------------------------------------------------------------------------------------------------------------------------------
#There are 2 problems with the columns "Leeftijd1eGGZ" & "Leeftijd1ePsykL" that need to be fixed.
#1.there are birthyears in the columns. They do not impact the data, so can be removed.
#2.there are different answers in the two columns, this is not possible --> The most noted answer (modus) is chosen.
#If the modus is equal, I take the average of all the answers who are noted the most and rounded it to above.
cols_ggz = [
    "Leeftijd1eGGZ_b.1", "Leeftijd1eGGZ_b.2", "Leeftijd1eGGZ_b.3",
    "Leeftijd1eGGZ_b.4", "Leeftijd1eGGZ_b.5", "Leeftijd1eGGZ_b.6"]
cols_psykl = [
    "Leeftijd1ePsyKl_b.1", "Leeftijd1ePsyKl_b.2", "Leeftijd1ePsyKl_b.3",
    "Leeftijd1ePsyKl_b.4", "Leeftijd1ePsyKl_b.5", "Leeftijd1ePsyKl_b.6"]
#To numeric, unknown value --> NaN
df_final[cols_ggz + cols_psykl] = df_final[cols_ggz + cols_psykl].apply(
    pd.to_numeric, errors="coerce")
#Remove birthyears
df_final[cols_ggz]   = df_final[cols_ggz].mask(df_final[cols_ggz] > 1900)
df_final[cols_psykl] = df_final[cols_psykl].mask(df_final[cols_psykl] > 1900)
def modusmean(values: pd.Series):
    x = values.dropna()
    if x.empty:
        return np.nan
    freq = x.value_counts()
    max_freq = freq.max()
    modes = freq.index[freq == max_freq].astype(float)
    return math.ceil(np.mean(modes))
df_final["modusmeanGGZ"] = (
    df_final[cols_ggz].apply(modusmean, axis=1).astype("Int64"))
df_final["modusmeanPsyKl"] = (
    df_final[cols_psykl].apply(modusmean, axis=1).astype("Int64"))
#%%------------------------------------------------------------------------------------------------------------------------------------
#Remove age outliers (0,12,16) and make the dataset between 17 and 75 years old
df_final["Age"] = pd.to_numeric(df_final["Age"], errors="coerce")
df_final = df_final[df_final["Age"].between(17, 75, inclusive="both")]
#%%------------------------------------------------------------------------------------------------------------------------------------
#Total score of Brief Inspire-O over the six timepoints
df_final = df_final.assign(Inspire_totaal_1 = (df_final["IHS_01.1"] + df_final["IHS_02.1"] +
                        df_final["IHS_03.1"] + df_final["IHS_04.1"] +
                        df_final["IHS_05.1"]) * 5)
df_final = df_final.assign(Inspire_totaal_2 = (df_final["IHS_01.2"] + df_final["IHS_02.2"] +
                        df_final["IHS_03.2"] + df_final["IHS_04.2"] +
                        df_final["IHS_05.2"]) * 5)
df_final = df_final.assign(Inspire_totaal_3 = (df_final["IHS_01.3"] + df_final["IHS_02.3"] +
                        df_final["IHS_03.3"] + df_final["IHS_04.3"] +
                        df_final["IHS_05.3"]) * 5)
df_final = df_final.assign(Inspire_totaal_4 = (df_final["IHS_01.4"] + df_final["IHS_02.4"] +
                        df_final["IHS_03.4"] + df_final["IHS_04.4"] +
                        df_final["IHS_05.4"]) * 5)
df_final = df_final.assign(Inspire_totaal_5 = (df_final["IHS_01.5"] + df_final["IHS_02.5"] +
                        df_final["IHS_03.5"] + df_final["IHS_04.5"] +
                        df_final["IHS_05.5"]) * 5)
df_final = df_final.assign(Inspire_totaal_6 = (df_final["IHS_01.6"] + df_final["IHS_02.6"] +
                        df_final["IHS_03.6"] + df_final["IHS_04.6"] +
                        df_final["IHS_05.6"]) * 5)
#Binary score 1 = score of 55 of higher and 0 if it is lower than 55:
for i in range(1, 6 + 1):
    col = f"Inspire_totaal_{i}"
    out = f"Inspire_binair.{i}"
    s = df_final[col]
    df_final[out] = s.gt(54).astype("Int64").where(~s.isna(), pd.NA)
#%%-----------------------------------------------------------------------------------------------------------------------------
#Functional Remission Scale
#Make this a total + binary outcome with binary outcome with 1 if all three questions have answer "0", 0 in all other cases en NaN if 1 of more of the 3 questions are missing
for i in range(1,7):
    cols_FR = [f"FR_1.{i}", f"FR_2.{i}", f"FR_3.{i}"]
    X = df_final[cols_FR].apply(pd.to_numeric, errors = "coerce").replace(99, np.nan) 
    df_final[f"FR_totaal.{i}"] = np.where(X.isna().any(axis = 1), np.nan, X.sum(axis = 1))
    s = df_final[f"FR_totaal.{i}"]
    df_final[f"FR_binair.{i}"] = s.eq(0).astype("Int64").where(~s.isna(), pd.NA)
#%%-----------------------------------------------------------------------------------------------------
#MANSA
#Total score is the mean of the 12 questions of MANSA. The 4 binary questions are not taken into account here.
likert_items = [1,2,3,4,5,6,8,12,13,14,15,16]
for i in range(1,7):
    cols_mansa = [f"MANSA_PH_{j}.{i}" for j in likert_items]
    X = df_final[cols_mansa].apply(pd.to_numeric, errors = "coerce") 
    mansa_missing = X.isna().sum(axis = 1)
    mansa_mean = X.mean(axis = 1, skipna = True)
    mansa_total = (mansa_mean * 12).copy()
    mansa_total = mansa_total.where(mansa_missing <= 2, np.nan)
    df_final[f"mansa_totaal.{i}"] = mansa_total
#%%--------------------------------------------------------------------------------------------------------
#HONOS
#The total HoNOS score is calculated as the mean of the 12 items (item 13-15 are not taken into account)
#multiplied by 12. If a participant has more than two missing items, the total score is set to NaN.
honos_items = ["01", "02", "03", "04", "05", "06", "07", "08b", "09", "10", "11", "12"]
for i in range(1,7):
    cols_honos = [f"HONOS_add_{j}.{i}" for j in honos_items]
    X1 = df_final[cols_honos].apply(pd.to_numeric, errors = "coerce") 
    honos_missing = X1.isna().sum(axis=1)
    honos_mean = X1.mean(axis = 1, skipna = True)
    honos_total = (honos_mean * 12).copy()
    honos_total = honos_total.where(honos_missing <= 2, np.nan)
    df_final[f"honos_totaal.{i}"] = honos_total
#%%---------------------------------------------------------------------------------------------------------
#Export the preprocessed data
df_final.to_csv(processed_data, index=False)
#Making the CSV file into an external excel file for better alignment and understanding
excel_file = p("processed_data.xlsx")
df_final.to_excel(excel_file, index=False, header = True, sheet_name = "Processed_Data") 
#Making a different file with all the colums for a better overview
cols = pd.Series(df_final.columns)
cols.to_csv(p("column_names.txt"), index=False)
#Make the data_decrypted.csv into an excel file for comparison before preprocessing
excel_file1 = p("decrypted_processed_data.xlsx")
df.to_excel(excel_file1, index=False, header = True, sheet_name = "Decrypted_processed_Data") 
#%%---------------------------------------------------------------------------------------------------------
#Visualizing the data
img_dir = p("data_images")
os.makedirs(img_dir, exist_ok=True)
#MANSA
#Distribution of MANSA total score at timepoint 1 + a dense plot overlay
data1 = df_final["mansa_totaal.1"].dropna()
plt.figure(figsize=(8, 5)) 
sns.histplot(data1, bins=20, color="#c858b9", edgecolor="black",
             alpha=0.6, stat="density")
sns.kdeplot(data1, color="black", linewidth=2) 
plt.title("Distribution of MANSA total score at timepoint 1", fontsize=14, fontweight="bold") 
plt.xlabel("MANSA total score", fontsize=12) 
plt.ylabel("Density", fontsize=12) 
plt.grid(axis="y", color="gray", alpha=0.4) 
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "mansa_distribution_density_1.png"), dpi=300, bbox_inches="tight") 
#Distribution of MANSA total score at timepoint 2 + a dense plot overlay
data2 = df_final["mansa_totaal.2"].dropna()
plt.figure(figsize=(8, 5)) 
sns.histplot(data2, bins=20, color="#e298c9", edgecolor="black", 
             alpha=0.6, stat="density")
sns.kdeplot(data2, color="black", linewidth=2) 
plt.title("Distribution of MANSA total score at timepoint 2", fontsize=14, fontweight="bold") 
plt.xlabel("MANSA total score", fontsize=12) 
plt.ylabel("Density", fontsize=12) 
plt.grid(axis="y", color="gray", alpha=0.4) 
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "mansa_distribution_density_2.png"), dpi=300, bbox_inches="tight") 
#Boxplot MANSA visual
data = pd.melt(df_final, 
               value_vars=[f"mansa_totaal.{i}" for i in range(1,7)],
               var_name="Timepoint", value_name="MANSA_total")
plt.figure(figsize=(8,5)) 
sns.boxplot(data=data, x="Timepoint", y="MANSA_total", color="#D06CA8")
plt.title("Distribution of MANSA total scores across timepoints", fontsize=14, fontweight="bold") 
plt.xlabel("Timepoint"); plt.ylabel("MANSA total score") 
plt.ylim(10,85)
plt.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "mansa_boxplot_per_timepoint.png"), dpi=300, bbox_inches="tight") 
#A scatterplot with the 2 timepoints. 
subset = df_final[["mansa_totaal.1", "mansa_totaal.2", "month_diff_1_and_2"]].dropna()
x = subset["mansa_totaal.1"]
y = subset["mansa_totaal.2"]
r, p_val = pearsonr(x, y)
plt.figure(figsize=(6,6))
sc = plt.scatter(x, y, c=subset["month_diff_1_and_2"], cmap="viridis",
                 alpha=0.7, edgecolor="black", linewidth=0.3)
lims = [12, 84]
plt.plot(lims, lims, linestyle="--", color="black", linewidth=1)  
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("MANSA total score (T1)")
plt.ylabel("MANSA total score (T2)")
plt.title(f"Correlation between MANSA T1 and T2 (r = {r:.2f}, p = {p_val:.3f})")
cbar = plt.colorbar(sc)
cbar.set_label("Months between T1 and T2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "mansa_T1_T2_scatter_corr.png"), dpi=300, bbox_inches="tight")
#Showing the distribution of MANSA scores in months between 1 and 2 measurement
mask = df_final["mansa_totaal.1"].notna() & df_final["mansa_totaal.2"].notna()
m = df_final.loc[mask, "month_diff_1_and_2"].dropna()
lo = int(np.floor(m.min()))
hi = int(np.ceil(m.max()))
bins = np.arange(lo, hi + 1, 1)
plt.figure(figsize=(10, 6))
plt.hist(m, bins=bins, color="#69b3a2", edgecolor="black", alpha=0.8)
plt.axvline(8,  color="red", linestyle="--", linewidth=1.8, label="9–15 months window")
plt.axvline(15, color="red", linestyle="--", linewidth=1.8)
plt.legend(frameon=False)
plt.axvspan(8, 15, color="red", alpha=0.08)
plt.title("Distribution of time between timepoint 1 and 2 (MANSA pairs only)",
          fontsize=14, fontweight="bold")
plt.xlabel("Months")
plt.ylabel("Count")
plt.xticks(np.arange(lo, hi + 1, 1), rotation=45, ha="right")
plt.grid(axis="y", color="gray", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "months_T1_T2_hist_mansa_pairs.png"),
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------------------------------------
#Working with the first smaller subset (between 9 and 15 months)
df_mansa_9_15 = df_final[
    (df_final["mansa_totaal.1"].notna()) &
    (df_final["mansa_totaal.2"].notna()) &
    (df_final["month_diff_1_and_2"].between(9, 15, inclusive="both")) ].copy()
#Getting information about this subset:
age_mean = df_mansa_9_15["Age"].mean()
age_sd   = df_mansa_9_15["Age"].std()
age_min  = df_mansa_9_15["Age"].min()
age_max  = df_mansa_9_15["Age"].max()
print(f"Mean age = {age_mean:.2f}, SD = {age_sd:.2f}, Min = {age_min:.0f}, Max = {age_max:.0f}")
print("9–15 maanden subset:", df_mansa_9_15.shape)
#Making visuals about this smaller subset
img_dir_1 = p("data_images_MANSA_9_15_months")
os.makedirs(img_dir_1, exist_ok=True)
#Distribution timepoint 1 with smaller subset
data_first_visual = df_mansa_9_15["mansa_totaal.1"].dropna()
plt.figure(figsize=(8, 5)) 
sns.histplot(data_first_visual, bins=20, color = "#4b357b", edgecolor="black" , alpha=0.6, stat="density")
sns.kdeplot(data_first_visual, color="black", linewidth=2) 
plt.title("Distribution of MANSA total score at timepoint 1 (subset)", fontsize=14, fontweight="bold") 
plt.xlabel("MANSA total score with subset", fontsize=12) 
plt.ylabel("Density", fontsize=12) 
plt.grid(axis="y", color="gray", alpha=0.4) 
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "mansa_distribution_density_1_subset.png"), dpi=300, bbox_inches="tight") 
#Distribution of timepoint 2 with subset
data_second_visual = df_mansa_9_15["mansa_totaal.2"].dropna()
plt.figure(figsize=(8, 5)) 
sns.histplot(data_second_visual, bins=20, color = "#4b357b", edgecolor="black" , alpha=0.6, stat="density")
sns.kdeplot(data_second_visual, color="black", linewidth=2) 
plt.title("Distribution of MANSA total score at timepoint 2 (subset)", fontsize=14, fontweight="bold") 
plt.xlabel("MANSA total score with subset", fontsize=12) 
plt.ylabel("Density", fontsize=12) 
plt.grid(axis="y", color="gray", alpha=0.4) 
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "mansa_distribution_density_2_subset.png"), dpi=300, bbox_inches="tight") 
#The correlation between the 2
subset = df_mansa_9_15[["mansa_totaal.1", "mansa_totaal.2", "month_diff_1_and_2"]].dropna()
x = subset["mansa_totaal.1"]
y = subset["mansa_totaal.2"]
r, p_val = pearsonr(x, y)
plt.figure(figsize=(6,6))
sc = plt.scatter(x, y, c=subset["month_diff_1_and_2"], cmap="viridis",
                 alpha=0.7, edgecolor="black", linewidth=0.3)
lims = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
plt.plot(lims, lims, linestyle="--", color="black", linewidth=1)  
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("MANSA total score (T1)")
plt.ylabel("MANSA total score (T2)")
plt.title(f"Correlation between MANSA T1 and T2 subset (r = {r:.2f}, p = {p_val:.3f})")
cbar = plt.colorbar(sc)
cbar.set_label("Months between T1 and T2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "mansa_T1_T2_scatter_corr_subset.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%-------------------------------------------------------------------------------------------------------------------------
#Making the model with just MANSA_totaal.1 as the predictor of MANSA_totaal.2
y_true = df_mansa_9_15["mansa_totaal.2"]
y_pred_identity = df_mansa_9_15["mansa_totaal.1"]
r2_id = r2_score(y_true, y_pred_identity)
rmse_id = mean_squared_error(y_true, y_pred_identity) ** 0.5
mae_id = mean_absolute_error(y_true, y_pred_identity)
print(f"Identity baseline: R²={r2_id:.3f}, RMSE={rmse_id:.3f}, MAE={mae_id:.3f}")
#%%--------------------------------------------------------------------------------------------------------------------------
#Training the first model (=linear regression between mansa_totaal_1 & mansa_totaal_2)
subset = df_mansa_9_15.dropna(subset=["mansa_totaal.1", "mansa_totaal.2"]).copy()
subset["mansa_totaal.1"] = pd.to_numeric(subset["mansa_totaal.1"], errors="coerce")
subset["mansa_totaal.2"] = pd.to_numeric(subset["mansa_totaal.2"], errors="coerce")
subset = subset.dropna(subset=["mansa_totaal.1", "mansa_totaal.2"])
X = sm.add_constant(subset["mansa_totaal.1"].astype(float))
y = subset["mansa_totaal.2"].astype(float)
model_sm = sm.OLS(y, X).fit()
print(model_sm.summary())
#Visualizing the regression plot
subset = df_mansa_9_15.dropna(subset=["mansa_totaal.1", "mansa_totaal.2"]).copy()
plt.figure(figsize=(8, 6))
sns.regplot(
    data=subset,
    x="mansa_totaal.1",
    y="mansa_totaal.2",
    scatter_kws={"color": "black", "alpha": 0.7},
    line_kws={"color": "red", "linewidth": 2})
plt.title("Scatter Plot of MANSA_totaal.1 vs MANSA_totaal.2", fontsize=14, fontweight="bold")
plt.xlabel("MANSA_totaal.1")
plt.ylabel("MANSA_totaal.2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "mansa_scatter_regression.png"), dpi=300, bbox_inches="tight")
#Trying the Seaborn thing Mosi recommended
#On leefsituatie categorical feature
leef_map = {1: "Zelfstandig",2: "Samenwonend met derden",3: "Onzelfstandig",4: "Dakloos/anders"}
data_seaborn = df_mansa_9_15[["mansa_totaal.1","mansa_totaal.2","leefsituatie.1","month_diff_1_and_2","modusmeanGGZ","modusmeanPsyKl"]].copy()
valid = list(leef_map.keys())
data_seaborn = data_seaborn[data_seaborn["leefsituatie.1"].isin(valid)]
data_seaborn["leefsituatie.1"] = data_seaborn["leefsituatie.1"].map(leef_map)
order = [leef_map[k] for k in valid]
ren = {"month_diff_1_and_2": "Maanden T1→T2","modusmeanGGZ": "Leeftijd 1e GGZ (modus)","modusmeanPsyKl": "Leeftijd 1e PsyKl (modus)",}
data_seaborn = data_seaborn.rename(columns=ren)
g = sns.pairplot(data_seaborn, hue="leefsituatie.1", hue_order=order, diag_kind="hist", markers=["o","s","D","v"], height=1.6, y_vars=["mansa_totaal.1","mansa_totaal.2"],x_vars=["Maanden T1→T2","Leeftijd 1e GGZ (modus)","Leeftijd 1e PsyKl (modus)"], palette="Set2",plot_kws={"alpha": 0.8, "edgecolor": "white", "linewidths": 0.3, "s": 28})
g.fig.subplots_adjust(right=0.82, top=0.95, bottom=0.08)
if g._legend is not None:
    g._legend.remove()
g.add_legend(title="Leefsituatie",bbox_to_anchor=(1.02, 0.5),loc="center left",borderaxespad=0)
g.fig.savefig(os.path.join(img_dir_1, "seaborn_on_leefsituatie.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------------------------------------------
#OLS: normality check of residuals (Shapiro, skewness, kurtosis + diagnostics)
#Residuals from the OLS-model
residuals = pd.Series(model_sm.resid).dropna()
#Shapiro–Wilk test
w_stat, p_val = shapiro(residuals)
skew_val = skew(residuals)
kurt_val = kurtosis(residuals)
print("\nNormality check of OLS residuals")
print(f"Shapiro–Wilk W = {w_stat:.3f}, p = {p_val:.4f}")
print(f"Skewness       = {skew_val:.3f}")
print(f"Kurtosis       = {kurt_val:.3f}")
if p_val > 0.05:
    print("Residuals are approximately normal (Shapiro p > .05).")
else:
    print("Residuals deviate from normality (Shapiro p ≤ .05).")
#Q–Q plot
plt.figure(figsize=(6, 4))
probplot(residuals, dist="norm", plot=plt)
plt.title("Q–Q plot of OLS residuals")
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "qqplot_ols_residuals.png"),
            dpi=300, bbox_inches="tight")
#Histogram of the residuals
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30, edgecolor="k", alpha=0.7)
plt.title("Histogram of OLS residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "hist_ols_residuals.png"),
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Handling the results with adding a robust SE model
subset = df_mansa_9_15[["mansa_totaal.1", "mansa_totaal.2"]].dropna().copy()
model_sm_robust = smf.ols(
    "Q('mansa_totaal.2') ~ Q('mansa_totaal.1')",
    data=subset).fit(cov_type="HC3")
print(model_sm_robust.summary())
#%%-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Checking for outliers just for intepreting (OLS outliers and influence diagnostics)
#Extreme values are investigated but not excluded. Scaling reduces their impact for distance-based and linear models.
influence = OLSInfluence(model_sm)
#Extract key influence metrics
df_influence = df_mansa_9_15.copy()
df_influence["student_resid"] = influence.resid_studentized_external
df_influence["leverage"] = influence.hat_matrix_diag
df_influence["cooks_d"] = influence.cooks_distance[0]
#Set thresholds
n = len(df_influence)
k = int(model_sm.df_model)
student_resid_thresh = 3
leverage_thresh = 2 * (k + 1) / n
cooks_d_thresh = 4 / n
#Flag potential outliers
df_influence["is_outlier"] = (
    (np.abs(df_influence["student_resid"]) > student_resid_thresh) |
    (df_influence["leverage"] > leverage_thresh) |
    (df_influence["cooks_d"] > cooks_d_thresh))
#Print the summary
print("\nTop 10 by Cook's distance:")
display(df_influence.sort_values("cooks_d", ascending=False)
        [["mansa_totaal.1", "mansa_totaal.2", "student_resid", "leverage", "cooks_d"]]
        .head(10))
#Scatterplot: leverage vs studentized residuals
plt.figure(figsize=(6,4))
plt.scatter(df_influence["leverage"], df_influence["student_resid"], alpha=0.7)
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Leverage")
plt.ylabel("Studentized Residuals")
plt.title("Outlier & Leverage Plot (OLS baseline)")
#Highlight the potential outliers in red
plt.scatter(df_influence.loc[df_influence["is_outlier"], "leverage"],
            df_influence.loc[df_influence["is_outlier"], "student_resid"],
            color="red", label="Potential outliers")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "OLS_outliers.png"), dpi=300)
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------------------------------------------------
#Fit model with all data to get the slope change when removing the outliers
model_full = model_sm  
#Fit model without the flagged outliers
df_no_out = df_influence.loc[~df_influence["is_outlier"]]
X_clean = sm.add_constant(df_no_out["mansa_totaal.1"])
y_clean = df_no_out["mansa_totaal.2"]
model_clean = sm.OLS(y_clean, X_clean).fit()
#Compute percent change in the slope
beta_full = model_full.params.iloc[1]
beta_clean = model_clean.params.iloc[1]
pct_change = abs(beta_clean - beta_full) / abs(beta_full) * 100
print("Percent slope change:", pct_change)
#%%----------------------------------------------------------------------------------------------------------------------------
#Doing the Shapiro test for MANSA(9-15 month subset)
ycol = "mansa_totaal.2"
mask = df_mansa_9_15[ycol].notna()
y = df_mansa_9_15.loc[mask, ycol].astype(float)
#Skewness
skew_val = skew(y, nan_policy="omit")
print(f"Skewness: {skew_val:.3f}")
#Shapiro-Wilk normality test
w_stat, p_val = shapiro(y)
print(f"Shapiro-Wilk: W = {w_stat:.3f}, p = {p_val:.3e}")
if p_val < 0.05:
    print("Target deviates from normality (p < 0.05).")
else:
    print("No strong deviations from normality.")
#%%-----------------------------------------------------------------------------------------------------------------------------
#Start with the building of the variables for the models:
#Income variables (T1 only)
inkomsten_vars = [
    c for c in df_mansa_9_15.columns
    if c.startswith("inkomsten_") and c.endswith(".1")]
context_vars = [
    "Age","geslacht_GegevensAfname","geboortemandsocio","Leeftijd1eGGZ_a.1","Leeftijd1ePsyKl_a.1","modusmeanGGZ","modusmeanPsyKl","burgerlijkestaat.1",
    "leefsituatie.1","leefsituatie_steun.1","levenspartner.1","opleiding_nieuw","month_diff_1_and_2","dagbest_betaald.1","dagbest_opleiding.1","dagbest_dagact.1",
    "dagbest_vrijwillig.1","dagbest_huishouden.1","dagbest_overig.1","dagbest_geen.1","Betrouwbaarheid_1_.1","vrijwilligerswerk_2_.1","HerstelHV_4_10_.1"] + inkomsten_vars
#Questionnaire totals
questionnaire_totals = [
    "Inspire_totaal_1",
    "FR_totaal.1",
    "mansa_totaal.1",
    "honos_totaal.1",
    "Inspire_binair.1",
    "FR_binair.1",]
#Items: Inspire
inspire_items = [f"IHS_0{i}.1" for i in range(1, 6)]
#Items: FR
fr_items = [f"FR_{i}.1" for i in range(1, 4)]
#Items: MANSA (likert + binary)
mansa_likert_items = [1,2,3,4,5,6,8,12,13,14,15,16]
mansa_binary_items = [7,9,10,11]
mansa_items = [f"MANSA_PH_{i}.1" for i in mansa_likert_items + mansa_binary_items]
#Items: HoNOS
honos_items_1 = (
    [f"HONOS_add_{c}.1" for c in honos_items] +
    ["HONOS_add_13.1", "HONOS_add_14.1", "HONOS_add_15.1"])
#Alle predictors
predictors_all_questionnaires = (
    context_vars + inspire_items + fr_items + mansa_items + honos_items_1)
all_predictors = (
    context_vars + questionnaire_totals + inspire_items + fr_items + mansa_items + honos_items_1)
#%%------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Checking what percentage of each variable is missing for in the exploratory baseline models
OUTCOME = "mansa_totaal.2"
cols_for_missing = [c for c in all_predictors if c in df_mansa_9_15.columns and c != OUTCOME]
missing_df = (df_mansa_9_15[cols_for_missing].isna().mean() * 100).reset_index()
missing_df.columns = ["variable", "missing_percent"]
missing_df = missing_df.sort_values("missing_percent", ascending=False)
missing_df.to_csv(p("missing_values_all_features.csv"), index=False)
missing_df.to_excel(
    p("missing_values_all_features.xlsx"),
    index=False, header=True, sheet_name="Missing values")
overall_missing_pct = df_mansa_9_15[cols_for_missing].isna().mean().mean() * 100
print(f"Overall missingness across predictors: {overall_missing_pct:.1f}%")
#%%-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Visualizing missings top 30 and top 10 (predictors only)
if len(cols_for_missing) == 0:
    print("No columns present right now to visualize.")
else:
    #Matrix + heatmap 
    plt.figure(figsize=(14, 6))
    msno.matrix(df_mansa_9_15[cols_for_missing], sparkline=False, labels=True)
    plt.title("Missing values in analysis subset (predictors only)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir_1, "missing_matrix_df_mansa_9_15.png"), dpi=300)
    plt.close()
    if len(cols_for_missing) > 1:
        plt.figure(figsize=(10, 6))
        msno.heatmap(df_mansa_9_15[cols_for_missing])
        plt.title("Correlation of missingness (predictors only)")
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir_1, "missing_heatmap_df_mansa_9_15.png"), dpi=300)
        plt.close()
    #Barplots: top 30 en top 10
    missing_pct = (df_mansa_9_15[cols_for_missing]
                   .isna().mean() * 100).sort_values(ascending=False)
    for topN in [30, 10]:
        top = missing_pct.head(topN)
        plt.figure(figsize=(8, 10))
        plt.barh(top.index[::-1], top.values[::-1])
        plt.axvline(x=50, color="red", linestyle="--", linewidth=1.5)
        plt.xlabel("Missing (%)")
        plt.ylabel("Variable")
        plt.title(f"Top {topN} missing variables (predictors only)")
        plt.tight_layout()
        fname = f"missing_bar_top{topN}_df_mansa_9_15.png"
        plt.savefig(os.path.join(img_dir_1, fname), dpi=300)
        plt.show()
#%%---------------------------------------------------------------------------------------
#Dropping the columns with >50% missings, baseline only! (= "HerstelHV_4_10_.1" & "vrijwilligerswerk_2_.1")
df_base = df_mansa_9_15.copy()
cols_to_drop = missing_df.loc[missing_df["missing_percent"] > 50, "variable"]
df_base = df_base.drop(columns=list(cols_to_drop), errors="ignore")
#Update the predictor list
existing_predictors = [c for c in all_predictors if c in df_base.columns and c != OUTCOME]
all_predictors = existing_predictors
context_vars = [c for c in context_vars if c in df_base.columns]
print(context_vars)
print(all_predictors)
#%%-----------------------------------------------------------------------------------------------
#Making the smaller correct subset
df_mansa_9_15.to_csv(p("subset_9_15.csv"), index=False)
df_mansa_9_15.to_excel(p("subset_9_15.xlsx"),
                       index=False, header=True,
                       sheet_name="Subset 9-15 months")
#%%------------------------------------------------------------------------------------------------
#Map for EXPLORATORY BASELINE ONLY - NO FEATURE SELECTION OR EXCLUSION
models_dir = p("models")
os.makedirs(models_dir, exist_ok=True)
#%%---------------------------------------------------------------------------------------------------
#Goal variable + indices
ycol = "mansa_totaal.2"
valid_idx = df_base.index[df_base[ycol].notna()]
y_all = df_base.loc[valid_idx, ycol].astype(float)
#%%---------------------------------------------------------------------------------------------------------------
#One train/test split for all the linear regression feature chosen models
train_idx, test_idx = train_test_split(valid_idx, test_size=0.2, random_state=42)
print(f"Split sizes → train={len(train_idx)}, test={len(test_idx)}")
#%%----------------------------------------------------------------------------------------------------------------
#Feature lists
demographic_vars_base = [
    "Age",
    "geslacht_GegevensAfname", "geboortemandsocio",
    "modusmeanGGZ", "modusmeanPsyKl",
    "burgerlijkestaat.1", "leefsituatie.1", "leefsituatie_steun.1",
    "levenspartner.1", "opleiding_nieuw","dagbest_betaald.1", "dagbest_opleiding.1", "dagbest_dagact.1",
    "dagbest_vrijwillig.1", "dagbest_huishouden.1", "dagbest_overig.1",
    "dagbest_geen.1", "vrijwilligerswerk.1",
    "Leeftijd1eGGZ_a.1", "Leeftijd1ePsyKl_a.1",] + inkomsten_vars
questionnaire_totals_regression_only_base = ["Inspire_totaal_1", "FR_totaal.1", "honos_totaal.1"]
binary_questionnaires_base = ["Inspire_binair.1", "FR_binair.1"]
separate_questionnaires_base = inspire_items + fr_items + mansa_items + honos_items_1
#%%------------------------------------------------------------------------------------------------------------------
#Global categorical + binary columns
categorical_cols_global_base = ["geslacht_GegevensAfname","geboortemandsocio","leefsituatie.1","opleiding_nieuw",]
binary_cols_global_base = [
    "burgerlijkestaat.1","leefsituatie_steun.1","levenspartner.1",
    "dagbest_betaald.1","dagbest_opleiding.1","dagbest_dagact.1","dagbest_vrijwillig.1",
    "dagbest_huishouden.1","dagbest_overig.1","dagbest_geen.1","vrijwilligerswerk.1",
    "Inspire_binair.1","FR_binair.1",
    "Leeftijd1eGGZ_a.1", "Leeftijd1ePsyKl_a.1",] + inkomsten_vars
for c in binary_cols_global_base:
    if c in df_base.columns:
        df_base[c] = pd.to_numeric(df_base[c], errors="coerce").astype(float)
#%%----------------------------------------------------------------------------------------------
#Helper: ColumnTransformer (MICE voor numeric, most_frequent voor binary, one-hot voor categoricals)
def build_preprocessor(features: list[str]) -> ColumnTransformer:
    feats_exist = [c for c in features if c in df_base.columns]
    if len(feats_exist) == 0:
        raise ValueError("No requested features found in dataframe.")
    #The different categories of data's
    cat_cols = [c for c in categorical_cols_global_base if c in feats_exist]
    bin_cols = [c for c in binary_cols_global_base if c in feats_exist]
    num_cols = [c for c in feats_exist if c not in cat_cols + bin_cols]
    #Printen what is happening
    print(f"\nBuilding preprocessor for features: {feats_exist}")
    print(f"  Numeric: {num_cols}")
    print(f"  Binary:  {bin_cols}")
    print(f"  Categorical (one-hot): {cat_cols}")
    #Applying columntransformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", IterativeImputer(random_state=42, max_iter=50, sample_posterior=False)),
        ("scaler", StandardScaler()),])
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),]) #No scaler → 0/1 stays 0/1
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",sparse_output=False)),])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("bin", binary_transformer, bin_cols),
            ("cat", categorical_transformer, cat_cols),],
        remainder="drop",)
    return preprocessor
#%%--------------------------------------------------------------------------------------------------
#Helper: model fit + evaluate + visualize + matrix to excel
def fit_eval_linear_model(
    model_name: str,
    features: list[str],
    plot_filename: str | None = None,
    design_prefix: str | None = None,):
    feats_exist = [c for c in features if c in df_base.columns]
    X = df_base.loc[valid_idx, feats_exist].copy()
    y = y_all
    X_train = X.loc[train_idx]
    X_test  = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test  = y.loc[test_idx]
    preprocessor = build_preprocessor(feats_exist)
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("reg", LinearRegression()),])
    #Fitting
    pipe.fit(X_train, y_train)
    #Saving design-matrix after preprocessing
    if design_prefix is not None:
        pre = pipe.named_steps["preprocess"]
        X_train_trans = pre.transform(X_train)
        X_test_trans  = pre.transform(X_test)
        try:
            feature_names = pre.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feat_{i}" for i in range(X_train_trans.shape[1])]
        df_X_train = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
        df_X_test  = pd.DataFrame(X_test_trans,  columns=feature_names, index=X_test.index)
        df_X_train.to_excel(os.path.join(models_dir, f"{design_prefix}_X_train.xlsx"), index=False)
        df_X_test.to_excel(os.path.join(models_dir,  f"{design_prefix}_X_test.xlsx"),  index=False)
    #Predictions
    y_pred = pipe.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae  = mean_absolute_error(y_test, y_pred)
    mean_pred = float(np.mean(y_pred))
    print(f"\n{model_name}")
    print(f"  R²   = {r2:.3f}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAE  = {mae:.3f}")
    print(f"  mean_pred = {mean_pred:.2f}")
    #scatterplot
    if plot_filename is not None:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="black", linewidth=0.3)
        lims = [min(y_test.min(), y_pred.min()) - 2,
                max(y_test.max(), y_pred.max()) + 2]
        plt.plot(lims, lims, linestyle="--", linewidth=1)
        plt.xlim(lims); plt.ylim(lims)
        plt.xlabel("Actual MANSA_totaal.2")
        plt.ylabel("Predicted MANSA_totaal.2")
        plt.title(model_name)
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(img_dir_1, plot_filename),
                    dpi=300, bbox_inches="tight")
        plt.close()
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mean_pred": mean_pred,
        "y_test": y_test,
        "y_pred": y_pred,}
#%%-----------------------------------------------------------------------------------------------------------------------
#Dummy baseline (no preprocessing, just the mean)
X_dummy = df_base.loc[valid_idx, ["mansa_totaal.1"]].copy()
X_dummy_train = X_dummy.loc[train_idx]
X_dummy_test  = X_dummy.loc[test_idx]
y_dummy_train = y_all.loc[train_idx]
y_dummy_test  = y_all.loc[test_idx]
#Fitting it and getting results
dummy_reg_1 = DummyRegressor(strategy="mean").fit(X_dummy_train, y_dummy_train)
y_pred_dummy = dummy_reg_1.predict(X_dummy_test)
r2_dummy_1   = r2_score(y_dummy_test, y_pred_dummy)
rmse_dummy_1 = mean_squared_error(y_dummy_test, y_pred_dummy) ** 0.5
mae_dummy_1  = mean_absolute_error(y_dummy_test, y_pred_dummy)
mean_pred_dummy_1 = float(np.mean(y_pred_dummy))
#Printing the results
print(f"\nDummy baseline: R²={r2_dummy_1:.3f}, RMSE={rmse_dummy_1:.3f}, MAE={mae_dummy_1:.3f}, mean_pred={mean_pred_dummy_1:.2f}")
#%%-----------------------------------------------------------------------------------------------------------------------------------------------
#Model 1: demographics to predict MANSA_totaal.2
model1_feats = demographic_vars_base
res1 = fit_eval_linear_model(
    model_name="Model 1: Demographics/context → MANSA_totaal.2",
    features=model1_feats,
    plot_filename="linear_regression_model_1_actual_vs_predicted.png",
    design_prefix="model1_design_matrix",)
r2_reg_1   = res1["r2"]
rmse_reg_1 = res1["rmse"]
mae_reg_1  = res1["mae"]
#%%---------------------------------------------------------------------------------------------------------------------------
#Model 2: Demographics + MANSA_totaal.1 to predict MANSA_totaal.2
model2_feats = demographic_vars_base + ["mansa_totaal.1"]
res2 = fit_eval_linear_model(
    model_name="Model 2: Demographics + MANSA_totaal.1",
    features=model2_feats,
    plot_filename="linear_regression_model_2_actual_vs_predicted.png",
    design_prefix="model2_design_matrix",)
r2_reg_2  = res2["r2"]
rmse_reg_2 = res2["rmse"]
mae_reg_2  = res2["mae"]
#Improvements (vs Model 1)
print("\nImprovement over Model 1:")
print(f"  ΔR²   = {r2_reg_2 - r2_reg_1:.3f}")
print(f"  ΔRMSE = {rmse_reg_1 - rmse_reg_2:.3f}")
print(f"  ΔMAE  = {mae_reg_1 - mae_reg_2:.3f}")
#%%---------------------------------------------------------------------------------------------------------------------------
#Model 3: model 2 + totals (Inspire/FR/HONOS)
model3_feats = demographic_vars_base + ["mansa_totaal.1"] + questionnaire_totals_regression_only_base
res3 = fit_eval_linear_model(
    model_name="Model 3: Demographics + MANSA_totaal.1 + questionnaire totals",
    features=model3_feats,
    plot_filename="linear_regression_model_3_actual_vs_predicted.png",
    design_prefix="model3_design_matrix")
r2_reg_3   = res3["r2"]
rmse_reg_3 = res3["rmse"]
mae_reg_3  = res3["mae"]
#Improvements (vs Model 2)
print("\nImprovement over Model 2:")
print(f"  ΔR²   = {r2_reg_3 - r2_reg_2:.3f}")
print(f"  ΔRMSE = {rmse_reg_2 - rmse_reg_3:.3f}")
print(f"  ΔMAE  = {mae_reg_2 - mae_reg_3:.3f}")
#%%------------------------------------------------------------------------------------------------
#Model 4: Model 2 + binary questionnaire outcomes (Inspire/FR)
model4_feats = demographic_vars_base + ["mansa_totaal.1"] + binary_questionnaires_base
res4 = fit_eval_linear_model(
    model_name="Model 4: Demographics + MANSA_totaal.1 + binary questionnaire outcomes (Inspire/FR)",
    features=model4_feats,
    plot_filename="linear_regression_model_4_actual_vs_predicted.png",
    design_prefix="model4_design_matrix",)
r2_reg_4   = res4["r2"]
rmse_reg_4 = res4["rmse"]
mae_reg_4  = res4["mae"]
#Improvements (vs Model 3)
print("\nImprovement over Model 3 (binary-only extension):")
print(f"  ΔR²   = {r2_reg_4 - r2_reg_3:.3f}")
print(f"  ΔRMSE = {rmse_reg_3 - rmse_reg_4:.3f}")
print(f"  ΔMAE  = {mae_reg_3 - mae_reg_4:.3f}")
#%%---------------------------------------------------------------------------------------------------------------------------
#Model 5: Model 2 + all questionnaire items (Inspire, FR, MANSA, HoNOS)
model5_feats = demographic_vars_base + ["mansa_totaal.1"] + separate_questionnaires_base
res5 = fit_eval_linear_model(
    model_name="Model 5: Demographics + MANSA_totaal.1 + all questionnaire items",
    features=model5_feats,
    plot_filename="linear_regression_model_5_actual_vs_predicted.png",
    design_prefix="model5_design_matrix",)
r2_reg_5   = res5["r2"]
rmse_reg_5 = res5["rmse"]
mae_reg_5  = res5["mae"]
#Improvements (vs Model 4)
print("\nImprovement over Model 4:")
print(f"  ΔR²   = {r2_reg_5 - r2_reg_4:.3f}")
print(f"  ΔRMSE = {rmse_reg_4 - rmse_reg_5:.3f}")
print(f"  ΔMAE  = {mae_reg_4 - mae_reg_5:.3f}")
#%%---------------------------------------------------------------------------------
#Model 6: Demographics + MANSA_totaal.1 + totals + binary questionnaire outcomes
model6_feats = (
    demographic_vars_base
    + ["mansa_totaal.1"]
    + questionnaire_totals_regression_only_base   
    + binary_questionnaires_base)                 
print("Running Model 6...")
res6 = fit_eval_linear_model(
    model_name="Model 6: Demographics + MANSA_totaal.1 + totals + binary outcomes",
    features=model6_feats,
    plot_filename="linear_regression_model_6_actual_vs_predicted.png",
    design_prefix="model6_design_matrix")
#Improvements (vs Model 5)
print("\nImprovement over Model 5:")
print(f"  ΔR²   = {res6['r2']   - res5['r2']:.3f}")
print(f"  ΔRMSE = {res5['rmse'] - res6['rmse']:.3f}")
print(f"  ΔMAE  = {res5['mae']  - res6['mae']:.3f}")
#%%-------------------------------------------------------------------------------------
#Visualizing the results and comparing all the models
comparison = pd.DataFrame({
    "Model": ["Dummy (predicts mean)","Model 1: Demographics","Model 2: + MANSA T1","Model 3: + Totals",
        "Model 4: + Binary outcomes","Model 5: + All questionnaire items","Model 6: + Totals + Binaries",],
    "R²":   [r2_dummy_1,r2_reg_1,r2_reg_2,r2_reg_3,r2_reg_4,r2_reg_5,res6["r2"],],
    "RMSE": [rmse_dummy_1,rmse_reg_1,rmse_reg_2,rmse_reg_3,rmse_reg_4,rmse_reg_5,res6["rmse"],],
    "MAE":  [mae_dummy_1,mae_reg_1,mae_reg_2,mae_reg_3,mae_reg_4,mae_reg_5,res6["mae"],],})
print("Comparison of Models 1–6")
print(comparison.round(3))
comparison.to_csv(os.path.join(models_dir, "comparison_models_1_to_6.csv"), index=False)
comparison.to_excel(os.path.join(models_dir, "comparison_models_1_to_6.xlsx"), index=False)
#Visuals
plt.figure(figsize=(10,5))
sns.barplot(data=comparison, x="Model", y="R²")
plt.xticks(rotation=25, ha="right")
plt.title("Comparison of R² per model")
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "comparison_R2_bar.png"), dpi=300)
plt.figure(figsize=(10,5))
sns.lineplot(data=comparison, x="Model", y="RMSE", marker="o")
plt.xticks(rotation=25, ha="right")
plt.title("Comparison of RMSE per model")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "comparison_RMSE_line.png"), dpi=300)
plt.show()
plt.close()
#%%--------------------------------------------------------------------------------------------------
#Check balance of the main binary outcomes
binary_targets = ["Inspire_binair.2", "FR_binair.2"]
for col in binary_targets:
    if col in df_mansa_9_15.columns:
        print(f"\n{col} value counts:")
        print(df_mansa_9_15[col].value_counts(dropna=False))
        imbalance = df_mansa_9_15[col].value_counts(normalize=True).to_dict()
        print(f"Class proportions: {imbalance}")
#Visualize it
plt.figure(figsize=(5,4))
sns.countplot(data=df_mansa_9_15, x="Inspire_binair.2")
plt.title("Inspire_binair.2 class balance")
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "inspire_binair_2"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%------------------------------------------------------------------------------------------------------------------------------------------------------
#The feature lists without inkomsten 
open_num_cols = ["HerstelHV_4_10_.1", "vrijwilligerswerk_2_.1"]
for c in open_num_cols:
    if c in df_mansa_9_15.columns:
        df_mansa_9_15[c] = pd.to_numeric(df_mansa_9_15[c], errors="coerce")
#Renewed feature lists
demographic_vars = [
    "Age",
    "geslacht_GegevensAfname", "geboortemandsocio",
    "modusmeanGGZ", "modusmeanPsyKl",
    "burgerlijkestaat.1", "leefsituatie.1", "leefsituatie_steun.1",
    "levenspartner.1", "opleiding_nieuw","dagbest_betaald.1", "dagbest_opleiding.1", "dagbest_dagact.1",
    "dagbest_vrijwillig.1", "dagbest_huishouden.1", "dagbest_overig.1",
    "dagbest_geen.1", "vrijwilligerswerk.1",
    "Leeftijd1eGGZ_a.1", "Leeftijd1ePsyKl_a.1","HerstelHV_4_10_.1","vrijwilligerswerk_2_.1"]
questionnaire_totals_regression_only = ["Inspire_totaal_1", "FR_totaal.1", "honos_totaal.1", "mansa_totaal.1"]
binary_questionnaires = ["Inspire_binair.1", "FR_binair.1"]   #Thresholded, not used as predictors
separate_questionnaires = inspire_items + fr_items + mansa_items + honos_items_1
all_predictors_no_inkomsten = (demographic_vars + questionnaire_totals_regression_only) #No incomes, no binary threshold outcomes, no seperate questionnaire items
features_for_nested_cv = all_predictors_no_inkomsten
#%%--------------------------------------------------------------------------------------------------------------------------------------------------------
#Global categorical + binary columns (for one-hot-encoder)
categorical_cols_global = ["geslacht_GegevensAfname","geboortemandsocio","leefsituatie.1","opleiding_nieuw",]
binary_cols_global = [
    "burgerlijkestaat.1","leefsituatie_steun.1","levenspartner.1",
    "dagbest_betaald.1","dagbest_opleiding.1","dagbest_dagact.1","dagbest_vrijwillig.1",
    "dagbest_huishouden.1","dagbest_overig.1","dagbest_geen.1","vrijwilligerswerk.1",
    "Leeftijd1eGGZ_a.1", "Leeftijd1ePsyKl_a.1",]
for c in binary_cols_global:
    if c in df_mansa_9_15.columns:
        df_mansa_9_15[c] = pd.to_numeric(df_mansa_9_15[c], errors="coerce").astype(float)

#%%------------------------------------------------------------------------------------------
#Helper: Columntransformer with MinMax-Scaling for the nested CV
def build_preprocessor_minmax(df: pd.DataFrame,
                              features: list[str], 
                              categorical_cols:list[str],
                              binary_cols: list[str],
                              random_state: int
                              ) -> ColumnTransformer:
    feats_exist = [c for c in features if c in df.columns]
    if not feats_exist:
        raise ValueError("No requested features found in dataframe.")
    cat_cols = [c for c in categorical_cols if c in feats_exist]
    bin_cols = [c for c in binary_cols if c in feats_exist]
    num_cols = [c for c in feats_exist if c not in cat_cols + bin_cols]
    numeric_transformer = Pipeline(steps=[
        ("imputer", IterativeImputer(
            random_state=random_state,
            max_iter=50,
            sample_posterior=False)),
        ("scaler", MinMaxScaler()),])
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("bin", binary_transformer, bin_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",)
    return preprocessor
#%%------------------------------------------------------------------------------------------------------------------------
#Helper: Removing variables with missingness > 50%
def drop_high_missing(X_train, threshold: float = 0.50):
    miss = X_train.isna().mean()
    keep_cols = miss[miss <= threshold].index.tolist()
    drop_cols = miss[miss > threshold].index.tolist()
    return keep_cols, drop_cols
#%%------------------------------------------------------------------------------------------------------------------------
#Global setting for all the nested CV models
#Repeated nested GroupKFold CV with group-level shuffling per repetition
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
N_OUTER_SPLITS = 10   
N_INNER_SPLITS = 5     
GROUP_COL = "Proefpersoonnummer"
ycol = "mansa_totaal.2"
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
def repeated_groupkfold_splits(valid_idx, groups_all, n_splits=N_OUTER_SPLITS, seeds=SEEDS):
    valid_idx  = np.asarray(valid_idx)
    groups_all = np.asarray(groups_all)
    uniq_groups = np.unique(groups_all)
    if len(uniq_groups) < n_splits:
        raise ValueError(f"n_splits={n_splits} > #unique groups={len(uniq_groups)}")
    for rep, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        shuffled_groups = rng.permutation(uniq_groups)
        rank = {g: i for i, g in enumerate(shuffled_groups)}
        order = np.argsort([rank[g] for g in groups_all])
        vidx = valid_idx[order]
        garr = groups_all[order]
        cv = GroupKFold(n_splits=n_splits)
        for fold, (tr, te) in enumerate(cv.split(X=np.zeros(len(vidx)), y=None, groups=garr), start=1):
            yield rep, fold, vidx[tr], vidx[te]

#%%---------------------------------------------------------------------------------------------------
#XGBoost: repeated nested GroupKFold cross validation (10 seeds via SEEDS)
xgboost_cv_dir_2 = p("cross_validation_results", "xgboost_cv_results.2")
os.makedirs(xgboost_cv_dir_2, exist_ok=True)
#Target variable and groups
FEATURES = features_for_nested_cv
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)
#Parameter grid
param_grid_xgb = {
    "xgb__n_estimators":     [200,600],
    "xgb__learning_rate":    [0.03, 0.1],
    "xgb__max_depth":        [3, 6, 8 ],
    "xgb__min_child_weight": [1,3],
    "xgb__subsample":        [0.8],
    "xgb__colsample_bytree": [0.8],
    "xgb__reg_lambda":       [0.0 ,1.0],}
#Storage
rows = []
best_params_list = []
importances_per_outer = []
dropped_cols_log = []
SAVE_PREDS = False
#Running the repeated nested CV
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = xgboost_cv_dir_2
    np.save(os.path.join(out_dir, f"XGB_mm_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(out_dir, f"XGB_mm_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)
    #Raw split
    FEATURES_EXIST = [c for c in FEATURES if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, FEATURES_EXIST].copy()
    Xte_raw = df_valid.loc[outer_test_idx, FEATURES_EXIST].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    for c in categorical_cols_global:
        if c in Xte_raw.columns and c in Xtr_raw.columns:
            unseen = set(Xte_raw[c].dropna().unique()) - set(Xtr_raw[c].dropna().unique())
            if unseen:
                print(f"[rep {rep} fold {ofold}] unseen in {c}: {unseen}")
    #Drop >50% missingness based on OUTER TRAIN ONLY
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols]
    Xte_raw = Xte_raw[keep_cols]
    dropped_cols_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else""})
    #Build preprocessor (will be fit inside the pipeline during CV)
    preproc = build_preprocessor_minmax(df=Xtr_raw,
    features=keep_cols,
    categorical_cols=categorical_cols_global,
    binary_cols=binary_cols_global,
    random_state=seed)
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    #Pipeline
    pipe_xgb = Pipeline([
        ("preprocess", preproc),
        ("xgb", XGBRegressor(
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0,
        ))])
    gcv = GridSearchCV(
        estimator=pipe_xgb,
        param_grid=param_grid_xgb,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,)
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    SAVE_DESIGN = (rep == 1 and ofold == 1)
    if SAVE_DESIGN:
        pre = best_model.named_steps["preprocess"]
        Xtr_design = pd.DataFrame(
            pre.transform(Xtr_raw),
            index=Xtr_raw.index,
            columns=pre.get_feature_names_out())
        Xte_design = pd.DataFrame(
            pre.transform(Xte_raw),
            index=Xte_raw.index,
            columns=pre.get_feature_names_out())
        Xtr_design.to_csv(os.path.join(out_dir, f"XGB_mm_rep{rep:02d}_outer{ofold:02d}_Xtrain_scaled.csv"))
        Xte_design.to_csv(os.path.join(out_dir, f"XGB_mm_rep{rep:02d}_outer{ofold:02d}_Xtest_scaled.csv"))
        Xtr_design.to_excel(os.path.join(out_dir, f"XGB_mm_rep{rep:02d}_outer{ofold:02d}_Xtrain_scaled.xlsx"))
        Xte_design.to_excel(os.path.join(out_dir, f"XGB_mm_rep{rep:02d}_outer{ofold:02d}_Xtest_scaled.xlsx"))
    #Evaluating on outer test set
    ypred = best_model.predict(Xte_raw)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred) ** 0.5
    mae   = mean_absolute_error(yte, ypred)
    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2":   r2,
        "RMSE": rmse,
        "MAE":  mae,})
    best_params_list.append({
        "rep": rep,
        "fold": ofold,
        **gcv.best_params_,})
    xgb_step = best_model.named_steps["xgb"]
    try:
        feat_names = best_model.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        feat_names = [f"feat_{i}" for i in range(len(xgb_step.feature_importances_))]
    importances_per_outer.append(
        pd.Series(xgb_step.feature_importances_, index=feat_names)
          .sort_values(ascending=False)
          .rename(f"rep{rep:02d}_fold{ofold:02d}"))
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"XGB_mm_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,)
    print(
        f"[XGB_mm rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
        f"seed={seed} | dropped={len(drop_cols)} | best={gcv.best_params_} | "
        f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2":   ["mean", "std"],
    "RMSE": ["mean", "std"],
    "MAE":  ["mean", "std"],})
res_df.to_csv(os.path.join(xgboost_cv_dir_2, "xgb_mm_nestedcv_folds.csv"), index=False)
summ.to_csv(os.path.join(xgboost_cv_dir_2, "xgb_mm_nestedcv_summary.csv"))
#Saving the importance per outer fold
if importances_per_outer:
    imp_df   = pd.concat(importances_per_outer, axis=1).fillna(0.0)
    imp_mean = imp_df.mean(axis=1).sort_values(ascending=False)
    imp_mean.to_csv(os.path.join(xgboost_cv_dir_2, "xgb_mm_feature_importances_mean.csv"))
else:
    imp_mean = pd.Series(dtype=float)
pd.DataFrame(best_params_list).to_csv(
    os.path.join(xgboost_cv_dir_2, "xgb_mm_bestparams_per_outer.csv"),
    index=False,)
#Saving to excel
excel_path = os.path.join(xgboost_cv_dir_2, "xgb_mm_nestedcv_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ.to_excel(w, sheet_name="summary")
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    if not imp_mean.empty:
        imp_mean.rename("mean_importance").to_frame().to_excel(
            w, sheet_name="mean_importance")
pd.DataFrame(dropped_cols_log).to_csv(os.path.join(xgboost_cv_dir_2,"gxb_mm_dropped_cols_per_outer.csv"),index = False)
#%%--------------------------------------------------------------------------------------------------------------
#kNN: repeated nested GroupKFold cross-validation (10 seeds via SEEDS)
kNN_cv_dir_2 = p("cross_validation_results", "kNN_cv_results.2")
os.makedirs(kNN_cv_dir_2, exist_ok=True)
#Param grid
param_grid_knn = {
    "knn__n_neighbors": [3,5,7,9,11,15],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1,2],
    "knn__leaf_size": [20, 30,40],}
#Storage
rows = []
best_params_list = []
dropped_cols_log = []
SAVE_PREDS = False
inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)
#Target varible and groups
FEATURES = features_for_nested_cv
ycol = "mansa_totaal.2"
GROUP_COL = 'Proefpersoonnummer'
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()

for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx =valid_idx, groups_all=groups_all, n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = kNN_cv_dir_2
    #Saving the outputs
    np.save(os.path.join(out_dir, f"kNN_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"), outer_train_idx)
    np.save(os.path.join(out_dir, f"kNN_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),  outer_test_idx)
    #Raw split
    FEATURES_EXIST = [c for c in FEATURES if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, FEATURES_EXIST].copy()
    Xte_raw = df_valid.loc[outer_test_idx, FEATURES_EXIST].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    for c in categorical_cols_global:
        if c in Xte_raw.columns and c in Xtr_raw.columns:
            unseen = set(Xte_raw[c].dropna().unique()) - set(Xtr_raw[c].dropna().unique())
            if unseen:
                print(f"[rep {rep} fold {ofold}] unseen in {c}: {unseen}")
    #Drop > 50% missingness based on OUTER TRAIN ONLY
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols]
    Xte_raw = Xte_raw[keep_cols]
    dropped_cols_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else""})
    #Build preprocessor (fit happens inside Pipeline during inner CV)
    preproc = build_preprocessor_minmax(df=Xtr_raw, features=keep_cols, categorical_cols = categorical_cols_global, binary_cols=binary_cols_global,random_state=seed)
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    #Pipeline:Preprocess + kNN
    pipe_knn = Pipeline([
        ("preprocess", preproc),
        ("knn", KNeighborsRegressor()),
    ])
    gcv = GridSearchCV(
        estimator=pipe_knn,
        param_grid=param_grid_knn,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,)
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    SAVE_DESIGN = (rep == 1 and ofold == 1)
    if SAVE_DESIGN:
        pre = best_model.named_steps["preprocess"]
        Xtr_design = pd.DataFrame(pre.transform(Xtr_raw),index=Xtr_raw.index, columns=pre.get_feature_names_out())
        Xte_design = pd.DataFrame(pre.transform(Xte_raw),index=Xte_raw.index,columns=pre.get_feature_names_out())
        Xtr_design.to_csv(os.path.join(out_dir, f"kNN_mm_rep{rep:02d}_outer{ofold:02d}_Xtrain_scaled.csv"))
        Xte_design.to_csv(os.path.join(out_dir, f"kNN_mm_rep{rep:02d}_outer{ofold:02d}_Xtest_scaled.csv"))
    #Outer test evaluation
    ypred = best_model.predict(Xte_raw)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred)**0.5
    mae   = mean_absolute_error(yte, ypred)
    rows.append({"rep": rep, "outer_fold": ofold, "R2": r2, "RMSE": rmse, "MAE": mae})
    best_params_list.append({"rep": rep, "fold": ofold, **gcv.best_params_})
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"kNN_rep{rep:02d}_outer{ofold:02d}_preds_2.csv"),
            index=False,)
    print(f"[kNN rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | dropped={len(drop_cols)} |best={gcv.best_params_} | R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Save results
res_df = pd.DataFrame(rows)
summ = res_df.agg({"R2": ["mean","std"], "RMSE": ["mean","std"], "MAE": ["mean","std"]})
res_df.to_csv(os.path.join(kNN_cv_dir_2, "knn_repeated_nestedcv_folds_2.csv"), index=False)
summ.to_csv(os.path.join(kNN_cv_dir_2, "knn_repeated_nestedcv_summary_2.csv"))
pd.DataFrame(best_params_list).to_csv(
    os.path.join(kNN_cv_dir_2, "knn_repeated_bestparams_per_outer_2.csv"),
    index=False,)
#Save to excel
excel_path = os.path.join(kNN_cv_dir_2, "knn_repeated_nestedcv_results_2.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ.to_excel(w, sheet_name="summary")
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
pd.DataFrame(dropped_cols_log).to_csv(os.path.join(kNN_cv_dir_2, "knn_mm_dropped_cols_per_outer.csv"), index=False)
print(f"Saved all kNN results → {excel_path}")
#%%---------------------------------------------------------------------------------------------------
#Random Forest: repeated nested GroupKFold cross-validation (10 seeds via SEEDS)
random_forest_dir_2 = p("cross_validation_results", "random_forest.2")
os.makedirs(random_forest_dir_2, exist_ok=True)
#Paragram grid
param_grid_rf_m3 = {
    "rf__n_estimators":      [100, 150],
    "rf__max_depth":         [None, 10],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf":  [1, 5],
    "rf__max_features":      [0.5],}
#Storage
rows = []
best_params_list = []
importances_per_outer = []
dropped_cols_log = []
SAVE_PREDS = True
inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)
FEATURES = features_for_nested_cv
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = random_forest_dir_2
    #Save fold indices
    np.save(os.path.join(out_dir, f"RF_mm_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(out_dir, f"RF_mm_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)
    #Raw split
    FEATURES_EXIST = [c for c in FEATURES if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, FEATURES_EXIST].copy()
    Xte_raw = df_valid.loc[outer_test_idx, FEATURES_EXIST].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    for c in categorical_cols_global:
        if c in Xte_raw.columns and c in Xtr_raw.columns:
            unseen = set(Xte_raw[c].dropna().unique()) - set(Xtr_raw[c].dropna().unique())
            if unseen:
                print(f"[rep {rep} fold {ofold}] unseen in {c}: {unseen}")
    #Drop >50% missingness based on OUTER TRAIN ONLY
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols]
    Xte_raw = Xte_raw[keep_cols]
    dropped_cols_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else""})
    #Build preprocessor (fit happens inside Pipeline during inner CV)
    preproc = build_preprocessor_minmax(df=Xtr_raw,features=keep_cols,categorical_cols=categorical_cols_global,binary_cols=binary_cols_global, random_state=seed)
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    #Random Forest pipeline 
    pipe_rf = Pipeline([
        ("preprocess", preproc),
        ("rf", RandomForestRegressor(
            random_state=seed,
            n_jobs=-1,
            )),
    ])
    gcv = GridSearchCV(
        estimator=pipe_rf,
        param_grid=param_grid_rf_m3,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        refit=True,
        verbose=0,)
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    #Outer-loop evaluation
    ypred = best_model.predict(Xte_raw)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred) ** 0.5
    mae   = mean_absolute_error(yte, ypred)
    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae})
    best_params_list.append({
        "rep": rep,
        "fold": ofold,
        **gcv.best_params_})
    #Feature importances (after preprocessing)
    rf_step = best_model.named_steps["rf"]
    try:
        feat_names = best_model.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        feat_names = [f"feat_{i}" for i in range(len(rf_step.feature_importances_))]
    importances_per_outer.append(
        pd.Series(rf_step.feature_importances_, index=feat_names)
          .sort_values(ascending=False)
          .rename(f"rep{rep:02d}_fold{ofold:02d}"))
    #Save predictions
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"RF_mm_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,)
    print(f"[RF_mm rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | dropped={len(drop_cols)} |best={gcv.best_params_} | "
          f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Save results
res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2":   ["mean", "std"],
    "RMSE": ["mean", "std"],
    "MAE":  ["mean", "std"],})
res_df.to_csv(os.path.join(random_forest_dir_2, "RF_mm_nestedcv_folds.csv"), index=False)
summ.to_csv(os.path.join(random_forest_dir_2, "RF_mm_nestedcv_summary.csv"))
if importances_per_outer:
    imp_df = pd.concat(importances_per_outer, axis=1).fillna(0.0)
    imp_mean = imp_df.mean(axis=1).sort_values(ascending=False)
    imp_mean.to_csv(os.path.join(random_forest_dir_2, "RF_mm_feature_importances_mean.csv"))
else:
    imp_mean = pd.Series(dtype=float)
pd.DataFrame(best_params_list).to_csv(
    os.path.join(random_forest_dir_2, "RF_mm_bestparams_per_outer.csv"),
    index=False,)
#Saving to Excel
excel_path = os.path.join(random_forest_dir_2, "RF_mm_nestedcv_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ.to_excel(w, sheet_name="summary")
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    if not imp_mean.empty:
        imp_mean.rename("mean_importance").to_frame().to_excel(
            w, sheet_name="mean_importance")
pd.DataFrame(dropped_cols_log).to_csv(
    os.path.join(random_forest_dir_2, "RF_mm_dropped_cols_per_outer.csv"),
    index=False)
print(f"Saved all RF results → {excel_path}")
#%%---------------------------------------------------------------------------------------------------
#Linear SVR + RFE(5) with nested cross-validation with 10 seeds via SEEDS
svr_rfe5_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
os.makedirs(svr_rfe5_dir, exist_ok=True)
importances_dir = os.path.join(svr_rfe5_dir, "selected_features_per_fold")
os.makedirs(importances_dir, exist_ok=True)
#Param grid
Cs   = [0.01, 0.1, 1, 10, 30]
eps  = [0.01, 0.05, 0.1, 0.2]
tols = [1e-4, 1e-3]
param_grid_svr = {
    "svr__C":       [0.01, 0.1, 1, 10, 30],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    "svr__tol":     [1e-4, 1e-3],}     
#Storage
rows = []
best_params_list = []
dropped_cols_log = []
selected_feats_log = []
SAVE_PREDS = True
inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)
FEATURES = features_for_nested_cv
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = (
  df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()]
  .copy()
  .sort_values([GROUP_COL]) 
  .reset_index(drop=True))
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = svr_rfe5_dir
    #Save fold indices
    np.save(os.path.join(out_dir, f"svr_linear_rfe5_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(out_dir, f"svr_linear_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)
    #Raw split
    FEATURES_EXIST = [c for c in FEATURES if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, FEATURES_EXIST].copy()
    Xte_raw = df_valid.loc[outer_test_idx, FEATURES_EXIST].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    for c in categorical_cols_global:
        if c in Xte_raw.columns and c in Xtr_raw.columns:
            unseen = set(Xte_raw[c].dropna().unique()) - set(Xtr_raw[c].dropna().unique())
            if unseen:
                print(f"[rep {rep} fold {ofold}] unseen in {c}: {unseen}")
    #Drop >50% missingness based on OUTER TRAIN ONLY
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols]
    Xte_raw = Xte_raw[keep_cols]
    dropped_cols_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else ""})
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    #Build preprocessor (fit happens inside Pipeling during inner cv)
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols_global,
        binary_cols=binary_cols_global,
        random_state=seed)
    base_svr_for_rfe = LinearSVR(C=1.0, epsilon=0.1, tol=1e-4,random_state=seed, max_iter=20000)
    pipe = Pipeline([
        ("preprocess", preproc),
        ("rfe", RFE(estimator=base_svr_for_rfe, n_features_to_select=5, step=1)),
        ("svr", LinearSVR(random_state=seed, max_iter=20000)),
        ])
    gcv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_svr,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,)
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    #Save selected top-5 features (after preprocessing)
    pre = best_model.named_steps["preprocess"]
    rfe = best_model.named_steps["rfe"]
    feat_names = pre.get_feature_names_out()
    selected_feat_names = list(pd.Index(feat_names)[rfe.support_])
    if not hasattr(rfe, "estimator_") or not hasattr(rfe.estimator_, "coef_"):
        raise RuntimeError("RFE estimator has no coef_ (fit failed).")
    coefs = np.ravel(rfe.estimator_.coef_)  
    if len(coefs) != len(selected_feat_names):
        raise RuntimeError(f"coef length {len(coefs)} != selected feats {len(selected_feat_names)}")
    abs_coefs = np.abs(coefs)
    order = np.argsort(-abs_coefs)
    ranked_feats = [selected_feat_names[i] for i in order]
    ranked_abs   = [abs_coefs[i] for i in order]
    ranked_coef  = [coefs[i] for i in order]
    selected_feats_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "seed": seed,
        "selected_top5_unordered": ";".join(selected_feat_names),
        "selected_top5_ranked": ";".join(ranked_feats),
        "abs_coef_ranked": ";".join([f"{v:.6g}" for v in ranked_abs]),
        "coef_ranked": ";".join([f"{v:.6g}" for v in ranked_coef]),})
    #CSV per fold (ranked)
    pd.DataFrame({
        "feature": ranked_feats,
        "coef": ranked_coef,
        "abs_coef": ranked_abs,
        "rank": list(range(1, 6))
        }).to_csv(
            os.path.join(importances_dir, f"svr_rfe5_rep{rep:02d}_fold{ofold:02d}_features_ranked.csv"),
    index=False)
    #Outer test performance
    ypred = best_model.predict(Xte_raw)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred)**0.5
    mae   = mean_absolute_error(yte, ypred)
    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,})
    best_params_list.append({
        "rep": rep,
        "fold": ofold,
        **gcv.best_params_,})
    #Save predictions
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"svr_linear_rfe5_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,)
    print(f"[SVR-linear RFE5 rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | dropped={len(drop_cols)} | best={gcv.best_params_} | "
          f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Frequency of features in top 5
all_selected = []
for d in selected_feats_log:
    s = d.get("selected_top5_unordered", "")
    if s:
        all_selected.extend(s.split(";"))
freq = (pd.Series(Counter(all_selected))
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="count"))
freq["pct_of_folds"] = freq["count"] / len(selected_feats_log) * 100
#Mean rank of chosen features
rank_rows = []
for d in selected_feats_log:
    s = d.get("selected_top5_ranked", "")
    if not s:
        continue
    feats = s.split(";")
    for r, f in enumerate(feats, start=1):
        rank_rows.append({"feature": f, "rank_in_fold": r})
rank_df = pd.DataFrame(rank_rows)
mean_rank = (rank_df.groupby("feature")["rank_in_fold"]
             .agg(n_folds_selected="count", mean_rank="mean", sd_rank="std")
             .reset_index())
top_summary = (freq.merge(mean_rank, on="feature", how="left")
               .sort_values(["count", "mean_rank"], ascending=[False, True]))
top5_summary = top_summary.head(5).copy()
#Aggregation
res_df = pd.DataFrame(rows)
summ = res_df.agg({"R2": ["mean","std"], "RMSE": ["mean","std"], "MAE": ["mean","std"]})
summ_flat = summ.reset_index().rename(columns={"index": "metric"})
#Save outputs
res_df.to_csv(os.path.join(svr_rfe5_dir, "svr_linear_rfe5_repeated_nestedcv_folds.csv"), index=False)
summ_flat.to_csv(os.path.join(svr_rfe5_dir, "svr_linear_rfe5_repeated_nestedcv_summary.csv"), index=False)
pd.DataFrame(best_params_list).to_csv(os.path.join(svr_rfe5_dir, "svr_linear_rfe5_best_params.csv"), index=False)
pd.DataFrame(dropped_cols_log).to_csv(os.path.join(svr_rfe5_dir, "svr_mm_dropped_cols_per_outer.csv"), index=False)
pd.DataFrame(selected_feats_log).to_csv(os.path.join(svr_rfe5_dir, "svr_rfe5_selected_features_log.csv"), index=False)
freq.to_csv(os.path.join(svr_rfe5_dir, "svr_rfe5_feature_frequency.csv"), index=False)
top_summary.to_csv(os.path.join(svr_rfe5_dir, "svr_rfe5_feature_summary_all.csv"), index=False)
top5_summary.to_csv(os.path.join(svr_rfe5_dir, "svr_rfe5_top5_overall.csv"), index=False)
excel_path = os.path.join(svr_rfe5_dir, "svr_linear_rfe5_repeated_nestedcv_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ_flat.to_excel(w, sheet_name="summary", index=False)
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    pd.DataFrame(dropped_cols_log).to_excel(w, sheet_name="dropped_cols", index=False)
    pd.DataFrame(selected_feats_log).to_excel(w, sheet_name="selected_top5", index=False)
    freq.to_excel(w, sheet_name="feature_frequency", index=False)
    top_summary.to_excel(w, sheet_name="feature_summary_all", index=False)
    top5_summary.to_excel(w, sheet_name="top5_overall", index=False)
#%%---------------------------------------------------------------------------------------------------
#Linear Regression + RFE(5) repeated GroupKFold (10 seeds via SEEDS)
linear_regression_baseline_rfe5 = p("cross_validation_results", "linear_regression_rfe5")
os.makedirs(linear_regression_baseline_rfe5, exist_ok=True)
importances_dir_rfe5 = os.path.join(linear_regression_baseline_rfe5, "selected_features_per_fold")
os.makedirs(importances_dir_rfe5, exist_ok=True)
#Storage
rows = []
selected_feats_log = []
dropped_cols_log = []
SAVE_PREDS = True
#Settings
inner_cv = GroupKFold(n_splits = N_INNER_SPLITS)
FEATURES = features_for_nested_cv
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = linear_regression_baseline_rfe5
    valid_set = set(valid_idx)
    assert set(outer_train_idx).issubset(valid_set), "train_idx not subset of valid_idx"
    assert set(outer_test_idx).issubset(valid_set), "test_idx not subset of valid_idx"
    #Save indices
    np.save(os.path.join(out_dir, f"linreg_rfe5_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(out_dir, f"linreg_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)
    #Raw split
    FEATURES_EXIST = [c for c in FEATURES if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, FEATURES_EXIST].copy()
    Xte_raw = df_valid.loc[outer_test_idx, FEATURES_EXIST].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    for c in categorical_cols_global:
        if c in Xte_raw.columns and c in Xtr_raw.columns:
            unseen = set(Xte_raw[c].dropna().unique()) - set(Xtr_raw[c].dropna().unique())
            if unseen:
                print(f"[rep {rep} fold {ofold}] unseen in {c}: {unseen}")
    #Drop >50% missings based on OUTER TRAIN ONLY
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols]
    Xte_raw = Xte_raw[keep_cols]
    dropped_cols_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else ""})
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    #Preprocessing  (fit happens inside CV)
    preproc = build_preprocessor_minmax(df=Xtr_raw,features=keep_cols,categorical_cols=categorical_cols_global,binary_cols=binary_cols_global, random_state=seed)
    #Pipeline: preprocess + RFE(5)
    pipe = Pipeline([
        ("preprocess", preproc),
        ("rfe", RFE(estimator=LinearRegression(), n_features_to_select=5, step=1)),
        ("linreg", LinearRegression()),
    ])
    #inner CV (no hyperparameters to tune)
    gcv = GridSearchCV(
        estimator=pipe,
        param_grid={},          # nothing to tune
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,)
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    #Extract selected features (after preprocessing)
    pre = best_model.named_steps["preprocess"]
    rfe_step = best_model.named_steps["rfe"]
    Xtr_design = pre.transform(Xtr_raw)
    feat_names = pre.get_feature_names_out()
    assert Xtr_design.shape[1] == len(feat_names)
    assert rfe_step.support_.shape[0] == len(feat_names)
    selected_feats = list(pd.Index(feat_names)[rfe_step.support_])
    selected_feats_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "seed": seed,
        "selected_top5": ";".join(selected_feats),})
    pd.DataFrame({"selected_features": selected_feats}).to_csv(
        os.path.join(importances_dir_rfe5,
                     f"linreg_rfe5_rep{rep:02d}_fold{ofold:02d}_features.csv"),
        index=False)
    #Outer test evaluation
    ypred = best_model.predict(Xte_raw)
    r2   = r2_score(yte, ypred)
    rmse = mean_squared_error(yte, ypred) ** 0.5
    mae  = mean_absolute_error(yte, ypred)
    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,})
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir,
                         f"linreg_rfe5_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False)

    print(f"[LinReg-RFE5 rep {rep} fold {ofold}] " f"{time.time()-t0:.1f}s | dropped={len(drop_cols)} | "f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Aggregation
res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2":   ["mean", "std"],
    "RMSE": ["mean", "std"],
    "MAE":  ["mean", "std"],})
res_df.to_csv(os.path.join(linear_regression_baseline_rfe5, "linreg_rfe5_folds.csv"), index=False)
summ.to_csv(os.path.join(linear_regression_baseline_rfe5, "linreg_rfe5_summary.csv"))
pd.DataFrame(selected_feats_log).to_csv(
    os.path.join(linear_regression_baseline_rfe5, "linreg_rfe5_selected_features_log.csv"),
    index=False)
pd.DataFrame(dropped_cols_log).to_csv(
    os.path.join(linear_regression_baseline_rfe5, "linreg_rfe5_dropped_cols_per_outer.csv"),
    index=False)
excel_path = os.path.join(linear_regression_baseline_rfe5, "linreg_rfe5_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ.to_excel(w, sheet_name="summary")
    pd.DataFrame(selected_feats_log).to_excel(w, sheet_name="selected_top5", index=False)
    pd.DataFrame(dropped_cols_log).to_excel(w, sheet_name="dropped_cols", index=False)
print(f"\nSaved → {excel_path}")
print("\nLinearRegression (full feature set + RFE5) summary:")
print(summ.round(3))
#%%---------------------------------------------------------------------------------------------------
#Elastic Net (Nested CV, 10 seeds)
elastic_net_dir = p("cross_validation_results", "elastic_net_cv_results")
os.makedirs(elastic_net_dir, exist_ok=True)
#Paragram grid
param_grid_en = {
    "elasticnet__alpha":    [0.0005, 0.001, 0.01, 0.1, 1.0, 10.0],
    "elasticnet__l1_ratio": [0.1, 0.5,0.7,0.9,1.0],}
#Storage
rows = []
best_params_list = []
dropped_cols_log = []
SAVE_PREDS = True
#Settings
inner_cv = GroupKFold(n_splits = N_INNER_SPLITS)
FEATURES = features_for_nested_cv
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = elastic_net_dir
    #Save indices
    np.save(os.path.join(elastic_net_dir,
                         f"EN_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(elastic_net_dir,
                         f"EN_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)
    #Raw outer splits
    FEATURES_EXIST = [c for c in FEATURES if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, FEATURES_EXIST].copy()
    Xte_raw = df_valid.loc[outer_test_idx, FEATURES_EXIST].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    #Drop >50% missings based on OUTER TRAIN ONLY
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols]
    Xte_raw = Xte_raw[keep_cols]
    dropped_cols_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else ""})
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    #Build preprocessor (fit happens inside Pipeline during inner CV)
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols_global,
        binary_cols=binary_cols_global,
        random_state=seed)
    pipe_en = Pipeline([
        ("preprocess", preproc),
        ("elasticnet", ElasticNet(random_state=seed, max_iter=20000)),
    ])
    gcv = GridSearchCV(
        estimator=pipe_en,
        param_grid=param_grid_en,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    #Outer test evaluation
    ypred = best_model.predict(Xte_raw)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred)**0.5
    mae   = mean_absolute_error(yte, ypred)
    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
    })
    best_params_list.append({
        "rep": rep,
        "fold": ofold,
        **gcv.best_params_,
    })
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(elastic_net_dir,
                         f"EN_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False)
    print(f"[ElasticNet rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | dropped={len(drop_cols)} |best={gcv.best_params_} | "
          f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Aggregate + save
res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2":   ["mean", "std"],
    "RMSE": ["mean", "std"],
    "MAE":  ["mean", "std"],})
res_df.to_csv(os.path.join(elastic_net_dir, "elastic_net_nestedcv_folds.csv"),
              index=False)
summ_flat = summ.reset_index().rename(columns={"index": "metric"})
summ_flat.to_csv(os.path.join(elastic_net_dir, "elastic_net_nestedcv_summary.csv"), index=False)
pd.DataFrame(best_params_list).to_csv(
    os.path.join(elastic_net_dir, "elastic_net_bestparams_per_outer.csv"),
    index=False,)
pd.DataFrame(dropped_cols_log).to_csv(
    os.path.join(elastic_net_dir, "en_dropped_cols_per_outer.csv"),
    index=False)
excel_path = os.path.join(elastic_net_dir, "elastic_net_nestedcv_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ_flat.to_excel(w, sheet_name="summary", index = False)
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
print(f"Saved all ElasticNet nested CV results → {excel_path}")
print(summ_flat)
#%%----------------------------------------------------------------------------------------------------------------------------
#Making the comparison of these second run of cross validations
base_dir = p("cross_validation_results")
out_dir = os.path.join(base_dir, "comparison_all_models_2")
os.makedirs(out_dir, exist_ok=True)
#The models
models = {
    "Random Forest":     os.path.join(base_dir, "random_forest.2", "RF_mm_nestedcv_summary.csv"),
    "XGBoost":           os.path.join(base_dir, "xgboost_cv_results.2", "xgb_mm_nestedcv_summary.csv"),
    "kNN":               os.path.join(base_dir, "kNN_cv_results.2", "knn_repeated_nestedcv_summary_2.csv"),
    "Linear SVR":        os.path.join(base_dir, "svr_linear_cv_results.2_rfe5", "svr_linear_rfe5_repeated_nestedcv_summary.csv"),
    "Linear Regression": os.path.join(base_dir, "linear_regression_rfe5", "linreg_rfe5_summary.csv"),
    "Elastic Net":       os.path.join(base_dir, "elastic_net_cv_results", "elastic_net_nestedcv_summary.csv"),}
#Making the defenitions
def load_summary(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.columns[0].startswith("Unnamed"):
        wide = pd.read_csv(path, index_col=0)
        if not set(["mean","std"]).issubset(set(wide.index.astype(str))):
            raise ValueError(f"Expected mean/std rows in {path}, got {list(wide.index)}")
        df = wide.T.reset_index().rename(columns={"index": "metric", "mean": "mean", "std": "sd"})
        return df[["metric", "mean", "sd"]]
    if "metric" in raw.columns and ("mean" in set(raw["metric"])) and ("std" in set(raw["metric"])):
        wide = raw.set_index("metric")  
        df = wide.T.reset_index().rename(columns={"index": "metric", "mean": "mean", "std": "sd"})
        return df[["metric", "mean", "sd"]]
    raise ValueError(f"Unknown summary format: {path} | columns={list(raw.columns)}")
rows = []
for model_name, path in models.items():
    if os.path.exists(path):
        df = load_summary(path)
        df["Model"] = model_name
        rows.append(df)
    else:
        print(f" NOT FOUND: {model_name}: {path}")
if not rows:
    raise RuntimeError("no summary files found")
all_results = pd.concat(rows, ignore_index=True)
comparison = all_results.pivot_table(
    index="metric", columns="Model", values="mean", aggfunc="first")
sd_table = all_results.pivot_table(
    index="metric", columns="Model", values="sd", aggfunc="first")
comparison_with_sd = (
    comparison.round(3).astype(str) + " ± " + sd_table.round(3).astype(str))
#The ranking
rank_rows = {}
if "R2" in comparison.index:
    rank_rows["R2"] = comparison.loc["R2"].rank(ascending=False, method="min").astype(int)
if "RMSE" in comparison.index:
    rank_rows["RMSE"] = comparison.loc["RMSE"].rank(ascending=True, method="min").astype(int)
if "MAE" in comparison.index:
    rank_rows["MAE"] = comparison.loc["MAE"].rank(ascending=True, method="min").astype(int)
rank_df = pd.DataFrame(rank_rows)
#Write to excel
out_excel = os.path.join(out_dir, "comparison_summary_ML_models_2.xlsx")
with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    all_results.to_excel(writer, sheet_name="raw", index=False)
    comparison.to_excel(writer, sheet_name="mean_only")
    comparison_with_sd.to_excel(writer, sheet_name="mean±sd")
    rank_df.to_excel(writer, sheet_name="ranks (1=best)")
print("Comparison saved →", out_excel)
print("MEANS:")
print(comparison.round(3))
print("RANKS:")
print(rank_df)
#%%------------------------------------------------------------------------------------------
#Make visualisations
#Output directory
save_dir = p("cross_validation_results", "comparison_all_models_2")
#Load comparison results
comp_path = os.path.join(save_dir, "comparison_summary_ML_models_2.xlsx")
mean_df = pd.read_excel(comp_path, sheet_name="mean_only", index_col=0).T  
sd_with_text = pd.read_excel(comp_path, sheet_name="mean±sd", index_col=0).T
sd_df = sd_with_text.applymap(
    lambda x: float(str(x).split("±")[1].strip()) if "±" in str(x) else np.nan)
sd_df = sd_df.apply(pd.to_numeric, errors="coerce")
#Colors per model
colors = {
    "Linear Regression": "#1f77b4",
    "Linear SVR": "#ff7f0e",
    "Elastic Net": "#5a645e",
    "Random Forest": "#2ca02c",
    "XGBoost": "#9467bd",
    "kNN": "#8c564b",}
#Shorter labels for the overview
short_labels = {
    "Linear Regression": "LinReg",
    "Linear SVR": "Linear SVR",
    "Elastic Net": "Elastic Net",
    "Random Forest": "RF",
    "XGBoost": "XGBoost",
    "kNN": "kNN",}
models = mean_df.index.tolist()
x = np.arange(len(models))
#Barplot
plt.figure(figsize=(10, 5))
means = mean_df["R2"].values
errors = sd_df["R2"].values 
bar_colors = [colors[m] for m in models]
plt.bar(x, means, yerr=errors, capsize=5, edgecolor="black", color=bar_colors)
plt.ylabel("R²")
plt.title("Model Performance: R² (mean ± 1 SD across folds)")
plt.xticks(x, [short_labels[m] for m in models], rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "plot_R2_models.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "plot_R2_models.pdf"))
plt.show()
#Barplot RMSE
plt.figure(figsize=(10, 5))
means = mean_df["RMSE"].values
errors = sd_df["RMSE"].values 
plt.bar(x, means, yerr=errors, capsize=5, edgecolor="black", color=bar_colors)
plt.ylabel("RMSE")
plt.title("Model Performance: RMSE (mean ± 1 SD across folds)")
plt.xticks(x, [short_labels[m] for m in models], rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "plot_RMSE_models.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "plot_RMSE_models.pdf"))
plt.show()
#Radarplot
metrics = ["R2", "RMSE", "MAE"]
#Normalise
norm_df = mean_df.copy()
norm_df["R2"]  = (mean_df["R2"]  - mean_df["R2"].min())  / (mean_df["R2"].max()  - mean_df["R2"].min())
norm_df["RMSE"]= 1 - (mean_df["RMSE"]- mean_df["RMSE"].min())/(mean_df["RMSE"].max()- mean_df["RMSE"].min())
norm_df["MAE"] = 1 - (mean_df["MAE"] - mean_df["MAE"].min()) / (mean_df["MAE"].max() - mean_df["MAE"].min())
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  
plt.figure(figsize=(8, 8))
for model in models:
    vals = norm_df.loc[model, metrics].tolist()
    vals += vals[:1]
    c = colors[model]
    label = short_labels[model]
    plt.polar(angles, vals, marker='o', label=label, color=c)
    plt.fill(angles, vals, alpha=0.1, color=c)
plt.xticks(angles[:-1], metrics)
plt.title("Model Comparison (Normalized Radar Plot)")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "plot_radar_models.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "plot_radar_models.pdf"))
plt.show()
#%%-----------------------------------------------------------------------------------------
#Make a per-seed comparison table of all models
base_dir = p("cross_validation_results")
#mapping:
fold_files = {
    "Elastic Net": os.path.join(base_dir, "elastic_net_cv_results", "elastic_net_nestedcv_folds.csv"),
    "Linear Regression": os.path.join(base_dir, "linear_regression_rfe5", "linreg_rfe5_folds.csv"),
    "Linear SVR": os.path.join(base_dir, "svr_linear_cv_results.2_rfe5", "svr_linear_rfe5_repeated_nestedcv_folds.csv"),
    "Random Forest": os.path.join(base_dir, "random_forest.2", "RF_mm_nestedcv_folds.csv"),
    "XGBoost": os.path.join(base_dir, "xgboost_cv_results.2", "xgb_mm_nestedcv_folds.csv"),
    "kNN": os.path.join(base_dir, "kNN_cv_results.2", "knn_repeated_nestedcv_folds_2.csv"),}
all_models = []
for model_name, path in fold_files.items():
    if not os.path.exists(path):
        print("Niet gevonden:", model_name, path)
        continue
    df = pd.read_csv(path) 
    #Means per seed
    grp = (
        df.groupby("rep")[["R2", "RMSE", "MAE"]]
        .mean()
        .reset_index())
    grp["Model"] = model_name
    all_models.append(grp)
per_seed = pd.concat(all_models, ignore_index=True)
#One table per seed to excel
output_path = os.path.join(base_dir, "comparison_all_models_per_seed.xlsx")
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for rep in sorted(per_seed["rep"].unique()):
        rep_df = per_seed[per_seed["rep"] == rep]
        table = rep_df.pivot_table(
            index="Model",
            values=["MAE", "R2", "RMSE"]
        ).T 
        table.to_excel(writer, sheet_name=f"seed_{rep}")
print("10 tables saved in:", output_path)
#%%-----------------------------------------------------------------------------------
#Linear SVR uncertainty analysis (abs threshold + top 5% cutoff)
GROUP_COL = "Proefpersoonnummer"
ycol = "mansa_totaal.2"
svr_cv_dir   = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
pred_prefix  = "svr_linear_rfe5"
idx_prefix   = "svr_linear_rfe5"
UNC_THRESHOLD_ABS = 1.5
TOP_PCT = 0.05
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
pred_files = sorted(glob(os.path.join(svr_cv_dir, f"{pred_prefix}_rep*_outer*_preds.csv")))
if not pred_files:
    raise FileNotFoundError(f"No prediction files found under {svr_cv_dir}")
pred_rows = []
for pred_path in pred_files:
    m = re.search(r"rep(\d+)_outer(\d+)_preds", os.path.basename(pred_path))
    if m is None:
        continue
    rep = int(m.group(1))
    outer_fold = int(m.group(2))
    idx_path = os.path.join(
        svr_cv_dir, f"{idx_prefix}_rep{rep:02d}_outer_test_idx_{outer_fold:02d}.npy")
    preds = pd.read_csv(pred_path)
    test_idx = np.load(idx_path)
    assert test_idx.max() < len(df_valid), "test_idx out of range -> df_valid mismatch"
    if len(test_idx) != len(preds):
        raise ValueError(
            f"Length mismatch rep{rep} fold{outer_fold}: idx {len(test_idx)} vs preds {len(preds)}"
        )
    preds["rep"] = rep
    preds["outer_fold"] = outer_fold
    preds["patient_index"] = test_idx
    preds[GROUP_COL] = df_valid.iloc[test_idx][GROUP_COL].values
    pred_rows.append(preds)
all_preds = pd.concat(pred_rows, ignore_index=True)
unc_dir = os.path.join(svr_cv_dir, "uncertainty_analysis")
os.makedirs(unc_dir, exist_ok=True)
all_preds.to_csv(os.path.join(unc_dir, f"{pred_prefix}_all_preds.csv"), index=False)
#Per-patient stability
uncertainty_df = (
    all_preds.groupby(GROUP_COL)
    .agg(
        n_predictions=("y_pred", "size"),
        y_true_mean=("y_true", "mean"),
        y_pred_mean=("y_pred", "mean"),
        y_pred_std=("y_pred", "std"),
    )
    .reset_index()
)
#Absolute + relative flags
uncertainty_df["is_uncertain_abs"] = uncertainty_df["y_pred_std"] > UNC_THRESHOLD_ABS
unc_threshold_top = uncertainty_df["y_pred_std"].quantile(1 - TOP_PCT)
uncertainty_df["is_uncertain_top"] = uncertainty_df["y_pred_std"] >= unc_threshold_top
uncertainty_df.to_csv(os.path.join(unc_dir, f"{pred_prefix}_uncertainty.csv"), index=False)
#Summary
n_patients = uncertainty_df[GROUP_COL].nunique()
p50 = uncertainty_df["y_pred_std"].median()
p75 = uncertainty_df["y_pred_std"].quantile(0.75)
p90 = uncertainty_df["y_pred_std"].quantile(0.90)
p95 = uncertainty_df["y_pred_std"].quantile(0.95)
pct_abs = 100 * uncertainty_df["is_uncertain_abs"].mean()
n_abs = int(uncertainty_df["is_uncertain_abs"].sum())
pct_top = 100 * uncertainty_df["is_uncertain_top"].mean()
n_top = int(uncertainty_df["is_uncertain_top"].sum())
print(f"Linear SVR stability (prediction SD across folds) — patients: {n_patients}")
print(f"y_pred_std: median={p50:.3f}, 75th={p75:.3f}, 90th={p90:.3f}, 95th={p95:.3f}")
print(f"Abs threshold SD > {UNC_THRESHOLD_ABS}: {pct_abs:.1f}% ({n_abs}/{n_patients})")
print(f"Top {int(TOP_PCT*100)}% most variable: SD >= {unc_threshold_top:.3f}: "
      f"{pct_top:.1f}% ({n_top}/{n_patients})")
#Histogram
plt.figure(figsize=(6, 4))
sns.histplot(uncertainty_df["y_pred_std"].dropna(), bins=30)
plt.axvline(UNC_THRESHOLD_ABS, color="red", linestyle="--", label=f"Abs SD = {UNC_THRESHOLD_ABS}")
plt.axvline(unc_threshold_top, color="orange", linestyle="--", label=f"Top {int(TOP_PCT*100)}% cutoff")
plt.xlabel("Prediction SD across folds")
plt.ylabel("Count")
plt.title("Linear SVR prediction stability (per patient)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(unc_dir, f"{pred_prefix}_uncertainty_hist.png"), dpi=300)
plt.show()
plt.close()
#Scatter 
plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    uncertainty_df["y_true_mean"],
    uncertainty_df["y_pred_mean"],
    c=uncertainty_df["y_pred_std"],
    cmap="viridis",
    s=45,
)
xmin = min(uncertainty_df["y_true_mean"].min(), uncertainty_df["y_pred_mean"].min())
xmax = max(uncertainty_df["y_true_mean"].max(), uncertainty_df["y_pred_mean"].max())
pad = 1.0
xmin -= pad
xmax += pad
plt.xlim(xmin, xmax)
plt.ylim(xmin, xmax)
plt.gca().set_aspect("equal", adjustable="box")
plt.plot([xmin, xmax], [xmin, xmax], linestyle="--", color="gray")
plt.xlabel("Mean observed score")
plt.ylabel("Mean predicted score")
plt.title("Linear SVR per-patient prediction stability")
cbar = plt.colorbar(scatter)
cbar.set_label("Prediction SD")
plt.tight_layout()
plt.savefig(os.path.join(unc_dir, f"{pred_prefix}_uncertainty_scatter.png"), dpi=300)
plt.show()
plt.close()
#Robustness checks 
def calc_metrics(df, label, ytrue_col="y_true", ypred_col="y_pred"):
    r2 = r2_score(df[ytrue_col], df[ypred_col])
    rmse = mean_squared_error(df[ytrue_col], df[ypred_col]) ** 0.5
    mae = mean_absolute_error(df[ytrue_col], df[ypred_col])
    print(f"{label}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, n={len(df)}")
#patient-level dataset
patient_preds = (
    all_preds.groupby(GROUP_COL, as_index=False)
    .agg(
        y_true=("y_true", "mean"),
        y_pred=("y_pred", "mean"),
        n_preds=("y_pred", "size"),
    )
)
print("Baseline metrics (patient-level means)")
calc_metrics(patient_preds, "All patients")
#IDs to exclude
uncertain_ids_abs = set(uncertainty_df.loc[uncertainty_df["is_uncertain_abs"], GROUP_COL])
uncertain_ids_top = set(uncertainty_df.loc[uncertainty_df["is_uncertain_top"], GROUP_COL])
#ABS threshold exclusion
if len(uncertain_ids_abs) == 0:
    print(f"ABS robustness: no patients exceed SD > {UNC_THRESHOLD_ABS} -> skipped.")
else:
    patient_abs = patient_preds[~patient_preds[GROUP_COL].isin(uncertain_ids_abs)].copy()
    print(f"ABS robustness: excluding SD > {UNC_THRESHOLD_ABS} "
          f"({len(uncertain_ids_abs)}/{n_patients} patients)")
    calc_metrics(patient_abs, "Excluding uncertain (abs)")
#TOP 5% exclusion
patient_top = patient_preds[~patient_preds[GROUP_COL].isin(uncertain_ids_top)].copy()
print(f"TOP robustness: excluding top {int(TOP_PCT*100)}% most-uncertain "
      f"({len(uncertain_ids_top)}/{n_patients} patients)")
calc_metrics(patient_top, f"Excluding top {int(TOP_PCT*100)}% (relative)")

#%%------------------------------------------------------------------------------------------------------
#I want to test it for random forest to see if this explains why this model performs worse
#Add uncertainty estimation using repeated outer-fold predictions (group-aware CV)
GROUP_COL = "Proefpersoonnummer"
ycol = "mansa_totaal.2"
rf_cv_dir     = p("cross_validation_results", "random_forest.2")
pred_prefix   = "RF_mm"
idx_prefix    = "RF_mm"
UNC_THRESHOLD_ABS = 1.5   
TOP_PCT = 0.05
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
pred_files = sorted(glob(os.path.join(rf_cv_dir, f"{pred_prefix}_rep*_outer*_preds.csv")))
if not pred_files:
    raise FileNotFoundError(f"No prediction files found under {rf_cv_dir}")
pred_rows = []
for pred_path in pred_files:
    m = re.search(r"rep(\d+)_outer(\d+)_preds", os.path.basename(pred_path))
    if m is None:
        continue
    rep = int(m.group(1))
    outer_fold = int(m.group(2))
    idx_path = os.path.join(rf_cv_dir, f"{idx_prefix}_rep{rep:02d}_outer_test_idx_{outer_fold:02d}.npy")
    preds = pd.read_csv(pred_path)
    test_idx = np.load(idx_path)
    #Safety checks
    assert test_idx.max() < len(df_valid), "test_idx out of range -> df_valid mismatch"
    if len(test_idx) != len(preds):
        raise ValueError(f"Length mismatch rep{rep} fold{outer_fold}: idx {len(test_idx)} vs preds {len(preds)}")
    preds["rep"] = rep
    preds["outer_fold"] = outer_fold
    preds["patient_index"] = test_idx
    preds[GROUP_COL] = df_valid.iloc[test_idx][GROUP_COL].values
    pred_rows.append(preds)
all_preds = pd.concat(pred_rows, ignore_index=True)
unc_dir = os.path.join(rf_cv_dir, "uncertainty_analysis")
os.makedirs(unc_dir, exist_ok=True)
all_preds.to_csv(os.path.join(unc_dir, f"{pred_prefix}_all_preds.csv"), index=False)
#Per-patient stability across folds/repeats
uncertainty_df = (
    all_preds.groupby(GROUP_COL)
    .agg(
        n_predictions=("y_pred", "size"),
        y_true_mean=("y_true", "mean"),
        y_pred_mean=("y_pred", "mean"),
        y_pred_std=("y_pred", "std"),
    )
    .reset_index())
#Absolute + relative flags
uncertainty_df["is_uncertain_abs"] = uncertainty_df["y_pred_std"] > UNC_THRESHOLD_ABS
#relative cutoff so that ~TOP_PCT are flagged
unc_threshold_top = uncertainty_df["y_pred_std"].quantile(1 - TOP_PCT)
uncertainty_df["is_uncertain_top"] = uncertainty_df["y_pred_std"] >= unc_threshold_top
uncertainty_df.to_csv(os.path.join(unc_dir, f"{pred_prefix}_uncertainty.csv"), index=False)
#Summary
n_patients = uncertainty_df[GROUP_COL].nunique()
p50 = uncertainty_df["y_pred_std"].median()
p75 = uncertainty_df["y_pred_std"].quantile(0.75)
p90 = uncertainty_df["y_pred_std"].quantile(0.90)
p95 = uncertainty_df["y_pred_std"].quantile(0.95)
pct_abs = 100 * uncertainty_df["is_uncertain_abs"].mean()
n_abs = int(uncertainty_df["is_uncertain_abs"].sum())
pct_top = 100 * uncertainty_df["is_uncertain_top"].mean()
n_top = int(uncertainty_df["is_uncertain_top"].sum())
print(f"Random Forest stability (prediction SD across folds) — patients: {n_patients}")
print(f"y_pred_std: median={p50:.3f}, 75th={p75:.3f}, 90th={p90:.3f}, 95th={p95:.3f}")
print(f"Abs threshold SD > {UNC_THRESHOLD_ABS}: {pct_abs:.1f}% ({n_abs}/{n_patients})")
print(f"Top {int(TOP_PCT*100)}% most variable: SD >= {unc_threshold_top:.3f}: {pct_top:.1f}% ({n_top}/{n_patients})")
#Histogram
plt.figure(figsize=(6, 4))
sns.histplot(uncertainty_df["y_pred_std"].dropna(), bins=30)
plt.axvline(UNC_THRESHOLD_ABS, color="red", linestyle="--", label=f"Abs SD = {UNC_THRESHOLD_ABS}")
plt.axvline(unc_threshold_top, color="orange", linestyle="--", label=f"Top {int(TOP_PCT*100)}% cutoff")
plt.xlabel("Prediction SD across folds")
plt.ylabel("Count")
plt.title("Random Forest prediction stability (per patient)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(unc_dir, f"{pred_prefix}_uncertainty_hist.png"), dpi=300)
plt.show()
plt.close()
#Scatter 
plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    uncertainty_df["y_true_mean"],
    uncertainty_df["y_pred_mean"],
    c=uncertainty_df["y_pred_std"],
    cmap="viridis",
    s=45,
)
xmin = min(uncertainty_df["y_true_mean"].min(), uncertainty_df["y_pred_mean"].min())
xmax = max(uncertainty_df["y_true_mean"].max(), uncertainty_df["y_pred_mean"].max())
pad = 1.0
xmin -= pad
xmax += pad
plt.xlim(xmin, xmax)
plt.ylim(xmin, xmax)
plt.gca().set_aspect("equal", adjustable="box")
plt.plot([xmin, xmax], [xmin, xmax], linestyle="--", color="gray")
plt.xlabel("Mean observed score")
plt.ylabel("Mean predicted score")
plt.title("Random Forest per-patient prediction stability")
cbar = plt.colorbar(scatter)
cbar.set_label("Prediction SD")
plt.tight_layout()
plt.savefig(os.path.join(unc_dir, f"{pred_prefix}_uncertainty_scatter.png"), dpi=300)
plt.show()
plt.close()
#Robustness checks 
def calc_metrics(df, label, ytrue_col="y_true", ypred_col="y_pred"):
    r2 = r2_score(df[ytrue_col], df[ypred_col])
    rmse = mean_squared_error(df[ytrue_col], df[ypred_col]) ** 0.5
    mae = mean_absolute_error(df[ytrue_col], df[ypred_col])
    print(f"{label}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, n={len(df)}")
patient_preds = (
    all_preds.groupby(GROUP_COL, as_index=False)
    .agg(
        y_true=("y_true", "mean"),
        y_pred=("y_pred", "mean"),
        n_preds=("y_pred", "size"),
    )
)
print("Baseline metrics (patient-level means)")
calc_metrics(patient_preds, "All patients")
uncertain_ids_abs = set(uncertainty_df.loc[uncertainty_df["is_uncertain_abs"], GROUP_COL])
uncertain_ids_top = set(uncertainty_df.loc[uncertainty_df["is_uncertain_top"], GROUP_COL])
if len(uncertain_ids_abs) == 0:
    print(f"ABS robustness: no patients exceed SD > {UNC_THRESHOLD_ABS} -> skipped.")
else:
    patient_abs = patient_preds[~patient_preds[GROUP_COL].isin(uncertain_ids_abs)].copy()
    print(f"ABS robustness: excluding SD > {UNC_THRESHOLD_ABS} "
          f"({len(uncertain_ids_abs)}/{n_patients} patients)")
    calc_metrics(patient_abs, "Excluding uncertain (abs)")
patient_top = patient_preds[~patient_preds[GROUP_COL].isin(uncertain_ids_top)].copy()
print(f"TOP robustness: excluding top {int(TOP_PCT*100)}% most-uncertain "
      f"({len(uncertain_ids_top)}/{n_patients} patients)")
calc_metrics(patient_top, f"Excluding top {int(TOP_PCT*100)}% (relative)")
#%%------------------------------------------------------------------------------------------------------
#Coefficient analysis for the Linear SVR model used in nested CV
FEATURES = features_for_nested_cv
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
#Use the same df_valid as in the CV 
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
FEATURES_EXIST = [c for c in FEATURES if c in df_valid.columns]
X_all = df_valid[FEATURES_EXIST].copy()
y_all = df_valid[ycol].astype(float).values
FINAL_C = 30.0
svr_cv_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
idx_prefix = "svr_linear_rfe5"
#Find all folds tha exist
test_idx_files = sorted(glob(os.path.join(svr_cv_dir, f"{idx_prefix}_rep*_outer_test_idx_*.npy")))
if not test_idx_files:
    raise FileNotFoundError("No outer_test_idx files found.")
coef_rows = []
for test_idx_path in test_idx_files:
    m = re.search(r"rep(\d+)_outer_test_idx_(\d+)\.npy", os.path.basename(test_idx_path))
    if m is None:
        continue
    rep = int(m.group(1))
    outer = int(m.group(2))
    train_idx_path = os.path.join(svr_cv_dir, f"{idx_prefix}_rep{rep:02d}_outer_train_idx_{outer:02d}.npy")
    if not os.path.exists(train_idx_path):
        continue
    train_idx = np.load(train_idx_path)
    test_idx  = np.load(test_idx_path)
    #Build-in safety check
    assert train_idx.max() < len(df_valid) and test_idx.max() < len(df_valid), "idx out of range --> df_valid mismatch"
    #Mimix the dropping of the >50% missings
    Xtr_raw = X_all.iloc[train_idx].copy()
    keep_cols,drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols].copy()
    Xtr_raw = Xtr_raw.astype("object").where(pd.notna(Xtr_raw), np.nan)
    #Making sure col-lists match keep_cols
    categorical_cols = [c for c in categorical_cols_global if c in keep_cols]
    binary_cols = [c for c in binary_cols_global if c in keep_cols]
    #Build the same preprocessor as modeling 
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        random_state=rep
        )
    svr = LinearSVR(C=FINAL_C, random_state=rep, max_iter=20000)
    pipe = Pipeline([
        ("preprocess", preproc),
        ("svr", svr),
        ])
    pipe.fit(Xtr_raw, y_all[train_idx])
    #Names after preprocessing (one-hot etc.)
    feat_names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = np.ravel(pipe.named_steps["svr"].coef_)
    fold_df = pd.DataFrame({
        "rep": rep,
        "outer_fold": outer,
        "feature": feat_names,
        "coef": coefs
    })
    coef_rows.append(fold_df)
coef_long = pd.concat(coef_rows, ignore_index=True)
#Sanity check (must be 100)
print("Unique rep×outer:", coef_long[["rep", "outer_fold"]].drop_duplicates().shape[0])
#Aggregate over 10x10
coef_summary = (coef_long.groupby("feature")["coef"]
                .agg(mean="mean", sd="std", n="size")
                .assign(abs_mean=lambda d: d["mean"].abs())
                .sort_values("abs_mean", ascending=False)
                .reset_index())
#Save to excel
coef_long.to_excel(os.path.join(models_dir, "linearsvr_coefficients_10x10_long.xlsx"), index=False)
coef_summary.to_excel(os.path.join(models_dir, "linearsvr_coefficients_10x10_summary.xlsx"), index=False)
#Plot top 20
top20 = coef_summary.head(20).sort_values("mean")
plt.figure(figsize=(9, 10))
plt.barh(top20["feature"], top20["mean"])
plt.title("Linear SVR – coefficients (mean across 10x10 outer folds)")
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "linearSVR_coefplot_top20_10x10.png"), dpi=300)
plt.show()
plt.close()
#%%-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Check for multicollinearity (VIF), PURE EXPLORATORY/DESCRIPTIVE. NO DECISIONS FOR MODELLING OR FEATURE SELECTION WHERE BASED ON THIS OUTPUT
#Final predictor set used in the nested CV models
Xv = df_valid[features_for_nested_cv].copy()
#Fore numeric
Xv = Xv.apply(pd.to_numeric, errors="coerce")
#Remove constant columns
Xv = Xv.loc[:, Xv.nunique(dropna=True) > 1]
#Drop rows with missing values (VIF cannot handle missing data)
Xv = Xv.dropna(axis=0)
#Add intercept
Xv_const = sm.add_constant(Xv, has_constant="add")
#Compute VIFs
arr = Xv_const.to_numpy(dtype=float, copy=False)
vifs = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
vif_df = (
    pd.DataFrame({"feature": Xv_const.columns, "VIF": vifs})
    .query("feature!='const'")
    .sort_values("VIF", ascending=False))
#Save VIFs to excel
vif_path = os.path.join(models_dir, "model_features_vif.xlsx")
vif_df.to_excel(vif_path,index=False)
print(vif_df.head(15).round(2))
#%%--------------------------------------------------------------------------------------------------------------------
#I want to check for outliers with the features_for_nested_cv feature set on the SAME feature space as nested CV
#DESCRIPTIVE ONLY, NO DECISIONS WHERE MADE BASED ON THIS
GROUP_COL = "Proefpersoonnummer"
ycol = "mansa_totaal.2"
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
#Prepare the data
FEATURES = [c for c in features_for_nested_cv if c in df_valid.columns]
X_all = df_valid[FEATURES].copy()
svr_cv_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
idx_prefix = "svr_linear_rfe5"
#All folds
train_idx_files = sorted(glob(os.path.join(svr_cv_dir, f"{idx_prefix}_rep*_outer_train_idx_*.npy")))
if not train_idx_files:
    raise FileNotFoundError("No outer_train_idx files found.")
CONTAM = 0.05
fold_rows = []
patient_rows = []
for train_idx_path in train_idx_files:
    m = re.search(r"rep(\d+)_outer_train_idx_(\d+)\.npy", os.path.basename(train_idx_path))
    if m is None:
        continue
    rep = int(m.group(1))
    outer = int(m.group(2))
    train_idx = np.load(train_idx_path)
    assert train_idx.max() < len(df_valid), "train_idx out of range -> df_valid mismatch"
    Xtr_raw = X_all.iloc[train_idx].copy()
    #Mimic the dropping of >50% missing
    keep_cols,drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols].copy()
    Xtr_raw = Xtr_raw.astype("object").where(pd.notna(Xtr_raw), np.nan)
    #Make sure col-lists match keep_cols
    categorical_cols = [c for c in categorical_cols_global if c in keep_cols]
    binary_cols = [c for c in binary_cols_global if c in keep_cols]
    #Use same preprocessing as model
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols = categorical_cols,
        binary_cols=binary_cols,
        random_state = rep
        )
    Xtr_proc = preproc.fit_transform(Xtr_raw)
    #Isolationforest
    iso = IsolationForest(contamination=CONTAM, random_state=rep)
    labels = iso.fit_predict(Xtr_proc) # -1 = outlier, 1 = inlier
    is_out = (labels == -1)
    #Fold summary
    n = len(labels)
    n_out = int(is_out.sum())
    fold_rows.append({
        "rep": rep, "outer_fold": outer, "n_train": n,
        "n_outliers": n_out, "pct_outliers": 100*n_out/n
    })
    #save outlier indices for this fold 
    outlier_idx_train = Xtr_raw.index[is_out]
    out_path = os.path.join(models_dir, f"outliers_isoforest_train_rep{rep:02d}_outer{outer:02d}.csv")
    pd.Series(outlier_idx_train, name="row_index").to_csv(out_path, index=False)
    #per patient flag for “how often outlier”
    patient_rows.append(pd.DataFrame({
        "rep": rep,
        "outer_fold": outer,
        GROUP_COL: df_valid.loc[Xtr_raw.index, GROUP_COL].values,
        "row_index": Xtr_raw.index.values,
        "is_outlier": is_out.astype(int)
    }))
#Outputs
fold_df = pd.DataFrame(fold_rows).sort_values(["rep","outer_fold"])
print("Fold summaries (pct_outliers):")
print(fold_df["pct_outliers"].describe().round(2))
fold_df.to_excel(os.path.join(models_dir, "outliers_isoforest_fold_summaries_10x10.xlsx"), index=False)
patient_long = pd.concat(patient_rows, ignore_index=True)
patient_rate = (patient_long.groupby(GROUP_COL)["is_outlier"]
                .agg(n_folds="size", n_outlier="sum", outlier_rate="mean")
                .sort_values("outlier_rate", ascending=False)
                .reset_index())
patient_rate.to_excel(os.path.join(models_dir, "outliers_isoforest_patient_rates_10x10.xlsx"), index=False)
print("Top 10 most often flagged:")
print(patient_rate.head(10).round(3))
#Visualisation
plt.figure(figsize=(6,4))
sns.histplot(patient_rate["outlier_rate"], bins=30)
plt.xlabel("Outlier rate across folds (train appearances)")
plt.ylabel("Number of patients")
plt.title("IsolationForest outlier frequency across nested CV folds")
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "isoforest_outlier_rate_hist.png"), dpi=300)
plt.show()
plt.close()
#Another visualisation:
THRESH = 0.5 
plot_df = (
    df_valid
    .merge(patient_rate[[GROUP_COL, "outlier_rate"]],
           on=GROUP_COL, how="left")
    .fillna({"outlier_rate": 0}))
plt.figure(figsize=(6,4))
plt.scatter(
    plot_df["mansa_totaal.1"],
    plot_df["honos_totaal.1"],
    c=(plot_df["outlier_rate"] > THRESH),
    cmap="coolwarm",
    alpha=0.7)
plt.xlabel("MANSA T1")
plt.ylabel("HoNOS T1")
plt.title("Patients frequently flagged as outliers (IsolationForest)")
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "isoforest_outliers_mean_across_folds.png"),
            dpi=300)
plt.show()
plt.close()
#%%------------------------------------------------------------------------------------
#Residual analyses on Linear SVR
#Paths
svr_cv_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
img_dir = p("data_images_MANSA_9_15_months")
pred_pattern = os.path.join(svr_cv_dir, "svr_linear_rfe5_rep*_outer*_preds.csv")
pred_files = sorted(glob(pred_pattern))
if not pred_files:
    raise FileNotFoundError(f"No SVR pred files found at {pred_pattern}")
#Collect predictions
pred_rows = []
for pred_path in pred_files:
    m = re.search(r"rep(\d+)_outer(\d+)_preds", os.path.basename(pred_path))
    rep = int(m.group(1)) if m else None
    fold = int(m.group(2)) if m else None
    dfp = pd.read_csv(pred_path)
    #Sanity check
    required = {"y_true", "y_pred"}
    if not required.issubset(dfp.columns):
        raise ValueError(f"{os.path.basename(pred_path)} missing columns: {required - set(dfp.columns)}")
    dfp["rep"] = rep
    dfp["outer_fold"] = fold
    pred_rows.append(dfp)
all_preds_svr = pd.concat(pred_rows, ignore_index=True)
all_preds_svr["resid"] = all_preds_svr["y_true"] - all_preds_svr["y_pred"]
#Residuals vs predicted
plt.figure(figsize=(6, 4))
sns.scatterplot(data=all_preds_svr, x="y_pred", y="resid", alpha=0.25, s=20)
sns.regplot(data=all_preds_svr, x="y_pred", y="resid", scatter=False, lowess=True, color="red")
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Linear SVR: residuals vs predicted")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "svr_linear_rfe5_residuals_vs_pred.png"), dpi=300)
plt.show()
#Residual distribution
plt.figure(figsize=(6, 4))
sns.histplot(all_preds_svr["resid"], bins=40, kde=True)
plt.axvline(0, color="k", linestyle="--")
plt.xlabel("Residual")
plt.title("Linear SVR: residual distribution")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "svr_linear_rfe5_residual_hist.png"), dpi=300)
plt.show()
#QQ-plot
plt.figure(figsize=(5, 5))
sps.probplot(all_preds_svr["resid"], dist="norm", plot=plt)
plt.title("Linear SVR: residual QQ-plot")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "svr_linear_rfe5_residual_qq.png"), dpi=300)
plt.show()
plt.close()
#%%------------------------------------------------------------------------------------
#Residual analysis for Random Forest
#To see if it explains the reason that the data works the best with linear models
#Making the paths
random_forest_cv_dir = p("cross_validation_results", "random_forest.2")
img_dir = p("data_images_MANSA_9_15_months")
os.makedirs(img_dir, exist_ok=True)
#Look for the file
pred_pattern = os.path.join(random_forest_cv_dir, "RF_mm_rep*_outer*_preds.csv")
pred_files = sorted(glob(pred_pattern))
if not pred_files:
    raise FileNotFoundError(f"No RF pred files found at {pred_pattern}")
#Collect predictions (y_true, y_pred)
pred_rows = []
for pred_path in pred_files:
    m = re.search(r"rep(\d+)_outer(\d+)_preds", os.path.basename(pred_path))
    rep = int(m.group(1)) if m else None
    fold = int(m.group(2)) if m else None
    dfp = pd.read_csv(pred_path)
    if rep is not None and fold is not None:
        dfp["rep"] = rep
        dfp["outer_fold"] = fold
    pred_rows.append(dfp)
all_preds_rf = pd.concat(pred_rows, ignore_index=True)
all_preds_rf["resid"] = all_preds_rf["y_true"] - all_preds_rf["y_pred"]
#Residuals vs predicted
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=all_preds_rf, x="y_pred", y="resid",
    alpha=0.25, s=20, color="#1f77b4")
sns.regplot(
    data=all_preds_rf, x="y_pred", y="resid",
    scatter=False, lowess=True, color="red")
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Random Forest: residuals vs predicted")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "rf_mm_residuals_vs_pred.png"), dpi=300)
plt.show()
#Residual distribution
plt.figure(figsize=(6, 4))
sns.histplot(all_preds_rf["resid"], bins=40, kde=True, color="#1f77b4")
plt.axvline(0, color="k", linestyle="--")
plt.xlabel("Residual")
plt.title("Random Forest: residual distribution")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "rf_mm_residual_hist.png"), dpi=300)
plt.show()
#Making the QQ-plot
plt.figure(figsize=(5, 5))
sps.probplot(all_preds_rf["resid"], dist="norm", plot=plt)
plt.title("Random Forest: residual QQ-plot")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "rf_mm_residual_qq.png"), dpi=300)
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------
#Trying SHAP on the chosen Linear SVR model to see the features and their influence
img_dir = p("data_images_MANSA_9_15_months")
os.makedirs(img_dir, exist_ok=True)
svr_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
idx_prefix = "svr_linear_rfe5"
ycol = "mansa_totaal.2"
MAX_POINTS  = 3000
BG_MAX      = 200
MAX_DISPLAY = 20
#Helpers
def kill_pdna(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype("object")
    return df.where(pd.notna(df), np.nan)
def safe_float_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
#Load best SVR params (mode across folds)
bestparams_path = os.path.join(svr_dir, "svr_linear_rfe5_repeated_nestedcv_results.xlsx")
best_df = pd.read_excel(bestparams_path, sheet_name="best_params")
mode_params = best_df.mode(numeric_only=True).iloc[0].to_dict()
svr_kwargs = {}
for k, v in mode_params.items():
    if str(k).startswith("svr__"):
        try:
            svr_kwargs[k.replace("svr__", "")] = float(v)
        except Exception:
            pass
svr_kwargs["max_iter"] = 20000
print("Base SVR kwargs (mode):", svr_kwargs)
#Base data
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
FEATURES = [c for c in features_for_nested_cv if c in df_valid.columns]
X_all = df_valid[FEATURES].copy()
y_all = df_valid[ycol].astype(float).values
#Find ALL folds
train_idx_files = sorted(glob(os.path.join(svr_dir, f"{idx_prefix}_rep*_outer_train_idx_*.npy")))
if not train_idx_files:
    raise FileNotFoundError("No outer_train_idx files found.")
rng = np.random.default_rng(0)
X_df_list = []
S_df_list = []
n_used = 0
n_skipped_missing_idx = 0
n_skipped_bad_rep = 0
for train_idx_path in train_idx_files:
    m = re.search(r"rep(\d+)_outer_train_idx_(\d+)\.npy", os.path.basename(train_idx_path))
    if m is None:
        continue
    rep = int(m.group(1))      
    outer = int(m.group(2))
    test_idx_path = os.path.join(svr_dir, f"{idx_prefix}_rep{rep:02d}_outer_test_idx_{outer:02d}.npy")
    if not os.path.exists(test_idx_path):
        n_skipped_missing_idx += 1
        continue
    #rep01 -> SEEDS[0], rep02 -> SEEDS[1], ...
    rep_idx = rep - 1
    if rep_idx < 0 or rep_idx >= len(SEEDS):
        print(f"Skipping rep={rep} because SEEDS len={len(SEEDS)}")
        n_skipped_bad_rep += 1
        continue
    seed = SEEDS[rep_idx]
    train_idx = np.load(train_idx_path)
    test_idx  = np.load(test_idx_path)
    #Safety check
    if train_idx.max() >= len(df_valid) or test_idx.max() >= len(df_valid):
        print("Skipping fold (idx out of range):", rep, outer)
        continue
    Xtr_raw = X_all.iloc[train_idx].copy()
    Xte_raw = X_all.iloc[test_idx].copy()
    ytr = y_all[train_idx]
    #mimic dropping >50% missing based on TRAIN
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols].copy()
    Xte_raw = Xte_raw[keep_cols].copy()
    #kill pandas.NA everywhere
    Xtr_raw = kill_pdna(Xtr_raw)
    Xte_raw = kill_pdna(Xte_raw)
    categorical_cols = [c for c in categorical_cols_global if c in keep_cols]
    binary_cols      = [c for c in binary_cols_global if c in keep_cols]
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        random_state=seed
    )
    #Fit/transform
    Xtr = preproc.fit_transform(Xtr_raw, ytr)
    Xte = preproc.transform(Xte_raw)
    feat_names = preproc.get_feature_names_out()
    #force numeric floats for SHAP stability
    Xtr = safe_float_matrix(Xtr)
    Xte = safe_float_matrix(Xte)
    #Fit SVR
    fold_kwargs = dict(svr_kwargs)
    fold_kwargs["random_state"] = seed
    svr = LinearSVR(**fold_kwargs)
    svr.fit(Xtr, ytr)
    #SHAP explain outer-test
    bg_n = min(BG_MAX, Xtr.shape[0])
    bg_idx = rng.choice(Xtr.shape[0], size=bg_n, replace=False)
    X_bg = Xtr[bg_idx, :]
    explainer = shap.LinearExplainer(svr, X_bg)
    shap_vals = explainer(Xte).values
    shap_vals = safe_float_matrix(shap_vals)
    #store as DataFrames
    X_df_list.append(pd.DataFrame(Xte, columns=feat_names))
    S_df_list.append(pd.DataFrame(shap_vals, columns=feat_names))
    n_used += 1
if n_used == 0:
    raise RuntimeError("No folds processed. Check paths / idx files / preprocessing crashes.")
#Align to UNION of all features (missing cols -> 0)
all_cols = sorted(set().union(*[df.columns for df in X_df_list]))
X_aligned = [df.reindex(columns=all_cols, fill_value=0.0) for df in X_df_list]
S_aligned = [df.reindex(columns=all_cols, fill_value=0.0) for df in S_df_list]
X = pd.concat(X_aligned, ignore_index=True).to_numpy(dtype=np.float64)
S = pd.concat(S_aligned, ignore_index=True).to_numpy(dtype=np.float64)
#Downsample for readable beeswarm
if X.shape[0] > MAX_POINTS:
    idx = rng.choice(X.shape[0], size=MAX_POINTS, replace=False)
    X_plot = X[idx]
    S_plot = S[idx]
else:
    X_plot = X
    S_plot = S
#ONE global summary plot
plt.figure()
shap.summary_plot(S_plot, X_plot, feature_names=all_cols, max_display=MAX_DISPLAY, show=False)
plt.tight_layout()
out_sum = os.path.join(img_dir, "svr_linear_rfe5_SHAP_summary_ALLFOLDS.png")
plt.savefig(out_sum, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#Top-3 global + dependence plots
mean_abs = np.abs(S).mean(axis=0)
top_idx = np.argsort(mean_abs)[::-1][:3]
top_features = [all_cols[i] for i in top_idx]
print("Top-3 SHAP features (ALL folds):", top_features)
for feat in top_features:
    plt.figure()
    shap.dependence_plot(
        feat,
        S_plot,
        X_plot,
        feature_names=all_cols,
        interaction_index=None,
        show=False)
    plt.tight_layout()
    out_dep = os.path.join(img_dir, f"svr_linear_rfe5_SHAP_dependence_ALLFOLDS_{feat}.png")
    plt.savefig(out_dep, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
#%%-------------------------------------------------------------------
#Trying SHAP on Random Forest to see if it discovers some non-linear patterns
#Settings/paths
img_dir = p("data_images_MANSA_9_15_months")
os.makedirs(img_dir, exist_ok=True)
rf_dir = p("cross_validation_results", "random_forest.2")
idx_prefix = "RF_mm"
ycol = "mansa_totaal.2"
MAX_POINTS  = 3000  
MAX_DISPLAY = 20     
#helpers 
def kill_pdna(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype("object")
    return df.where(pd.notna(df), np.nan)
def safe_float_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
#Load RF best hyperparameters (mode across folds)
bestparams_path = os.path.join(rf_dir, "RF_mm_nestedcv_results.xlsx")
best_df = pd.read_excel(bestparams_path, sheet_name="best_params")
mode_params = best_df.mode(numeric_only=True).iloc[0].to_dict()
rf_kwargs = {k.replace("rf__", ""): mode_params[k]
             for k in mode_params if str(k).startswith("rf__")}
#make sure types are correct
for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
    if k in rf_kwargs and pd.notna(rf_kwargs[k]):
        rf_kwargs[k] = int(rf_kwargs[k])
#stable defaults
rf_kwargs.setdefault("n_jobs", -1)
print("Base RF kwargs (mode):", rf_kwargs)
#Base data (same as models)
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
FEATURES = [c for c in features_for_nested_cv if c in df_valid.columns]
X_all = df_valid[FEATURES].copy()
y_all = df_valid[ycol].astype(float).values
#Find ALL folds
train_idx_files = sorted(glob(os.path.join(rf_dir, f"{idx_prefix}_rep*_outer_train_idx_*.npy")))
if not train_idx_files:
    raise FileNotFoundError("No outer_train_idx files found.")
rng = np.random.default_rng(0)
X_df_list = []   
S_df_list = []   
n_used = 0
n_skipped_missing_idx = 0
n_skipped_crash = 0
for train_idx_path in train_idx_files:
    m = re.search(r"rep(\d+)_outer_train_idx_(\d+)\.npy", os.path.basename(train_idx_path))
    if m is None:
        continue
    rep = int(m.group(1))
    outer = int(m.group(2))
    test_idx_path = os.path.join(rf_dir, f"{idx_prefix}_rep{rep:02d}_outer_test_idx_{outer:02d}.npy")
    if not os.path.exists(test_idx_path):
        n_skipped_missing_idx += 1
        continue
    rep_idx = rep - 1
    if rep_idx < 0 or rep_idx >= len(SEEDS):
        print(f"Skipping rep={rep} (rep_idx={rep_idx}) because SEEDS len={len(SEEDS)}")
        n_skipped_crash += 1
        continue
    seed = SEEDS[rep_idx]
    train_idx = np.load(train_idx_path)
    test_idx  = np.load(test_idx_path)
    if train_idx.max() >= len(df_valid) or test_idx.max() >= len(df_valid):
        print("Skipping fold (idx out of range):", rep, outer)
        n_skipped_crash += 1
        continue
    Xtr_raw = X_all.iloc[train_idx].copy()
    Xte_raw = X_all.iloc[test_idx].copy()
    ytr = y_all[train_idx]
    #mimic dropping >50% missing (based on TRAIN)
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols].copy()
    Xte_raw = Xte_raw[keep_cols].copy()
    #kill pandas.NA
    Xtr_raw = kill_pdna(Xtr_raw)
    Xte_raw = kill_pdna(Xte_raw)
    categorical_cols = [c for c in categorical_cols_global if c in keep_cols]
    binary_cols      = [c for c in binary_cols_global if c in keep_cols]
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        random_state=seed)
    try:
        Xtr = preproc.fit_transform(Xtr_raw, ytr)
        Xte = preproc.transform(Xte_raw)
    except Exception:
        print(f"CRASH at rep={rep} outer={outer}")
        print("keep_cols n:", len(keep_cols))
        print("dtype counts:", Xtr_raw.dtypes.value_counts())
        raise
    feat_names = preproc.get_feature_names_out()
    #force numeric float (TreeExplainer)
    Xtr = safe_float_matrix(Xtr)
    Xte = safe_float_matrix(Xte)
    #Fit RF on outer-train
    fold_kwargs = dict(rf_kwargs)
    fold_kwargs["random_state"] = seed  
    rf = RandomForestRegressor(**fold_kwargs)
    rf.fit(Xtr, ytr)
    #SHAP for trees
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(Xte)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0]
    shap_vals = safe_float_matrix(shap_vals)
    #store fold results as DF for union-align later
    X_df_list.append(pd.DataFrame(Xte, columns=feat_names))
    S_df_list.append(pd.DataFrame(shap_vals, columns=feat_names))
    n_used += 1
if n_used == 0:
    raise RuntimeError("No folds processed. Check paths / idx files / preprocessing crashes.")
#Align to union of all features
all_cols = sorted(set().union(*[df.columns for df in X_df_list]))
X_aligned = [df.reindex(columns=all_cols, fill_value=0.0) for df in X_df_list]
S_aligned = [df.reindex(columns=all_cols, fill_value=0.0) for df in S_df_list]
X = pd.concat(X_aligned, ignore_index=True).to_numpy(dtype=np.float64)
S = pd.concat(S_aligned, ignore_index=True).to_numpy(dtype=np.float64)
#Downsample for readable beeswarm
if X.shape[0] > MAX_POINTS:
    idx = rng.choice(X.shape[0], size=MAX_POINTS, replace=False)
    X_plot = X[idx]
    S_plot = S[idx]
else:
    X_plot = X
    S_plot = S
#One global summary plot
plt.figure()
shap.summary_plot(S_plot, X_plot, feature_names=all_cols, max_display=MAX_DISPLAY, show=False)
plt.tight_layout()
out_sum = os.path.join(img_dir, "rf_mm_SHAP_summary_ALLFOLDS.png")
plt.savefig(out_sum, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#Top-3 global + dependence plots
mean_abs = np.abs(S).mean(axis=0)
top_idx = np.argsort(mean_abs)[::-1][:3]
top_features = [all_cols[i] for i in top_idx]
print("Top-3 SHAP features (RF, ALL folds):", top_features)
for feat in top_features:
    plt.figure()
    shap.dependence_plot(
        feat,
        S_plot,
        X_plot,
        feature_names=all_cols,
        interaction_index=None,
        show=False)
    plt.tight_layout()
    out_dep = os.path.join(img_dir, f"rf_mm_SHAP_dependence_ALLFOLDS_{feat}.png")
    plt.savefig(out_dep, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
#%%------------------------------------------------------------------------------------------
#Because of the fact that SHAP is not really useful for linear models, I am going to show the model through the coefficients and bootstrapping
#These results are emmpirical and not classical analytical CIs
#Settings
img_dir   = p("data_images_MANSA_9_15_months")
os.makedirs(img_dir, exist_ok=True)
svr_dir   = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
feat_dir  = os.path.join(svr_dir, "selected_features_per_fold")
idx_prefix = "svr_linear_rfe5"
best_xlsx  = os.path.join(svr_dir, "svr_linear_rfe5_repeated_nestedcv_results.xlsx")
ycol   = "mansa_totaal.2"
n_boot = 500  
out_dir = p("cross_validation_results", "svr_linear_bootstrap_top5_FINAL")
os.makedirs(out_dir, exist_ok=True)
TOPK_FINAL = 5
#Hyperparams (mode over all folds/reps)
best_df = pd.read_excel(best_xlsx, sheet_name="best_params")
svr_kwargs_base = {
    "C": float(best_df["svr__C"].mode().iloc[0]),
    "epsilon": float(best_df["svr__epsilon"].mode().iloc[0]),
    "tol": float(best_df["svr__tol"].mode().iloc[0]),
    "max_iter": 20000,
}
print("Using SVR kwargs (mode):", svr_kwargs_base)
#Data
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
FEATURES = [c for c in features_for_nested_cv if c in df_valid.columns]
X_all = df_valid[FEATURES].copy()
y_all = df_valid[ycol].astype(float).values
#Find outer train files
idx_files = sorted(glob(os.path.join(svr_dir, f"{idx_prefix}_rep*_outer_train_idx_*.npy")))
if not idx_files:
    raise FileNotFoundError(f"No outer train idx files found in {svr_dir}")
def parse_rep_outer(path: str):
    m = re.search(r"_rep(\d+)_outer_train_idx_(\d+)\.npy$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))
#Selection frequency (top-5 per fold)
sel_rows = []
fold_keys = []  
for train_path in idx_files:
    parsed = parse_rep_outer(train_path)
    if parsed is None:
        continue
    rep, outer = parsed
    ranked_path = os.path.join(feat_dir, f"svr_rfe5_rep{rep:02d}_fold{outer:02d}_features_ranked.csv")
    if not os.path.exists(ranked_path):
        continue
    ranked_df = pd.read_csv(ranked_path)
    top5_fold = ranked_df["feature"].head(5).tolist()
    fold_keys.append((rep, outer))
    for f in top5_fold:
        sel_rows.append({"rep": rep, "outer": outer, "feature": f})
sel_df = pd.DataFrame(sel_rows)
if sel_df.empty:
    raise RuntimeError("No ranked feature files found / selection frequency cannot be computed.")
n_folds_total = len(set(fold_keys))
freq = (sel_df.groupby("feature")
        .size()
        .reset_index(name="n_selected")
        .sort_values("n_selected", ascending=False)
        .reset_index(drop=True))
final_feats = freq["feature"].head(TOPK_FINAL).tolist()
freq_out = os.path.join(out_dir, "svr_top5_selection_frequency.csv")
freq.to_csv(freq_out, index=False)
print(f"Folds used for frequency: {n_folds_total}")
print("Top-5 most frequently selected features:", final_feats)
#Feature frequency
fold_top5 = (sel_df.groupby(["rep", "outer"])["feature"].apply(list).reset_index())
coverage_df = pd.DataFrame({
    "feature": final_feats,
    "n_folds_in_top5": [
        int((fold_top5["feature"].apply(lambda L: f in L)).sum()) for f in final_feats
    ],
})
coverage_df["pct_folds_in_top5"] = coverage_df["n_folds_in_top5"] / max(n_folds_total, 1)
#Bootstrap
pooled_rows = []
for train_path in idx_files:
    parsed = parse_rep_outer(train_path)
    if parsed is None:
        continue
    rep, outer = parsed
    ranked_path = os.path.join(feat_dir, f"svr_rfe5_rep{rep:02d}_fold{outer:02d}_features_ranked.csv")
    if not os.path.exists(ranked_path):
        continue
    #load train idx
    train_idx = np.load(train_path)
    Xtr_raw = X_all.iloc[train_idx].copy()
    ytr = y_all[train_idx]
    #mimic >50% missing dropping on outer-train
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols].copy()
    categorical_cols = [c for c in categorical_cols_global if c in keep_cols]
    binary_cols      = [c for c in binary_cols_global if c in keep_cols]
    #seed per rep
    if "SEEDS" in globals() and len(SEEDS) >= rep:
        seed = int(SEEDS[rep - 1])
    else:
        seed = int(10_000 + rep)
    svr_kwargs = dict(svr_kwargs_base)
    svr_kwargs["random_state"] = seed
    #preprocess fit on outer-train
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        random_state=seed
    )
    Xtr = preproc.fit_transform(Xtr_raw, ytr)
    feat_names = preproc.get_feature_names_out()
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    #fold top5 
    ranked_df = pd.read_csv(ranked_path)
    top5_fold = ranked_df["feature"].head(5).tolist()
    top5_fold = [f for f in top5_fold if f in name_to_idx]
    if len(top5_fold) == 0:
        continue
    Xtr_top = Xtr[:, [name_to_idx[f] for f in top5_fold]]
    #fold-specific bootstrap
    rng = np.random.default_rng(seed + 1_000_000 * outer)
    for b in range(n_boot):
        idxb = rng.integers(0, Xtr_top.shape[0], Xtr_top.shape[0])
        Xb = Xtr_top[idxb, :]
        yb = ytr[idxb]
        m = LinearSVR(**svr_kwargs)
        m.fit(Xb, yb)
        #map coef -> feature 
        coef_map = {top5_fold[j]: float(m.coef_[j]) for j in range(len(top5_fold))}
        #store only coefficients for final stable features 
        for f in final_feats:
            if f in coef_map:
                pooled_rows.append({"feature": f, "coef": coef_map[f]})
pooled = pd.DataFrame(pooled_rows)
if pooled.empty:
    raise RuntimeError("No pooled bootstrap coefs created for the final stable features.")
#The final dataframe with mean coef, 95% CI, and selection coverage
final_df = (pooled.groupby("feature")["coef"]
            .agg(
                coef_mean="mean",
                ci_2_5=lambda x: np.quantile(x, 0.025),
                ci_97_5=lambda x: np.quantile(x, 0.975),
                n_draws="count"
            )
            .reset_index())
final_df = final_df.merge(coverage_df, on="feature", how="left")
final_df["abs_coef_mean"] = final_df["coef_mean"].abs()
#Order
final_df["feature"] = pd.Categorical(final_df["feature"], categories=final_feats, ordered=True)
final_df = final_df.sort_values("feature").reset_index(drop=True)
out_csv = os.path.join(out_dir, "svr_bootstrap_FINAL_top5_stable.csv")
final_df.to_csv(out_csv, index=False)
out_xlsx = os.path.join(out_dir, "SVR_top5_feature_effects_FINAL_ALLFOLDS.xlsx")
final_df_rounded = final_df.copy()
for c in ["coef_mean", "ci_2_5", "ci_97_5", "abs_coef_mean", "pct_folds_in_top5"]:
    if c in final_df_rounded.columns:
        final_df_rounded[c] = final_df_rounded[c].astype(float).round(4)
final_df_rounded.to_excel(out_xlsx, index=False)
#Final plot
err_low  = final_df["coef_mean"] - final_df["ci_2_5"]
err_high = final_df["ci_97_5"] - final_df["coef_mean"]
plt.figure(figsize=(7, 5))
plt.barh(final_df["feature"], final_df["coef_mean"], xerr=[err_low, err_high], capsize=3)
plt.axvline(0, color="k", linewidth=1)
plt.gca().invert_yaxis()
plt.xlabel("Coefficient (preprocessed scale)")
plt.title("Linear SVR (RFE top-5) – pooled bootstrap 95% CI\n(Top-5 most frequently selected features)")
plt.tight_layout()
out_png = os.path.join(img_dir, "svr_bootstrap_ci_FINAL_top5_stable.png")
plt.savefig(out_png, dpi=300)
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------------------------------------------------
#Subquestion 2 
#Method: Repeated outer GroupKFold CV. Exact same method as the end linear svr model
subq2_dir = p("subquestion2")
os.makedirs(subq2_dir, exist_ok=True)
#Making the folder for model 2(a)
subquestion2a_dir = os.path.join(subq2_dir, "model2a")
os.makedirs(subquestion2a_dir, exist_ok=True)
importances_subquestion2a_dir = os.path.join(subquestion2a_dir, "selected_features_per_fold")
os.makedirs(importances_subquestion2a_dir, exist_ok=True)
#Defining Model features for clarity
subquestion2a_feats = demographic_vars + ["mansa_totaal.1"] + separate_questionnaires
#Param grid
Cs   = [0.01, 0.1, 1, 10, 30]
eps  = [0.01, 0.05, 0.1, 0.2]
tols = [1e-4, 1e-3]
param_grid_svr = {
    "svr__C":       [0.01, 0.1, 1, 10, 30],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    "svr__tol":     [1e-4, 1e-3],}  
#Storage
rows = []
best_params_list = []
dropped_cols_log = []
selected_feats_log = []
SAVE_PREDS = True
inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)
FEATURES = subquestion2a_feats
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = (
  df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()]
  .copy()
  .sort_values([GROUP_COL]) 
  .reset_index(drop=True))
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = subquestion2a_dir  
    #Save fold indices
    np.save(os.path.join(out_dir, f"svr_subquestion2a_linear_rfe5_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(out_dir, f"svr_subquestion2a_linear_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)
    #Raw split
    FEATURES_EXIST = [c for c in FEATURES if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, FEATURES_EXIST].copy()
    Xte_raw = df_valid.loc[outer_test_idx, FEATURES_EXIST].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    for c in categorical_cols_global:
        if c in Xte_raw.columns and c in Xtr_raw.columns:
            unseen = set(Xte_raw[c].dropna().unique()) - set(Xtr_raw[c].dropna().unique())
            if unseen:
                print(f"[rep {rep} fold {ofold}] unseen in {c}: {unseen}")
    #Drop >50% missingness based on OUTER TRAIN ONLY
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols]
    Xte_raw = Xte_raw[keep_cols]
    dropped_cols_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else ""})
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    #Build preprocessor (fit happens inside Pipeling during inner cv)
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols_global,
        binary_cols=binary_cols_global,
        random_state=seed)
    base_svr_for_rfe = LinearSVR(C=1.0, epsilon=0.1, tol=1e-4,random_state=seed, max_iter=20000)
    pipe = Pipeline([
        ("preprocess", preproc),
        ("rfe", RFE(estimator=base_svr_for_rfe, n_features_to_select=5, step=1)),
        ("svr", LinearSVR(random_state=seed, max_iter=20000)),
        ])
    gcv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_svr,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,)
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    #Save selected top-5 features (after preprocessing)
    pre = best_model.named_steps["preprocess"]
    rfe = best_model.named_steps["rfe"]
    feat_names = pre.get_feature_names_out()
    selected_feat_names = list(pd.Index(feat_names)[rfe.support_])
    if not hasattr(rfe, "estimator_") or not hasattr(rfe.estimator_, "coef_"):
        raise RuntimeError("RFE estimator has no coef_ (fit failed).")
    coefs = np.ravel(rfe.estimator_.coef_)  
    if len(coefs) != len(selected_feat_names):
        raise RuntimeError(f"coef length {len(coefs)} != selected feats {len(selected_feat_names)}")
    abs_coefs = np.abs(coefs)
    order = np.argsort(-abs_coefs)
    ranked_feats = [selected_feat_names[i] for i in order]
    ranked_abs   = [abs_coefs[i] for i in order]
    ranked_coef  = [coefs[i] for i in order]
    selected_feats_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "seed": seed,
        "selected_top5_unordered": ";".join(selected_feat_names),
        "selected_top5_ranked": ";".join(ranked_feats),
        "abs_coef_ranked": ";".join([f"{v:.6g}" for v in ranked_abs]),
        "coef_ranked": ";".join([f"{v:.6g}" for v in ranked_coef]),})
    #CSV per fold (ranked)
    pd.DataFrame({
        "feature": ranked_feats,
        "coef": ranked_coef,
        "abs_coef": ranked_abs,
        "rank": list(range(1, 6))
        }).to_csv(
            os.path.join(importances_subquestion2a_dir, f"Model2a_rep{rep:02d}_fold{ofold:02d}_features_ranked.csv"),
    index=False)
    #Outer test performance
    ypred = best_model.predict(Xte_raw)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred)**0.5
    mae   = mean_absolute_error(yte, ypred)
    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,})
    best_params_list.append({
        "rep": rep,
        "fold": ofold,
        **gcv.best_params_,})
    #Save predictions
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"Model2a_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,)
    print(f"[Model2a rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | dropped={len(drop_cols)} | best={gcv.best_params_} | "
          f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Frequency of features in top 5
all_selected = []
for d in selected_feats_log:
    s = d.get("selected_top5_unordered", "")
    if s:
        all_selected.extend(s.split(";"))
freq = (pd.Series(Counter(all_selected))
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="count"))
freq["pct_of_folds"] = freq["count"] / len(selected_feats_log) * 100
#Mean rank of chosen features
rank_rows = []
for d in selected_feats_log:
    s = d.get("selected_top5_ranked", "")
    if not s:
        continue
    feats = s.split(";")
    for r, f in enumerate(feats, start=1):
        rank_rows.append({"feature": f, "rank_in_fold": r})
rank_df = pd.DataFrame(rank_rows)
mean_rank = (rank_df.groupby("feature")["rank_in_fold"]
             .agg(n_folds_selected="count", mean_rank="mean", sd_rank="std")
             .reset_index())
top_summary = (freq.merge(mean_rank, on="feature", how="left")
               .sort_values(["count", "mean_rank"], ascending=[False, True]))
top5_summary = top_summary.head(5).copy()
#Aggregation
res_df = pd.DataFrame(rows)
summ = res_df.agg({"R2": ["mean","std"], "RMSE": ["mean","std"], "MAE": ["mean","std"]})
summ_flat = summ.reset_index().rename(columns={"index": "metric"})
#Save outputs
res_df.to_csv(os.path.join(subquestion2a_dir, "model2a_folds.csv"), index=False)
summ_flat.to_csv(os.path.join(subquestion2a_dir, "model2a_summary.csv"), index=False)
pd.DataFrame(best_params_list).to_csv(os.path.join(subquestion2a_dir, "model2a_best_params.csv"), index=False)
pd.DataFrame(dropped_cols_log).to_csv(os.path.join(subquestion2a_dir, "model2a_dropped_cols_per_outer.csv"), index=False)
pd.DataFrame(selected_feats_log).to_csv(os.path.join(subquestion2a_dir, "model2a_selected_features_log.csv"), index=False)
freq.to_csv(os.path.join(subquestion2a_dir, "model2a_feature_frequency.csv"), index=False)
top_summary.to_csv(os.path.join(subquestion2a_dir, "model2a_feature_summary_all.csv"), index=False)
top5_summary.to_csv(os.path.join(subquestion2a_dir, "model2a_top5_overall.csv"), index=False)
excel_path = os.path.join(subquestion2a_dir, "model2a_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ_flat.to_excel(w, sheet_name="summary", index=False)
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    pd.DataFrame(dropped_cols_log).to_excel(w, sheet_name="dropped_cols", index=False)
    pd.DataFrame(selected_feats_log).to_excel(w, sheet_name="selected_top5", index=False)
    freq.to_excel(w, sheet_name="feature_frequency", index=False)
    top_summary.to_excel(w, sheet_name="feature_summary_all", index=False)
    top5_summary.to_excel(w, sheet_name="top5_overall", index=False)
#%%---------------------------------------------------------------------------------------------------------------------------
#This made me wonder if the model would perform better without the MANSA_totaal.1 (limit redunancy) so I wanted to test that.
#Define features and the folder
subquestion2b_dir = os.path.join(subq2_dir, "model2b")
os.makedirs(subquestion2b_dir, exist_ok=True)
importances_subquestion2b_dir = os.path.join(subquestion2b_dir, "selected_features_per_fold")
os.makedirs(importances_subquestion2b_dir, exist_ok=True)
subquestion2b_feats = demographic_vars + separate_questionnaires
#Param grid
Cs   = [0.01, 0.1, 1, 10, 30]
eps  = [0.01, 0.05, 0.1, 0.2]
tols = [1e-4, 1e-3]
param_grid_svr = {
    "svr__C":       [0.01, 0.1, 1, 10, 30],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    "svr__tol":     [1e-4, 1e-3],}  
#Storage
rows = []
best_params_list = []
dropped_cols_log = []
selected_feats_log = []
SAVE_PREDS = True
inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)
FEATURES = subquestion2b_feats
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = (
  df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()]
  .copy()
  .sort_values([GROUP_COL]) 
  .reset_index(drop=True))
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = subquestion2b_dir  
    #Save fold indices
    np.save(os.path.join(out_dir, f"svr_subquestion2b_linear_rfe5_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(out_dir, f"svr_subquestion2b_linear_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)
    #Raw split
    FEATURES_EXIST = [c for c in FEATURES if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, FEATURES_EXIST].copy()
    Xte_raw = df_valid.loc[outer_test_idx, FEATURES_EXIST].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    for c in categorical_cols_global:
        if c in Xte_raw.columns and c in Xtr_raw.columns:
            unseen = set(Xte_raw[c].dropna().unique()) - set(Xtr_raw[c].dropna().unique())
            if unseen:
                print(f"[rep {rep} fold {ofold}] unseen in {c}: {unseen}")
    #Drop >50% missingness based on OUTER TRAIN ONLY
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols]
    Xte_raw = Xte_raw[keep_cols]
    dropped_cols_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else ""})
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    #Build preprocessor 
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols_global,
        binary_cols=binary_cols_global,
        random_state=seed)
    base_svr_for_rfe = LinearSVR(C=1.0, epsilon=0.1, tol=1e-4,random_state=seed, max_iter=20000)
    pipe = Pipeline([
        ("preprocess", preproc),
        ("rfe", RFE(estimator=base_svr_for_rfe, n_features_to_select=5, step=1)),
        ("svr", LinearSVR(random_state=seed, max_iter=20000)),
        ])
    gcv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_svr,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,)
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    #Save selected top-5 features (after preprocessing)
    pre = best_model.named_steps["preprocess"]
    rfe = best_model.named_steps["rfe"]
    feat_names = pre.get_feature_names_out()
    selected_feat_names = list(pd.Index(feat_names)[rfe.support_])
    if not hasattr(rfe, "estimator_") or not hasattr(rfe.estimator_, "coef_"):
        raise RuntimeError("RFE estimator has no coef_ (fit failed).")
    coefs = np.ravel(rfe.estimator_.coef_)  
    if len(coefs) != len(selected_feat_names):
        raise RuntimeError(f"coef length {len(coefs)} != selected feats {len(selected_feat_names)}")
    abs_coefs = np.abs(coefs)
    order = np.argsort(-abs_coefs)
    ranked_feats = [selected_feat_names[i] for i in order]
    ranked_abs   = [abs_coefs[i] for i in order]
    ranked_coef  = [coefs[i] for i in order]
    selected_feats_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "seed": seed,
        "selected_top5_unordered": ";".join(selected_feat_names),
        "selected_top5_ranked": ";".join(ranked_feats),
        "abs_coef_ranked": ";".join([f"{v:.6g}" for v in ranked_abs]),
        "coef_ranked": ";".join([f"{v:.6g}" for v in ranked_coef]),})
    #CSV per fold (ranked)
    pd.DataFrame({
        "feature": ranked_feats,
        "coef": ranked_coef,
        "abs_coef": ranked_abs,
        "rank": list(range(1, 6))
        }).to_csv(
            os.path.join(importances_subquestion2b_dir, f"Model2b_rep{rep:02d}_fold{ofold:02d}_features_ranked.csv"),
    index=False)
    #Outer test performance
    ypred = best_model.predict(Xte_raw)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred)**0.5
    mae   = mean_absolute_error(yte, ypred)
    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,})
    best_params_list.append({
        "rep": rep,
        "fold": ofold,
        **gcv.best_params_,})
    #Save predictions
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"Model2b_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,)
    print(f"[Model2b rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | dropped={len(drop_cols)} | best={gcv.best_params_} | "
          f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Frequency of features in top 5
all_selected = []
for d in selected_feats_log:
    s = d.get("selected_top5_unordered", "")
    if s:
        all_selected.extend(s.split(";"))
freq = (pd.Series(Counter(all_selected))
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="count"))
freq["pct_of_folds"] = freq["count"] / len(selected_feats_log) * 100
#Mean rank of chosen features
rank_rows = []
for d in selected_feats_log:
    s = d.get("selected_top5_ranked", "")
    if not s:
        continue
    feats = s.split(";")
    for r, f in enumerate(feats, start=1):
        rank_rows.append({"feature": f, "rank_in_fold": r})
rank_df = pd.DataFrame(rank_rows)
mean_rank = (rank_df.groupby("feature")["rank_in_fold"]
             .agg(n_folds_selected="count", mean_rank="mean", sd_rank="std")
             .reset_index())
top_summary = (freq.merge(mean_rank, on="feature", how="left")
               .sort_values(["count", "mean_rank"], ascending=[False, True]))
top5_summary = top_summary.head(5).copy()
#Aggregation
res_df = pd.DataFrame(rows)
summ = res_df.agg({"R2": ["mean","std"], "RMSE": ["mean","std"], "MAE": ["mean","std"]})
summ_flat = summ.reset_index().rename(columns={"index": "metric"})
#Save outputs
res_df.to_csv(os.path.join(subquestion2b_dir, "model2b_folds.csv"), index=False)
summ_flat.to_csv(os.path.join(subquestion2b_dir, "model2b_summary.csv"), index=False)
pd.DataFrame(best_params_list).to_csv(os.path.join(subquestion2b_dir, "model2b_best_params.csv"), index=False)
pd.DataFrame(dropped_cols_log).to_csv(os.path.join(subquestion2b_dir, "model2b_dropped_cols_per_outer.csv"), index=False)
pd.DataFrame(selected_feats_log).to_csv(os.path.join(subquestion2b_dir, "model2b_selected_features_log.csv"), index=False)
freq.to_csv(os.path.join(subquestion2b_dir, "model2b_feature_frequency.csv"), index=False)
top_summary.to_csv(os.path.join(subquestion2b_dir, "model2b_feature_summary_all.csv"), index=False)
top5_summary.to_csv(os.path.join(subquestion2b_dir, "model2b_overall.csv"), index=False)
excel_path = os.path.join(subquestion2b_dir, "model2b_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ_flat.to_excel(w, sheet_name="summary", index=False)
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    pd.DataFrame(dropped_cols_log).to_excel(w, sheet_name="dropped_cols", index=False)
    pd.DataFrame(selected_feats_log).to_excel(w, sheet_name="selected_top5", index=False)
    freq.to_excel(w, sheet_name="feature_frequency", index=False)
    top_summary.to_excel(w, sheet_name="feature_summary_all", index=False)
    top5_summary.to_excel(w, sheet_name="top5_overall", index=False)

#%%-------------------------------------------------------------------------------------------------------------------------------------------
#We know that MANSA_totaal.1 is the strongest predictor.
#I want to see what happens to the model if we remove the separate MANSA items and only leave the total in it
subquestion2c_dir = os.path.join(subq2_dir, "model2c")
os.makedirs(subquestion2c_dir, exist_ok=True)
importances_subquestion2c_dir = os.path.join(subquestion2c_dir, "selected_features_per_fold")
os.makedirs(importances_subquestion2c_dir, exist_ok=True)
subquestion2c_feats = demographic_vars + ["mansa_totaal.1"] + inspire_items + fr_items + honos_items_1
#Param grid
Cs   = [0.01, 0.1, 1, 10, 30]
eps  = [0.01, 0.05, 0.1, 0.2]
tols = [1e-4, 1e-3]
param_grid_svr = {
    "svr__C":       [0.01, 0.1, 1, 10, 30],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    "svr__tol":     [1e-4, 1e-3],}  
#Storage
rows = []
best_params_list = []
dropped_cols_log = []
selected_feats_log = []
SAVE_PREDS = True
inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)
FEATURES = subquestion2c_feats
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = (
  df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()]
  .copy()
  .sort_values([GROUP_COL]) 
  .reset_index(drop=True))
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = subquestion2c_dir  
    #Save fold indices
    np.save(os.path.join(out_dir, f"svr_subquestion2c_linear_rfe5_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(out_dir, f"svr_subquestion2c_linear_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)
    #Raw split
    FEATURES_EXIST = [c for c in FEATURES if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, FEATURES_EXIST].copy()
    Xte_raw = df_valid.loc[outer_test_idx, FEATURES_EXIST].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    for c in categorical_cols_global:
        if c in Xte_raw.columns and c in Xtr_raw.columns:
            unseen = set(Xte_raw[c].dropna().unique()) - set(Xtr_raw[c].dropna().unique())
            if unseen:
                print(f"[rep {rep} fold {ofold}] unseen in {c}: {unseen}")
    #Drop >50% missingness based on outer train only
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols]
    Xte_raw = Xte_raw[keep_cols]
    dropped_cols_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else ""})
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    #Build preprocessor 
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols_global,
        binary_cols=binary_cols_global,
        random_state=seed)
    base_svr_for_rfe = LinearSVR(C=1.0, epsilon=0.1, tol=1e-4,random_state=seed, max_iter=20000)
    pipe = Pipeline([
        ("preprocess", preproc),
        ("rfe", RFE(estimator=base_svr_for_rfe, n_features_to_select=5, step=1)),
        ("svr", LinearSVR(random_state=seed, max_iter=20000)),
        ])
    gcv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_svr,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,)
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    #Save selected top-5 features (after preprocessing)
    pre = best_model.named_steps["preprocess"]
    rfe = best_model.named_steps["rfe"]
    feat_names = pre.get_feature_names_out()
    selected_feat_names = list(pd.Index(feat_names)[rfe.support_])
    if not hasattr(rfe, "estimator_") or not hasattr(rfe.estimator_, "coef_"):
        raise RuntimeError("RFE estimator has no coef_ (fit failed).")
    coefs = np.ravel(rfe.estimator_.coef_)  
    if len(coefs) != len(selected_feat_names):
        raise RuntimeError(f"coef length {len(coefs)} != selected feats {len(selected_feat_names)}")
    abs_coefs = np.abs(coefs)
    order = np.argsort(-abs_coefs)
    ranked_feats = [selected_feat_names[i] for i in order]
    ranked_abs   = [abs_coefs[i] for i in order]
    ranked_coef  = [coefs[i] for i in order]
    selected_feats_log.append({
        "rep": rep,
        "outer_fold": ofold,
        "seed": seed,
        "selected_top5_unordered": ";".join(selected_feat_names),
        "selected_top5_ranked": ";".join(ranked_feats),
        "abs_coef_ranked": ";".join([f"{v:.6g}" for v in ranked_abs]),
        "coef_ranked": ";".join([f"{v:.6g}" for v in ranked_coef]),})
    #CSV per fold (ranked)
    pd.DataFrame({
        "feature": ranked_feats,
        "coef": ranked_coef,
        "abs_coef": ranked_abs,
        "rank": list(range(1, 6))
        }).to_csv(
            os.path.join(importances_subquestion2c_dir, f"Model2c_rep{rep:02d}_fold{ofold:02d}_features_ranked.csv"),
    index=False)
    #Outer test performance
    ypred = best_model.predict(Xte_raw)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred)**0.5
    mae   = mean_absolute_error(yte, ypred)
    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,})
    best_params_list.append({
        "rep": rep,
        "fold": ofold,
        **gcv.best_params_,})
    #Save predictions
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"Model2c_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,)
    print(f"[Model2c rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | dropped={len(drop_cols)} | best={gcv.best_params_} | "
          f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Frequency of features in top 5
all_selected = []
for d in selected_feats_log:
    s = d.get("selected_top5_unordered", "")
    if s:
        all_selected.extend(s.split(";"))
freq = (pd.Series(Counter(all_selected))
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="count"))
freq["pct_of_folds"] = freq["count"] / len(selected_feats_log) * 100
#Mean rank of chosen features
rank_rows = []
for d in selected_feats_log:
    s = d.get("selected_top5_ranked", "")
    if not s:
        continue
    feats = s.split(";")
    for r, f in enumerate(feats, start=1):
        rank_rows.append({"feature": f, "rank_in_fold": r})
rank_df = pd.DataFrame(rank_rows)
mean_rank = (rank_df.groupby("feature")["rank_in_fold"]
             .agg(n_folds_selected="count", mean_rank="mean", sd_rank="std")
             .reset_index())
top_summary = (freq.merge(mean_rank, on="feature", how="left")
               .sort_values(["count", "mean_rank"], ascending=[False, True]))
top5_summary = top_summary.head(5).copy()
#Aggregation
res_df = pd.DataFrame(rows)
summ = res_df.agg({"R2": ["mean","std"], "RMSE": ["mean","std"], "MAE": ["mean","std"]})
summ_flat = summ.reset_index().rename(columns={"index": "metric"})
#Save outputs
res_df.to_csv(os.path.join(subquestion2c_dir, "model2c_folds.csv"), index=False)
summ_flat.to_csv(os.path.join(subquestion2c_dir, "model2c_summary.csv"), index=False)
pd.DataFrame(best_params_list).to_csv(os.path.join(subquestion2c_dir, "model2c_best_params.csv"), index=False)
pd.DataFrame(dropped_cols_log).to_csv(os.path.join(subquestion2c_dir, "model2c_dropped_cols_per_outer.csv"), index=False)
pd.DataFrame(selected_feats_log).to_csv(os.path.join(subquestion2c_dir, "model2c_selected_features_log.csv"), index=False)
freq.to_csv(os.path.join(subquestion2c_dir, "model2c_feature_frequency.csv"), index=False)
top_summary.to_csv(os.path.join(subquestion2c_dir, "model2c_feature_summary_all.csv"), index=False)
top5_summary.to_csv(os.path.join(subquestion2c_dir, "model2c_overall.csv"), index=False)
excel_path = os.path.join(subquestion2c_dir, "model2c_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ_flat.to_excel(w, sheet_name="summary", index=False)
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    pd.DataFrame(dropped_cols_log).to_excel(w, sheet_name="dropped_cols", index=False)
    pd.DataFrame(selected_feats_log).to_excel(w, sheet_name="selected_top5", index=False)
    freq.to_excel(w, sheet_name="feature_frequency", index=False)
    top_summary.to_excel(w, sheet_name="feature_summary_all", index=False)
    top5_summary.to_excel(w, sheet_name="top5_overall", index=False)

#%%----------------------------------------------------------------------------------------------------------------------
#Subquestion 2d — Subscales 
#Build subscales inside each OUTER fold (train/test separately)
#MANSA item definitions
#Continuous Likert items 
mansa_likert_items = [1,2,3,4,5,6,8,12,13,14,15,16]
#Binary items NOT used in subscale means
mansa_binary_items = [7,9,10,11]
#full MANSA item list for other models
mansa_items = [f"MANSA_PH_{i}.1" for i in (mansa_likert_items + mansa_binary_items)]
#Helpers
def mean_subscale_from_frame(X, cols, max_missing_ratio=0.5):
    cols = [c for c in cols if c in X.columns]
    if len(cols) == 0:
        return pd.Series(np.nan, index=X.index)
    vals = X[cols].apply(pd.to_numeric, errors="coerce")
    allowed_missing = int(np.floor(len(cols) * max_missing_ratio))
    valid = vals.isna().sum(axis=1) <= allowed_missing
    return vals.mean(axis=1, skipna=True).where(valid, np.nan)
def add_subscales(X, mansa_likert_items, honos_items_1, inspire_items, fr_items, max_missing_ratio=0.5):
    X = X.copy()
    #MANSA subscales
    #life: 1-6
    mansa_life_idx = [i for i in [1,2,3,4,5,6] if i in mansa_likert_items]
    #social: ONLY 8 (because 7/9/10/11 are treated as binary)
    mansa_social_idx = [i for i in [8] if i in mansa_likert_items]
    #self: 12-16
    mansa_self_idx = [i for i in [12,13,14,15,16] if i in mansa_likert_items]
    mansa_life_cols   = [f"MANSA_PH_{i}.1" for i in mansa_life_idx]
    mansa_social_cols = [f"MANSA_PH_{i}.1" for i in mansa_social_idx]
    mansa_self_cols   = [f"MANSA_PH_{i}.1" for i in mansa_self_idx]
    X["mansa_life_1"]   = mean_subscale_from_frame(X, mansa_life_cols,   max_missing_ratio)
    X["mansa_social_1"] = mean_subscale_from_frame(X, mansa_social_cols, max_missing_ratio)
    X["mansa_self_1"]   = mean_subscale_from_frame(X, mansa_self_cols,   max_missing_ratio)
    #HoNOS subscales
    honos_psych_cols    = honos_items_1[0:4]
    honos_social_cols   = honos_items_1[4:8]
    honos_behavior_cols = honos_items_1[8:12]
    X["honos_psych_1"]    = mean_subscale_from_frame(X, honos_psych_cols,    max_missing_ratio)
    X["honos_social_1"]   = mean_subscale_from_frame(X, honos_social_cols,   max_missing_ratio)
    X["honos_behavior_1"] = mean_subscale_from_frame(X, honos_behavior_cols, max_missing_ratio)
    #INSPIRE subscales
    inspire_relationship_cols = inspire_items[0:2]
    inspire_collab_cols       = inspire_items[2:5]
    X["inspire_relationship_1"] = mean_subscale_from_frame(X, inspire_relationship_cols, max_missing_ratio)
    X["inspire_collab_1"]       = mean_subscale_from_frame(X, inspire_collab_cols,       max_missing_ratio)
    #FR subscales
    fr_coping_cols = fr_items[0:1]
    fr_hope_cols   = fr_items[1:3]
    X["fr_coping_1"] = mean_subscale_from_frame(X, fr_coping_cols, max_missing_ratio)
    X["fr_hope_1"]   = mean_subscale_from_frame(X, fr_hope_cols,   max_missing_ratio)
    subscale_cols = [
        "mansa_life_1","mansa_social_1","mansa_self_1",
        "honos_psych_1","honos_social_1","honos_behavior_1",
        "inspire_relationship_1","inspire_collab_1",
        "fr_coping_1","fr_hope_1"
    ]
    return X, subscale_cols
#Paths
subq2_dir = p("subquestion2")
os.makedirs(subq2_dir, exist_ok=True)
subquestion2d_dir = os.path.join(subq2_dir, "model2d_subscales")
os.makedirs(subquestion2d_dir, exist_ok=True)
importances_subquestion2d_dir = os.path.join(subquestion2d_dir, "selected_features_per_fold")
os.makedirs(importances_subquestion2d_dir, exist_ok=True)
#Model settings
param_grid_svr = {
    "svr__C":       [0.01, 0.1, 1, 10, 30],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    "svr__tol":     [1e-4, 1e-3],
}
rows = []
best_params_list = []
dropped_cols_log = []
selected_feats_log = []
SAVE_PREDS = True
inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = (
    df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()]
    .copy()
    .sort_values([GROUP_COL])
    .reset_index(drop=True)
)
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
#Outer loop
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = subquestion2d_dir
    np.save(os.path.join(out_dir, f"svr_subq2d_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"), outer_train_idx)
    np.save(os.path.join(out_dir, f"svr_subq2d_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),  outer_test_idx)
    #grab base cols needed to build subscales
    base_cols = list(set(
        demographic_vars
        + honos_items_1
        + inspire_items
        + fr_items
        + [f"MANSA_PH_{i}.1" for i in mansa_likert_items]  
    ))
    base_cols = [c for c in base_cols if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, base_cols].copy()
    Xte_raw = df_valid.loc[outer_test_idx,  base_cols].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    #build subscales per fold (train & test separately)
    Xtr_raw, subscale_cols = add_subscales(
        Xtr_raw,
        mansa_likert_items=mansa_likert_items,
        honos_items_1=honos_items_1,
        inspire_items=inspire_items,
        fr_items=fr_items,
        max_missing_ratio=0.5
    )
    Xte_raw, _ = add_subscales(
        Xte_raw,
        mansa_likert_items=mansa_likert_items,
        honos_items_1=honos_items_1,
        inspire_items=inspire_items,
        fr_items=fr_items,
        max_missing_ratio=0.5
    )
    #define features (demographics + subscales)
    FEATURES = [c for c in (demographic_vars + subscale_cols) if c in Xtr_raw.columns and c in Xte_raw.columns]
    #Drop >50% missingness based on outer train only
    keep_cols, drop_cols = drop_high_missing(Xtr_raw[FEATURES], threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols].copy()
    Xte_raw = Xte_raw[keep_cols].copy()
    dropped_cols_log.append({
        "rep": rep, "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else ""
    })
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    #Preprocessor 
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols_global,
        binary_cols=binary_cols_global,
        random_state=seed
    )
    base_svr_for_rfe = LinearSVR(C=1.0, epsilon=0.1, tol=1e-4, random_state=seed, max_iter=20000)
    pipe = Pipeline([
        ("preprocess", preproc),
        ("rfe", RFE(estimator=base_svr_for_rfe, n_features_to_select=5, step=1)),
        ("svr", LinearSVR(random_state=seed, max_iter=20000)),
    ])
    gcv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_svr,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    #Selected top-5 (post-preprocess feature names)
    pre = best_model.named_steps["preprocess"]
    rfe = best_model.named_steps["rfe"]
    feat_names = pre.get_feature_names_out()
    selected_feat_names = list(pd.Index(feat_names)[rfe.support_])
    coefs = np.ravel(rfe.estimator_.coef_)
    abs_coefs = np.abs(coefs)
    order = np.argsort(-abs_coefs)
    ranked_feats = [selected_feat_names[i] for i in order]
    ranked_abs   = [abs_coefs[i] for i in order]
    ranked_coef  = [coefs[i] for i in order]
    selected_feats_log.append({
        "rep": rep, "outer_fold": ofold, "seed": seed,
        "selected_top5_unordered": ";".join(selected_feat_names),
        "selected_top5_ranked": ";".join(ranked_feats),
        "abs_coef_ranked": ";".join([f"{v:.6g}" for v in ranked_abs]),
        "coef_ranked": ";".join([f"{v:.6g}" for v in ranked_coef]),
    })
    pd.DataFrame({
        "feature": ranked_feats,
        "coef": ranked_coef,
        "abs_coef": ranked_abs,
        "rank": list(range(1, 6))
    }).to_csv(
        os.path.join(importances_subquestion2d_dir, f"Model2d_rep{rep:02d}_fold{ofold:02d}_features_ranked.csv"),
        index=False
    )
    #Outer test performance
    ypred = best_model.predict(Xte_raw)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred) ** 0.5
    mae   = mean_absolute_error(yte, ypred)
    rows.append({"rep": rep, "outer_fold": ofold, "R2": r2, "RMSE": rmse, "MAE": mae})
    best_params_list.append({"rep": rep, "fold": ofold, **gcv.best_params_})
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep, "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred
        }).to_csv(
            os.path.join(out_dir, f"Model2d_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False
        )
    print(f"[Model2d rep {rep} fold {ofold}] {time.time()-t0:.1f}s | seed={seed} | "
          f"dropped={len(drop_cols)} | best={gcv.best_params_} | R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Summaries + save 
res_df = pd.DataFrame(rows)
summ = res_df.agg({"R2": ["mean","std"], "RMSE": ["mean","std"], "MAE": ["mean","std"]})
summ_flat = summ.reset_index().rename(columns={"index": "metric"})
#feature frequency
all_selected = []
for d in selected_feats_log:
    s = d.get("selected_top5_unordered", "")
    if s:
        all_selected.extend(s.split(";"))
freq = (pd.Series(Counter(all_selected))
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="count"))
freq["pct_of_folds"] = freq["count"] / len(selected_feats_log) * 100
#Save
res_df.to_csv(os.path.join(subquestion2d_dir, "model2d_folds.csv"), index=False)
summ_flat.to_csv(os.path.join(subquestion2d_dir, "model2d_summary.csv"), index=False)
pd.DataFrame(best_params_list).to_csv(os.path.join(subquestion2d_dir, "model2d_best_params.csv"), index=False)
pd.DataFrame(dropped_cols_log).to_csv(os.path.join(subquestion2d_dir, "model2d_dropped_cols_per_outer.csv"), index=False)
pd.DataFrame(selected_feats_log).to_csv(os.path.join(subquestion2d_dir, "model2d_selected_features_log.csv"), index=False)
freq.to_csv(os.path.join(subquestion2d_dir, "model2d_feature_frequency.csv"), index=False)
excel_path = os.path.join(subquestion2d_dir, "model2d_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ_flat.to_excel(w, sheet_name="summary", index=False)
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    pd.DataFrame(dropped_cols_log).to_excel(w, sheet_name="dropped_cols", index=False)
    pd.DataFrame(selected_feats_log).to_excel(w, sheet_name="selected_top5", index=False)
    freq.to_excel(w, sheet_name="feature_frequency", index=False)

#%%--------------------------------------------------------------------------------------------------------------
#Subquestion 2e — like 2d but:
#NO MANSA subscales
#WITH mansa_totaal.1
def mean_subscale_from_frame(X, cols, max_missing_ratio=0.5):
    cols = [c for c in cols if c in X.columns]
    if len(cols) == 0:
        return pd.Series(np.nan, index=X.index)
    vals = X[cols].apply(pd.to_numeric, errors="coerce")
    allowed_missing = int(np.floor(len(cols) * max_missing_ratio))
    valid = vals.isna().sum(axis=1) <= allowed_missing
    return vals.mean(axis=1, skipna=True).where(valid, np.nan)
def add_subscales_no_mansa(X, honos_items_1, inspire_items, fr_items, max_missing_ratio=0.5):
    X = X.copy()
    #HoNOS subscales
    honos_psych_cols    = honos_items_1[0:4]
    honos_social_cols   = honos_items_1[4:8]
    honos_behavior_cols = honos_items_1[8:12]
    X["honos_psych_1"]    = mean_subscale_from_frame(X, honos_psych_cols,    max_missing_ratio)
    X["honos_social_1"]   = mean_subscale_from_frame(X, honos_social_cols,   max_missing_ratio)
    X["honos_behavior_1"] = mean_subscale_from_frame(X, honos_behavior_cols, max_missing_ratio)
    #INSPIRE subscales
    inspire_relationship_cols = inspire_items[0:2]
    inspire_collab_cols       = inspire_items[2:5]
    X["inspire_relationship_1"] = mean_subscale_from_frame(X, inspire_relationship_cols, max_missing_ratio)
    X["inspire_collab_1"]       = mean_subscale_from_frame(X, inspire_collab_cols,       max_missing_ratio)
    #FR subscales
    fr_coping_cols = fr_items[0:1]
    fr_hope_cols   = fr_items[1:3]
    X["fr_coping_1"] = mean_subscale_from_frame(X, fr_coping_cols, max_missing_ratio)
    X["fr_hope_1"]   = mean_subscale_from_frame(X, fr_hope_cols,   max_missing_ratio)
    subscale_cols_no_mansa = [
        "honos_psych_1", "honos_social_1", "honos_behavior_1",
        "inspire_relationship_1", "inspire_collab_1",
        "fr_coping_1", "fr_hope_1"
    ]
    return X, subscale_cols_no_mansa
#Folders
subq2_dir = p("subquestion2")
os.makedirs(subq2_dir, exist_ok=True)
subquestion2e_dir = os.path.join(subq2_dir, "model2e_no_mansa_subscales_plus_mansaT1")
os.makedirs(subquestion2e_dir, exist_ok=True)
importances_subquestion2e_dir = os.path.join(subquestion2e_dir, "selected_features_per_fold")
os.makedirs(importances_subquestion2e_dir, exist_ok=True)
#Storage 
rows = []
dropped_cols_log = []
selected_feats_log = []
best_params_list = []
SAVE_PREDS = True
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = (
    df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()]
    .copy()
    .sort_values([GROUP_COL])
    .reset_index(drop=True)
)
valid_idx  = np.arange(len(df_valid))
groups_all = df_valid[GROUP_COL].to_numpy()
param_grid_svr = {
    "svr__C":       [0.01, 0.1, 1, 10, 30],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    "svr__tol":     [1e-4, 1e-3],
}
inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)
#Outer CV loop 
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):
    seed = SEEDS[rep - 1]
    out_dir = subquestion2e_dir
    np.save(os.path.join(out_dir, f"svr_subq2e_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"), outer_train_idx)
    np.save(os.path.join(out_dir, f"svr_subq2e_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),  outer_test_idx)
    #Base cols needed to build subscales + include mansa_totaal.1
    base_cols = list(set(
        demographic_vars
        + ["mansa_totaal.1"]
        + honos_items_1
        + inspire_items
        + fr_items
    ))
    base_cols = [c for c in base_cols if c in df_valid.columns]
    Xtr_raw = df_valid.loc[outer_train_idx, base_cols].copy()
    Xte_raw = df_valid.loc[outer_test_idx,  base_cols].copy()
    ytr     = df_valid.loc[outer_train_idx, ycol].astype(float)
    yte     = df_valid.loc[outer_test_idx,  ycol].astype(float)
    #Build subscales per fold (train/test separately)
    Xtr_raw, subscale_cols_no_mansa = add_subscales_no_mansa(
        Xtr_raw,
        honos_items_1=honos_items_1,
        inspire_items=inspire_items,
        fr_items=fr_items,
        max_missing_ratio=0.5
    )
    Xte_raw, _ = add_subscales_no_mansa(
        Xte_raw,
        honos_items_1=honos_items_1,
        inspire_items=inspire_items,
        fr_items=fr_items,
        max_missing_ratio=0.5
    )
    #Define features 
    FEATURES = [c for c in (demographic_vars + ["mansa_totaal.1"] + subscale_cols_no_mansa)
                if c in Xtr_raw.columns and c in Xte_raw.columns]
    #Drop >50% missingness based on outer train only
    keep_cols, drop_cols = drop_high_missing(Xtr_raw[FEATURES], threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols].copy()
    Xte_raw = Xte_raw[keep_cols].copy()
    dropped_cols_log.append({
        "rep": rep, "outer_fold": ofold,
        "n_drop": len(drop_cols),
        "dropped_cols": ";".join(drop_cols) if drop_cols else ""
    })
    #Preprocessor (fit happens inside pipeline)
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols_global,
        binary_cols=binary_cols_global,
        random_state=seed
    )
    #inner groups (group-aware tuning)
    inner_groups = df_valid.loc[outer_train_idx, GROUP_COL].to_numpy()
    base_svr_for_rfe = LinearSVR(C=1.0, epsilon=0.1, tol=1e-4,
    random_state=seed, max_iter=20000)
    pipe = Pipeline([
        ("preprocess", preproc),
        ("rfe", RFE(estimator=base_svr_for_rfe, n_features_to_select=5, step=1)),
        ("svr", LinearSVR(random_state=seed, max_iter=20000)),
        ])
    gcv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_svr,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0
        )
    t0 = time.time()
    gcv.fit(Xtr_raw, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_
    best_params_list.append({"rep": rep, "fold": ofold, **gcv.best_params_})
    #Selected top-5 (after preprocessing)
    pre = best_model.named_steps["preprocess"]
    rfe = best_model.named_steps["rfe"]
    feat_names = pre.get_feature_names_out()
    selected_feat_names = list(pd.Index(feat_names)[rfe.support_])
    coefs = np.ravel(rfe.estimator_.coef_)
    abs_coefs = np.abs(coefs)
    order = np.argsort(-abs_coefs)
    ranked_feats = [selected_feat_names[i] for i in order]
    ranked_abs   = [abs_coefs[i] for i in order]
    ranked_coef  = [coefs[i] for i in order]
    selected_feats_log.append({
        "rep": rep, "outer_fold": ofold, "seed": seed,
        "selected_top5_unordered": ";".join(selected_feat_names),
        "selected_top5_ranked": ";".join(ranked_feats),
        "abs_coef_ranked": ";".join([f"{v:.6g}" for v in ranked_abs]),
        "coef_ranked": ";".join([f"{v:.6g}" for v in ranked_coef]),
    })
    pd.DataFrame({
        "feature": ranked_feats,
        "coef": ranked_coef,
        "abs_coef": ranked_abs,
        "rank": list(range(1, 6))
    }).to_csv(
        os.path.join(importances_subquestion2e_dir,
                     f"Model2e_rep{rep:02d}_fold{ofold:02d}_features_ranked.csv"),
        index=False
    )
    #Outer test performance
    ypred = best_model.predict(Xte_raw)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred) ** 0.5
    mae   = mean_absolute_error(yte, ypred)
    rows.append({"rep": rep, "outer_fold": ofold, "R2": r2, "RMSE": rmse, "MAE": mae})
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep, "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred
        }).to_csv(
            os.path.join(out_dir, f"Model2e_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False
        )
    print(f"[Model2e rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
        f"seed={seed} | dropped={len(drop_cols)} | best={gcv.best_params_} | "
        f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Summaries + save
res_df = pd.DataFrame(rows)
summ = res_df.agg({"R2": ["mean","std"], "RMSE": ["mean","std"], "MAE": ["mean","std"]})
summ_flat = summ.reset_index().rename(columns={"index": "metric"})
#Feature frequency
all_selected = []
for d in selected_feats_log:
    s = d.get("selected_top5_unordered", "")
    if s:
        all_selected.extend(s.split(";"))
freq = (pd.Series(Counter(all_selected))
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index(name="count"))
freq["pct_of_folds"] = freq["count"] / len(selected_feats_log) * 100
#Save
res_df.to_csv(os.path.join(subquestion2e_dir, "model2e_folds.csv"), index=False)
summ_flat.to_csv(os.path.join(subquestion2e_dir, "model2e_summary.csv"), index=False)
pd.DataFrame(dropped_cols_log).to_csv(os.path.join(subquestion2e_dir, "model2e_dropped_cols_per_outer.csv"), index=False)
pd.DataFrame(selected_feats_log).to_csv(os.path.join(subquestion2e_dir, "model2e_selected_features_log.csv"), index=False)
pd.DataFrame(best_params_list).to_csv(os.path.join(subquestion2e_dir, "model2e_best_params.csv"), index=False)
freq.to_csv(os.path.join(subquestion2e_dir, "model2e_feature_frequency.csv"), index=False)
excel_path = os.path.join(subquestion2e_dir, "model2e_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ_flat.to_excel(w, sheet_name="summary", index=False)
    pd.DataFrame(dropped_cols_log).to_excel(w, sheet_name="dropped_cols", index=False)
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    pd.DataFrame(selected_feats_log).to_excel(w, sheet_name="selected_top5", index=False)
    freq.to_excel(w, sheet_name="feature_frequency", index=False)
#%%-------------------------------------------------------------------------------------------------------------------------------------------
#Summary tabel of all the models
def read_mean_sd(summary_csv_path: str) -> dict:
    df = pd.read_csv(summary_csv_path)
    #Format:mean/std rows
    if "metric" in df.columns and set(["R2", "RMSE", "MAE"]).issubset(df.columns):
        mean_row = df[df["metric"].astype(str).str.lower() == "mean"]
        std_row  = df[df["metric"].astype(str).str.lower() == "std"]
        if mean_row.empty or std_row.empty:
            raise ValueError(f"Cannot find mean/std row in {summary_csv_path}")
        out = {}
        for m in ["R2", "RMSE", "MAE"]:
            out[f"{m}_mean"] = float(mean_row[m].iloc[0])
            out[f"{m}_sd"]   = float(std_row[m].iloc[0])
        return out
#Paths
paths = {
    "Final model (Linear SVR)": p("cross_validation_results", "svr_linear_cv_results.2_rfe5",
                                  "svr_linear_rfe5_repeated_nestedcv_summary.csv"),
    "Model 2a": p("subquestion2", "model2a", "model2a_summary.csv"),
    "Model 2b": p("subquestion2", "model2b", "model2b_summary.csv"),
    "Model 2c": p("subquestion2", "model2c", "model2c_summary.csv"),
    "Model 2d (subscales)": p("subquestion2", "model2d_subscales", "model2d_summary.csv"),
    "Model 2e": p("subquestion2", "model2e_no_mansa_subscales_plus_mansaT1", "model2e_summary.csv"),
}
#Comparison table
rows = []
for name, fp in paths.items():
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Niet gevonden: {fp}")
    stats = read_mean_sd(fp)
    rows.append({"Model": name, **stats})

comp = pd.DataFrame(rows)
#Column order
col_order = [
    "Model",
    "R2_mean", "R2_sd",
    "RMSE_mean", "RMSE_sd",
    "MAE_mean", "MAE_sd"]
comp = comp[col_order].round(3)
print(comp)
#Saving
out_csv = p("subquestion2", "comparison_subq2_models.csv")
comp.to_csv(out_csv, index=False)
out_xlsx = p("subquestion2", "comparison_subq2_models.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    comp.to_excel(w, sheet_name="comparison", index=False)
#%%---------------------------------------------------------------------------------------------------------------------------
#Compact bootstrapping of top 5 features of model 2C (all folds)
subq2_dir   = p("subquestion2")
model2c_dir = os.path.join(subq2_dir, "model2c")
feat_dir    = os.path.join(model2c_dir, "selected_features_per_fold")
img_dir     = os.path.join(model2c_dir, "bootstrap_plots")
out_dir     = os.path.join(model2c_dir, "bootstrap_feature_effects_ALLFOLDS")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)
ycol   = "mansa_totaal.2"
n_boot = 500
TOPK_FINAL = 5
#indices saved by model2C run
idx_prefix = "svr_subquestion2c_linear_rfe5"
#Best param files
best_xlsx = os.path.join(model2c_dir, "model2c_results.xlsx")  
best_sheet = "best_params"   
#Load best hyperparams (mode over folds/reps) 
best_df = pd.read_excel(best_xlsx, sheet_name=best_sheet)
svr_kwargs_base = {
    "C":       float(best_df["svr__C"].mode().iloc[0]),
    "epsilon": float(best_df["svr__epsilon"].mode().iloc[0]),
    "tol":     float(best_df["svr__tol"].mode().iloc[0]),
    "max_iter": 20000,
}
print("Using SVR kwargs (mode) for Model 2c:", svr_kwargs_base)
#Data
df_valid = df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()].copy().reset_index(drop=True)
FEATURES = [c for c in subquestion2c_feats if c in df_valid.columns]
X_all = df_valid[FEATURES].copy()
y_all = df_valid[ycol].astype(float).values
#Find all outer train files
idx_files = sorted(glob(os.path.join(model2c_dir, f"{idx_prefix}_rep*_outer_train_idx_*.npy")))
if not idx_files:
    raise FileNotFoundError(f"No outer train idx files found in {model2c_dir} with prefix {idx_prefix}")
def parse_rep_outer(path: str):
    m = re.search(r"_rep(\d+)_outer_train_idx_(\d+)\.npy$", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))
#Selection frequency
sel_rows = []
fold_keys = []
for train_path in idx_files:
    parsed = parse_rep_outer(train_path)
    if parsed is None:
        continue
    rep, outer = parsed
    ranked_path = os.path.join(feat_dir, f"Model2c_rep{rep:02d}_fold{outer:02d}_features_ranked.csv")
    if not os.path.exists(ranked_path):
        continue
    ranked_df = pd.read_csv(ranked_path)
    top5_fold = ranked_df["feature"].head(5).tolist()
    fold_keys.append((rep, outer))
    for f in top5_fold:
        sel_rows.append({"rep": rep, "outer": outer, "feature": f})
sel_df = pd.DataFrame(sel_rows)
if sel_df.empty:
    raise RuntimeError("No ranked feature files found for Model 2c (check feat_dir + filenames).")
n_folds_total = len(set(fold_keys))
freq = (sel_df.groupby("feature")
        .size()
        .reset_index(name="n_selected")
        .sort_values("n_selected", ascending=False)
        .reset_index(drop=True))
final_feats = freq["feature"].head(TOPK_FINAL).tolist()
freq.to_csv(os.path.join(out_dir, "model2c_top_selection_frequency.csv"), index=False)
#coverage in how many folds the final features appear in fold-top5
fold_top5 = sel_df.groupby(["rep", "outer"])["feature"].apply(list).reset_index()
coverage_df = pd.DataFrame({
    "feature": final_feats,
    "n_folds_in_top5": [int((fold_top5["feature"].apply(lambda L: f in L)).sum()) for f in final_feats],
})
coverage_df["pct_folds_in_top5"] = coverage_df["n_folds_in_top5"] / max(n_folds_total, 1)
coverage_df["pct_folds_in_top5"] = 100 * coverage_df["pct_folds_in_top5"]
#Bootstrap pooling
pooled_rows = []
for train_path in idx_files:
    parsed = parse_rep_outer(train_path)
    if parsed is None:
        continue
    rep, outer = parsed
    ranked_path = os.path.join(feat_dir, f"Model2c_rep{rep:02d}_fold{outer:02d}_features_ranked.csv")
    if not os.path.exists(ranked_path):
        continue
    train_idx = np.load(train_path)
    Xtr_raw = X_all.iloc[train_idx].copy()
    ytr = y_all[train_idx]
    #mimic missing-drop on outer-train
    keep_cols, drop_cols = drop_high_missing(Xtr_raw, threshold=0.50)
    Xtr_raw = Xtr_raw[keep_cols].copy()
    categorical_cols = [c for c in categorical_cols_global if c in keep_cols]
    binary_cols      = [c for c in binary_cols_global if c in keep_cols]
    #seed per rep 
    seed = int(SEEDS[rep - 1]) if ("SEEDS" in globals() and len(SEEDS) >= rep) else int(10_000 + rep)
    svr_kwargs = dict(svr_kwargs_base)
    svr_kwargs["random_state"] = seed
    #preprocess fit on outer-train only
    preproc = build_preprocessor_minmax(
        df=Xtr_raw,
        features=keep_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
        random_state=seed
    )
    Xtr = preproc.fit_transform(Xtr_raw, ytr)
    feat_names = preproc.get_feature_names_out()
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    #fold top5 (as used in that fold)
    ranked_df = pd.read_csv(ranked_path)
    use_feats = [f for f in final_feats if f in name_to_idx]
    if len(use_feats) == 0:
        continue
    Xtr_use = Xtr[:, [name_to_idx[f] for f in use_feats]]
    rng = np.random.default_rng(seed + 1_000_000 * outer)
    for b in range(n_boot):
        idxb = rng.integers(0, Xtr_use.shape[0], Xtr_use.shape[0])
        xb = Xtr_use[idxb, :]
        yb = ytr[idxb]
        m = LinearSVR(**svr_kwargs)
        m.fit(xb, yb)
        for j, f in enumerate(use_feats):
            pooled_rows.append({"feature": f, "coef": float(m.coef_[j])})
pooled = pd.DataFrame(pooled_rows)
if pooled.empty:
    raise RuntimeError("No pooled bootstrap coefs created for Model 2C stable features.")
#Final table
final_df = (pooled.groupby("feature")["coef"]
            .agg(
                coef_mean="mean",
                ci_2_5=lambda x: np.quantile(x, 0.025),
                ci_97_5=lambda x: np.quantile(x, 0.975),
                n_draws="count"
            )
            .reset_index())
final_df = final_df.merge(coverage_df, on="feature", how="left")
final_df["abs_coef_mean"] = final_df["coef_mean"].abs()
#order by selection frequency (final_feats)
final_df["feature"] = pd.Categorical(final_df["feature"], categories=final_feats, ordered=True)
final_df = final_df.sort_values("feature").reset_index(drop=True)
out_csv  = os.path.join(out_dir, "model2c_bootstrap_FINAL_top_stable.csv")
out_xlsx = os.path.join(out_dir, "model2c_bootstrap_FINAL_top_stable.xlsx")
final_df.to_csv(out_csv, index=False)
final_df_rounded = final_df.copy()
for c in ["coef_mean", "ci_2_5", "ci_97_5", "abs_coef_mean", "pct_folds_in_top5"]:
    if c in final_df_rounded.columns:
        final_df_rounded[c] = final_df_rounded[c].astype(float).round(4)
final_df_rounded.to_excel(out_xlsx, index=False)
#Final plot
err_low  = final_df["coef_mean"] - final_df["ci_2_5"]
err_high = final_df["ci_97_5"] - final_df["coef_mean"]
plt.figure(figsize=(7, 5))
plt.barh(final_df["feature"].astype(str), final_df["coef_mean"], xerr=[err_low, err_high], capsize=3)
plt.axvline(0, color="k", linewidth=1)
plt.gca().invert_yaxis()
plt.xlabel("Coefficient (preprocessed scale)")
plt.title("Model 2c – pooled bootstrap 95% CI\n(Top stable features across folds)")
plt.tight_layout()
out_png = os.path.join(img_dir, "model2c_bootstrap_ci_FINAL_top_stable.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%---------------------------------------------------------------------------------------------------------------
#Subquestion 3 To what extent does predictive performance differ across demographic subgroups (e.g., gender, age)
subq3_dir = p("subquestion3")
out_SVRfinal = os.path.join(subq3_dir, "final_linear_svr_rfe5")
out_2final = os.path.join(subq3_dir, "model2c")
os.makedirs(out_SVRfinal, exist_ok=True)
os.makedirs(out_2final, exist_ok=True)
#Helperfunction for the fairness computation
def rmse(y_t, y_p):
    return mean_squared_error(y_t, y_p) ** 0.5
metrics = {
    "R2": r2_score,
    "RMSE": rmse,
    "MAE": mean_absolute_error,}
def run_fairness_for_feature(fair_df, feature_col, name, prefix, out_dir, saveplots=True):
    mask = fair_df[feature_col].notna() & fair_df["y_true"].notna() & fair_df["y_pred"].notna()
    y_true = fair_df.loc[mask, "y_true"].astype(float)
    y_pred = fair_df.loc[mask, "y_pred"].astype(float)
    sensitive = fair_df.loc[mask, feature_col].astype("string")
    mf = MetricFrame(
        metrics=metrics, 
        y_true=y_true, 
        y_pred=y_pred, 
        sensitive_features=sensitive)
    byg = mf.by_group.sort_index()
    print(byg)
    print("Overall:")
    print(mf.overall)
    print(f"{name} ({prefix}) – differences (fairlearn):")
    print(mf.difference())
    print(f"{name} ({prefix}) – ratio (fairlearn):")
    print(mf.ratio())
    #Save table
    out_csv = os.path.join(out_dir, f"fairness_{prefix}_{name.replace(' ','_')}_by_group.csv")
    byg.to_csv(out_csv)
    if saveplots:
        ax = byg[["R2", "RMSE", "MAE"]].plot(
            kind="barh",
            subplots=True,
            layout=(1, 3),
            figsize=(12, 4),
            legend=False,
            title=[
                f"R² by {name} ({prefix})",
                f"RMSE by {name} ({prefix})",
                f"MAE by {name} ({prefix})",
            ],
        )
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"fairness_{prefix}_{name.replace(' ','_')}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
    return mf
#%%-----------------------------------------------------------------------------------------------------------------------------------
#Using fairlearn on the different demographic subgroups on the best model (Linear SVR + RFE(5))
svr_rfe5_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
pred_files = sorted(
    glob_module.glob(os.path.join(
        svr_rfe5_dir,
        "svr_linear_rfe5_rep??_outer??_preds.csv")))
all_rows = []
pattern = re.compile(r".*rep(\d+)_outer(\d+)_preds\.csv$")
for fp in pred_files:
    m = pattern.match(fp)
    if not m:
        continue
    rep = int(m.group(1))
    ofold = int(m.group(2))
    df_pred = pd.read_csv(fp)
    idx_path = os.path.join(svr_rfe5_dir, f"svr_linear_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy")
    test_idx = np.load(idx_path)
    if len(test_idx) != len(df_pred):
        raise ValueError(f"Length mismatch rep {rep}, fold {ofold}")
    df_pred["rep"] = rep
    df_pred["outer_fold"] = ofold
    df_pred["idx"] = test_idx
    all_rows.append(df_pred)
preds_all = pd.concat(all_rows, ignore_index=True)
print("Linear SVR fairness predictions loaded:", preds_all.shape)
#Build df_valid just as in the CV-runs
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = (
    df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()]
    .copy()
    .sort_values([GROUP_COL])
    .reset_index(drop=True))    
#Merge with demographics
meta_cols = ["Age", "geslacht_GegevensAfname"]
if "leefsituatie.1" in df_valid.columns:
    meta_cols.append("leefsituatie.1")
if "opleiding_nieuw" in df_valid.columns:
    meta_cols.append("opleiding_nieuw")
meta_valid = df_valid[meta_cols].copy()
meta_valid["idx"] = meta_valid.index
fair_df = preds_all.merge(meta_valid, on="idx", how="left")
print("Missing meta after merge:",
      fair_df["Age"].isna().mean(),
      fair_df["geslacht_GegevensAfname"].isna().mean())
fair_df = fair_df.dropna(subset=["y_true", "y_pred", "Age", "geslacht_GegevensAfname"])
#Age groups
fair_df["age_group"] = pd.cut(
    fair_df["Age"],
    bins=[17, 25, 55, 80],
    labels=["<25", "25–55", ">55"],
    include_lowest=True,)
#Sex labels
geslacht_map = {1: "Man", 2: "Vrouw", 3: "Onbepaald"}
fair_df["geslacht_label"] = fair_df["geslacht_GegevensAfname"].map(geslacht_map)
#Living situation labels
if "leefsituatie.1" in fair_df.columns:
    leef_map = {
        1: "Zelfstandig",
        2: "Samenwonend",
        3: "Onzelfstandig",
        4: "Dakloos/anders",}
    fair_df["leefsituatie_label"] = fair_df["leefsituatie.1"].map(leef_map)
#Education labels
if "opleiding_nieuw" in fair_df.columns:
    fair_df["opleiding_label"] = fair_df["opleiding_nieuw"].astype("Int64").astype("string")
#Fairness runs
mf_gender = run_fairness_for_feature(
    fair_df, "geslacht_label", "gender", "LinearSVR", out_SVRfinal)
mf_age = run_fairness_for_feature(
    fair_df, "age_group", "age group", "LinearSVR", out_SVRfinal)
if "leefsituatie_label" in fair_df.columns:
    mf_leef = run_fairness_for_feature(
        fair_df, "leefsituatie_label", "living situation", "LinearSVR", out_SVRfinal)
if "opleiding_label" in fair_df.columns:
    mf_opleiding = run_fairness_for_feature(
        fair_df, "opleiding_label", "education level", "LinearSVR", out_SVRfinal)
#%%-------------------------------------------------------------------------------------------
#Get subgroup counts
def subgroup_counts(fair_df, col):
    tmp = fair_df.dropna(subset=[col, "idx"]).copy()
    out = (
        tmp.groupby(col)
        .agg(
            n_patients=("idx", "nunique"),   
            n_rows=("idx", "size"),          )
        .sort_values("n_patients", ascending=True))
    return out
print(subgroup_counts(fair_df, "geslacht_label"))
print(subgroup_counts(fair_df, "age_group"))
print(subgroup_counts(fair_df, "leefsituatie_label"))
print(subgroup_counts(fair_df, "opleiding_label"))
#%%-------------------------------------------------------------------------------------------
#Fairness for Model 2C (SVR + RFE5, Model 2C feature set)
model2_dir = p("subquestion2", "model2c") 
pred_files = sorted(glob_module.glob(os.path.join(
    model2_dir, "Model2c_rep??_outer??_preds.csv")))
if len(pred_files) == 0:
    raise FileNotFoundError(f"No Model 2C pred files found in {model2_dir}")
all_rows = []
pattern = re.compile(r".*rep(\d+)_outer(\d+)_preds\.csv$")
for fp in pred_files:
    m = pattern.match(fp)
    if not m:
        continue
    rep = int(m.group(1))
    ofold = int(m.group(2))
    df_pred = pd.read_csv(fp)
    idx_path = os.path.join(model2_dir, f"svr_subquestion2c_linear_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy")
    test_idx = np.load(idx_path)
    if len(test_idx) != len(df_pred):
        raise ValueError(f"Mismatch rep{rep:02d} fold{ofold:02d}: idx={len(test_idx)} preds={len(df_pred)}")
    df_pred["rep"] = rep
    df_pred["outer_fold"] = ofold
    df_pred["idx"] = test_idx
    all_rows.append(df_pred)
preds2c = pd.concat(all_rows, ignore_index=True)
print("Model 2C fairness predictions loaded:", preds2c.shape)
#Build the same df_valid
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
df_valid = (df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()]
            .copy()
            .sort_values([GROUP_COL])
            .reset_index(drop=True))
#Merge with demographics
meta_cols = ["Age", "geslacht_GegevensAfname"]
if "leefsituatie.1" in df_valid.columns:
    meta_cols.append("leefsituatie.1")
if "opleiding_nieuw" in df_valid.columns:
    meta_cols.append("opleiding_nieuw")
meta_valid = df_valid[meta_cols].copy()
meta_valid["idx"] = meta_valid.index
fair2c = preds2c.merge(meta_valid, on="idx", how="left")
print("Missing meta after merge (2C):",
      fair2c["Age"].isna().mean(),
      fair2c["geslacht_GegevensAfname"].isna().mean())
fair2c = fair2c.dropna(subset=["y_true", "y_pred", "Age", "geslacht_GegevensAfname"])
#Age-groepen
fair2c["age_group"] = pd.cut(fair2c["Age"], bins=[17, 25, 55, 80],
                            labels=["<25", "25–55", ">55"], include_lowest=True)
#Sex map
geslacht_map = {1: "Man", 2: "Vrouw", 3: "Onbepaald"}
fair2c["geslacht_label"] = fair2c["geslacht_GegevensAfname"].map(geslacht_map)
#Living situation
if "leefsituatie.1" in fair2c.columns:
    leef_map = {1:"Zelfstandig",2:"Samenwonend met derden",3:"Onzelfstandig",4:"Dakloos/anders"}
    fair2c["leefsituatie_label"] = fair2c["leefsituatie.1"].map(leef_map)
#Education map
if "opleiding_nieuw" in fair2c.columns:
    fair2c["opleiding_label"] = fair2c["opleiding_nieuw"].astype("Int64").astype("string")
#Fairness runs
mf_gender_2c = run_fairness_for_feature(fair2c, "geslacht_label", "gender", "Model2C", out_2final)
mf_age_2c  = run_fairness_for_feature(fair2c, "age_group", "age group", "Model2C", out_2final)
if "leefsituatie_label" in fair2c.columns:
    mf_leef_2c = run_fairness_for_feature(fair2c, "leefsituatie_label", "living situation", "Model2C", out_2final)
if "opleiding_label" in fair2c.columns:
    mf_edu_2c = run_fairness_for_feature(fair2c, "opleiding_label", "education level", "Model2C", out_2final)
#%%-------------------------------------------------------------------------------------------------------------------------------------------
#Subquestion 4: To what extent does the best-performing model generalize to patients withfollow-up intervals outside the 9–15 month range?
#Making a seperate folder with the results for subquestion 4
subq4_dir = p("subquestion4")
os.makedirs(subq4_dir, exist_ok=True)
#%%---------------------------------------------------------------------------
#Settings + load hyperparams + load FINAL top-5 features from endmodel
ycol      = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
svr_dir   = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
best_xlsx = os.path.join(svr_dir, "svr_linear_rfe5_repeated_nestedcv_results.xlsx")
top5_csv  = os.path.join(svr_dir, "svr_rfe5_top5_overall.csv")
FEATURES = features_for_nested_cv  
def rmse(y_t, y_p):
    return mean_squared_error(y_t, y_p) ** 0.5
#Load best hyperparameters (mode over folds/reps)
best_df = pd.read_excel(best_xlsx, sheet_name="best_params")
mode_params = best_df.mode(numeric_only=False).iloc[0].to_dict()
svr_kwargs = {k.replace("svr__", ""): v for k, v in mode_params.items() if k.startswith("svr__")}
svr_kwargs["random_state"] = 42
svr_kwargs["max_iter"] = 20000
print("Subq4 — Final SVR params:", svr_kwargs)
#Rebuild df_valid exactly as in CV 
df_valid = (
    df_mansa_9_15.loc[df_mansa_9_15[ycol].notna()]
    .copy()
    .sort_values([GROUP_COL])
    .reset_index(drop=True))
#Training set = 9–15 only
X_train_raw = df_valid[[c for c in FEATURES if c in df_valid.columns]].copy()
y_train     = df_valid[ycol].astype(float).to_numpy()
#HARD sanitize TRAIN
X_train_raw = X_train_raw.copy()
X_train_raw = X_train_raw.astype("object").where(pd.notna(X_train_raw), np.nan)
num_like = [c for c in X_train_raw.columns
            if c not in categorical_cols_global + binary_cols_global]
for c in num_like:
    X_train_raw[c] = pd.to_numeric(X_train_raw[c], errors="coerce")
#Drop >50% missingness based on TRAIN only
keep_cols, drop_cols = drop_high_missing(X_train_raw, threshold=0.50)
X_train_raw = X_train_raw[keep_cols].copy()
categorical_cols = [c for c in categorical_cols_global if c in keep_cols]
binary_cols      = [c for c in binary_cols_global if c in keep_cols]
#Fit preprocessor on TRAIN only
preproc = build_preprocessor_minmax(
    df=X_train_raw,
    features=keep_cols,
    categorical_cols=categorical_cols,
    binary_cols=binary_cols,
    random_state=svr_kwargs["random_state"],)
X_train_proc = preproc.fit_transform(X_train_raw, y_train)
feat_names   = preproc.get_feature_names_out()
X_train = pd.DataFrame(X_train_proc, columns=feat_names)
#Load FINAL top-5 features 
top5_df = pd.read_csv(top5_csv)
if "feature" in top5_df.columns:
    selected_feats = top5_df["feature"].head(5).tolist()
else:
    selected_feats = top5_df.iloc[:, 0].head(5).tolist()
missing = [f for f in selected_feats if f not in X_train.columns]
if missing:
    raise ValueError(f"Selected top-5 not found in preprocessed design: {missing}")
print("Subq4 — using FINAL top-5 features:", selected_feats)
X_train_sel = X_train[selected_feats].to_numpy()
#Fit final Linear SVR on 9–15
svr_final = LinearSVR(**svr_kwargs)
svr_final.fit(X_train_sel, y_train)
#Out-of-range test set (outside 9–15)
df_out = df_final[
    (df_final["mansa_totaal.1"].notna()) &
    (df_final["mansa_totaal.2"].notna()) &
    (df_final["month_diff_1_and_2"].notna()) &
    (~df_final["month_diff_1_and_2"].between(9, 15, inclusive="both"))
].copy()
print("Out-of-range test set shape:", df_out.shape)
X_test_raw = df_out.reindex(columns=keep_cols).copy()   
#HARD sanitize TEST
X_test_raw = X_test_raw.copy()
X_test_raw = X_test_raw.astype("object").where(pd.notna(X_test_raw), np.nan)
num_like = [c for c in X_test_raw.columns if c not in categorical_cols + binary_cols]
for c in num_like:
    X_test_raw[c] = pd.to_numeric(X_test_raw[c], errors="coerce")
y_test     = df_out[ycol].astype(float).to_numpy()
#Transform with train-fitted preprocessor
X_test_proc = preproc.transform(X_test_raw)
X_test      = pd.DataFrame(X_test_proc, columns=feat_names, index=df_out.index)
#Select same final top-5
X_test_sel = X_test[selected_feats].to_numpy()
#Predict + metrics
y_pred = svr_final.predict(X_test_sel)
r2_out   = r2_score(y_test, y_pred)
rmse_out = rmse(y_test, y_pred)
mae_out  = mean_absolute_error(y_test, y_pred)
print("Subq4 — Out-of-range performance (FINAL LinearSVR+RFE5)")
print(f"R²   = {r2_out:.3f}")
print(f"RMSE = {rmse_out:.3f}")
print(f"MAE  = {mae_out:.3f}")
#Save predictions/design
out_df = X_test.copy()
out_df["y_true"] = y_test
out_df["y_pred"] = y_pred
out_df["month_diff_1_and_2"] = df_out["month_diff_1_and_2"].values
full_csv  = os.path.join(subq4_dir, "finalSVR_out_of_range_design_FULL.csv")
full_xlsx = os.path.join(subq4_dir, "finalSVR_out_of_range_design_FULL.xlsx")
out_df.to_csv(full_csv, index_label="index")
with pd.ExcelWriter(full_xlsx, engine="openpyxl") as w:
    out_df.to_excel(w, sheet_name="out_of_range_full", index=True)
#Scatter observed vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="black", linewidth=0.5)
lo = float(min(y_test.min(), y_pred.min()))
hi = float(max(y_test.max(), y_pred.max()))
plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
plt.xlabel("Observed MANSA_totaal.2 (out-of-range)")
plt.ylabel("Predicted MANSA_totaal.2")
plt.title("FINAL LinearSVR+RFE5 — predictions outside 9–15 months")
plt.grid(alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(subq4_dir, "finalSVR_out_of_range_actual_vs_predicted.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%---------------------------------------------------------------------------
#Short (<9) vs Long (>15) within out-of-range
df_short = out_df[out_df["month_diff_1_and_2"] < 9].copy()
df_long  = out_df[out_df["month_diff_1_and_2"] > 15].copy()
print("Out-of-range split sizes:")
print("SHORT (<9 months):", df_short.shape)
print("LONG  (>15 months):", df_long.shape)
for name, df_part in [("SHORT (<9)", df_short), ("LONG (>15)", df_long)]:
    y_t = df_part["y_true"].astype(float)
    y_p = df_part["y_pred"].astype(float)
    print(f"\n{name}")
    print("  R²   =", r2_score(y_t, y_p))
    print("  RMSE =", rmse(y_t, y_p))
    print("  MAE  =", mean_absolute_error(y_t, y_p))
#Residual boxplot
box_df = pd.DataFrame({
    "residual": pd.concat([
        (df_short["y_pred"] - df_short["y_true"]),
        (df_long["y_pred"]  - df_long["y_true"])
    ], ignore_index=True),
    "group": (["<9 months"] * len(df_short)) + ([">15 months"] * len(df_long))
})
plt.figure(figsize=(10, 5))
sns.boxplot(data=box_df, x="group", y="residual")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("FINAL LinearSVR+RFE5 — residuals short vs long follow-up")
plt.xlabel("Follow-up group")
plt.ylabel("Residual (predicted − observed)")
plt.tight_layout()
fig_box = os.path.join(subq4_dir, "finalSVR_residual_boxplots_short_vs_long.png")
plt.savefig(fig_box, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------------------------------