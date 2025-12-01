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
#%%-------------------------------------------------------------------------------------------------------------
#Making a new "opleiding_nieuw" variabele with the answers from opleiding.1 for better alignment and making it numeric for the one-hot-encoding
df_final = df_final.copy()
df_final["opleiding_nieuw"] = (
    pd.to_numeric(df_final["opleiding.1"], errors="coerce")
    .astype("Int64"))
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
    mansa_total = mansa_mean * 12
    mansa_total[mansa_missing > 2] = np.nan
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
    honos_total = honos_mean * 12
    honos_total[honos_missing > 2] = np.nan
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
#A scatterplot with the 2 timepoints. The first timepoint decides the second I think. How much do the scores change between the first and second timepoint?
subset = df_final[["mansa_totaal.1", "mansa_totaal.2", "month_diff_1_and_2"]].dropna()
x = subset["mansa_totaal.1"]
y = subset["mansa_totaal.2"]
r, p_val = pearsonr(x, y)
plt.figure(figsize=(6,6))
sc = plt.scatter(x, y, c=subset["month_diff_1_and_2"], cmap="viridis",
                 alpha=0.7, edgecolor="black", linewidth=0.3)
lims = [12, 84]
plt.plot(lims, lims, linestyle="--", color="black", linewidth=1)  # y = x line
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
plt.title("Distribution of time between timepoint 1 and 2 (MANSA pairs only)", fontsize=14, fontweight="bold")
plt.xlabel("Months"); plt.ylabel("Count")
plt.xticks(np.arange(lo, hi + 1, 1), rotation=45, ha="right")
plt.grid(axis="y", color="gray", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "months_T1_T2_hist_mansa_pairs.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------------------------------------
#Working with the first smaller subset (between 9 and 15 months)
#Making the first subset (only people with the second measurement between 9 and 15 months)
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
#Distribution of timepoint 2 with smaller subset
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
plt.plot(lims, lims, linestyle="--", color="black", linewidth=1)  # y = x line
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
#The results of the Shapiro test show that the residuals deviate from normality (p < 0.05).
#The results also show a W-statistic of 0.982 which means that the data is close to a normal distribution.
#The skewness (symmetry of a variable's distribution) is -0.110, the negative part means more larger values to the left. However, skewness between -1 and +1 is excellent and this is the case.
#The kurtosis (distribution too flat or peaked compared to normal distribution) the kurtosis is 1.328 which means it is not normal, but not too bad. It means that er meer piek is en zwaardere staarten dan normaal
#This means that the standard errors (p-values and confidence intervals) from the models might be a bit too optimistic.The other evaluation metrics stay the same.
#%%--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Handling the results with adding a robust SE model
subset = df_mansa_9_15[["mansa_totaal.1", "mansa_totaal.2"]].dropna().copy()
model_sm_robust = smf.ols(
    "Q('mansa_totaal.2') ~ Q('mansa_totaal.1')",
    data=subset).fit(cov_type="HC3")
print(model_sm_robust.summary())
#%%-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Checking for outliers just for intepreting (OLS outliers and influence diagnostics) --> Not going to do anything about them 
#Extreme values are investigated but not excluded. Scaling reduces their impact for distance-based and linear models.
#Influence must always be computed on the non-robust OLS model (not the HC3 version)
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
#Outlier diagnostics (Cook’s D, leverage) run separately on baseline OLS; model proved robust (<5% slope change, was 4.99%)
#%%----------------------------------------------------------------------------------------------------------------------------
#Fit model with all data to get the slope change when removing the outliers
model_full = model_sm  # your original fitted OLS model
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
#Making the possible predictor lists:
context_vars = [
    "Age","geslacht_GegevensAfname","geboortemandsocio","Leeftijd1eGGZ_a.1","Leeftijd1ePsyKl_a.1","modusmeanGGZ","modusmeanPsyKl","burgerlijkestaat.1",
    "leefsituatie.1","leefsituatie_steun.1","levenspartner.1","opleiding_nieuw","month_diff_1_and_2","dagbest_betaald.1","dagbest_opleiding.1","dagbest_dagact.1",
    "dagbest_vrijwillig.1","dagbest_huishouden.1","dagbest_overig.1","dagbest_geen.1","Betrouwbaarheid_1_.1","vrijwilligerswerk_2_.1","HerstelHV_4_10_.1"]
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
#Checking what percentage of each variable is missing
missing_df = (df_mansa_9_15[all_predictors].isna().mean() * 100).reset_index()
missing_df.columns = ["variable", "missing_percent"]
missing_df = missing_df.sort_values("missing_percent", ascending=False)
missing_df.to_csv(p("missing_values_all_features.csv"), index=False)
missing_df.to_excel(p("missing_values_all_features.xlsx"),
                    index=False, header=True, sheet_name="Missing values")
#%%-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Visualizing missings
cols_for_missing = [c for c in all_predictors if c in df_mansa_9_15.columns]
if len(cols_for_missing) == 0:
    print("No columns present right now to visualize.")
else:
    plt.figure(figsize=(14, 6))
    msno.matrix(df_mansa_9_15[cols_for_missing], sparkline=False, labels=True)
    plt.title("Missing values in analysis subset (df_mansa_9_15)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir_1, "missing_matrix_df_mansa_9_15.png"),
                dpi=300)
    plt.close()
    if len(cols_for_missing) > 1:
        plt.figure(figsize=(10, 6))
        msno.heatmap(df_mansa_9_15[cols_for_missing])
        plt.title("Correlation of missingness (df_mansa_9_15)")
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir_1, "missing_heatmap_df_mansa_9_15.png"),
                    dpi=300)
        plt.close()
    topN = 30
    top = (df_mansa_9_15[cols_for_missing]
           .isna().mean() * 100).sort_values(ascending=False).head(topN)
    plt.figure(figsize=(8, 10))
    plt.barh(top.index[::-1], top.values[::-1])
    plt.xlabel("Missing (%)")
    plt.ylabel("Variable")
    plt.title(f"Top {min(topN, len(cols_for_missing))} missing variables (df_mansa_9_15)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir_1, "missing_bar_top_df_mansa_9_15.png"),
                dpi=300)
    plt.show()
#%%---------------------------------------------------------------------------------------
#Dropping the columns with >50% missings (= "HerstelHV_4_10_.1" &"vrijwilligerswerk_2_.1")
cols_to_drop = missing_df.loc[missing_df["missing_percent"] > 50, "variable"]
df_mansa_9_15 = df_mansa_9_15.drop(columns=list(cols_to_drop), errors="ignore")
#Update the predictor list
existing_predictors = [c for c in all_predictors if c in df_mansa_9_15.columns]
all_predictors = existing_predictors
context_vars = [c for c in context_vars if c in df_mansa_9_15.columns]
print(context_vars)
print(all_predictors)
#%%-----------------------------------------------------------------------------------------------
#Making the smaller correct subset
df_mansa_9_15.to_csv(p("subset_9_15.csv"), index=False)
df_mansa_9_15.to_excel(p("subset_9_15.xlsx"),
                       index=False, header=True,
                       sheet_name="Subset 9-15 months")
#%%------------------------------------------------------------------------------------------------
#Leak-free baselines map
models_dir = p("models")
os.makedirs(models_dir, exist_ok=True)
#%%---------------------------------------------------------------------------------------------------------------------------
#Remove inkomsten columns (high multicollinearity with betaaldwerk etc.)
inkomsten_cols = [c for c in df_mansa_9_15.columns if c.startswith("inkomsten_")]
df_mansa_9_15 = df_mansa_9_15.drop(columns=inkomsten_cols, errors="ignore")
print(f"Removed {len(inkomsten_cols)} income-related columns due to multicollinearity.")
#%%---------------------------------------------------------------------------------------------------
#Goal variable + indices
ycol = "mansa_totaal.2"
valid_idx = df_mansa_9_15.index[df_mansa_9_15[ycol].notna()]
y_all = df_mansa_9_15.loc[valid_idx, ycol].astype(float)
#%%---------------------------------------------------------------------------------------------------------------
#One train/test split for all the linear regression feature chosen models
train_idx, test_idx = train_test_split(valid_idx, test_size=0.2, random_state=42)
print(f"Split sizes → train={len(train_idx)}, test={len(test_idx)}")
#%%----------------------------------------------------------------------------------------------------------------
#Feature lists
demographic_vars = [
    "Age",
    "geslacht_GegevensAfname", "geboortemandsocio",
    "modusmeanGGZ", "modusmeanPsyKl",
    "burgerlijkestaat.1", "leefsituatie.1", "leefsituatie_steun.1",
    "levenspartner.1", "opleiding_nieuw", "month_diff_1_and_2",
    "dagbest_betaald.1", "dagbest_opleiding.1", "dagbest_dagact.1",
    "dagbest_vrijwillig.1", "dagbest_huishouden.1", "dagbest_overig.1",
    "dagbest_geen.1", "vrijwilligerswerk.1",
    "Leeftijd1eGGZ_a.1", "Leeftijd1ePsyKl_a.1",]
questionnaire_totals_regression_only = ["Inspire_totaal_1", "FR_totaal.1", "honos_totaal.1"]
binary_questionnaires = ["Inspire_binair.1", "FR_binair.1"]
separate_questionnaires = inspire_items + fr_items + mansa_items + honos_items_1
#%%------------------------------------------------------------------------------------------------------------------
#Global categorical + binary columns (for one-hot-encoder)
categorical_cols_global = ["geslacht_GegevensAfname","geboortemandsocio","leefsituatie.1","opleiding_nieuw",]
binary_cols_global = [
    "burgerlijkestaat.1","leefsituatie_steun.1","levenspartner.1",
    "dagbest_betaald.1","dagbest_opleiding.1","dagbest_dagact.1","dagbest_vrijwillig.1",
    "dagbest_huishouden.1","dagbest_overig.1","dagbest_geen.1","vrijwilligerswerk.1",
    "Inspire_binair.1","FR_binair.1",
    "Leeftijd1eGGZ_a.1", "Leeftijd1ePsyKl_a.1",]
for c in binary_cols_global:
    if c in df_mansa_9_15.columns:
        df_mansa_9_15[c] = pd.to_numeric(df_mansa_9_15[c], errors="coerce").astype(float)
#%%----------------------------------------------------------------------------------------------
#Helper: ColumnTransformer (MICE voor numeric, most_frequent voor binary, one-hot voor categoricals)
def build_preprocessor(features: list[str]) -> ColumnTransformer:
    feats_exist = [c for c in features if c in df_mansa_9_15.columns]
    if len(feats_exist) == 0:
        raise ValueError("No requested features found in dataframe.")
    #The different categories of data's
    cat_cols = [c for c in categorical_cols_global if c in feats_exist]
    bin_cols = [c for c in binary_cols_global if c in feats_exist]
    num_cols = [c for c in feats_exist if c not in cat_cols + bin_cols]
    #Printen what is happening
    print(f"\nBuilding preprocessor for features: {feats_exist}")
    print(f"  Numeric: {num_cols}")
    print(f"  Binary:  {bin_cols}")
    print(f"  Categorical (one-hot): {cat_cols}")
    #Applying columntransformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", IterativeImputer(random_state=42, max_iter=50, sample_posterior=True)),
        ("scaler", StandardScaler()),])
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),]) #No scaler → 0/1 stays 0/1
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),])
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
    feats_exist = [c for c in features if c in df_mansa_9_15.columns]
    X = df_mansa_9_15.loc[valid_idx, feats_exist].copy()
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
X_dummy = df_mansa_9_15.loc[valid_idx, ["mansa_totaal.1"]].copy()
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
model1_feats = demographic_vars
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
model2_feats = demographic_vars + ["mansa_totaal.1"]
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
model3_feats = demographic_vars + ["mansa_totaal.1"] + questionnaire_totals_regression_only
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
model4_feats = demographic_vars + ["mansa_totaal.1"] + binary_questionnaires
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
print(f"  ΔR²   = {r2_reg_4 - r2_reg_2:.3f}")
print(f"  ΔRMSE = {rmse_reg_3 - rmse_reg_4:.3f}")
print(f"  ΔMAE  = {mae_reg_3 - mae_reg_4:.3f}")
#%%---------------------------------------------------------------------------------------------------------------------------
#Model 5: Model 2 + all questionnaire items (Inspire, FR, MANSA, HoNOS)
model5_feats = demographic_vars + ["mansa_totaal.1"] + separate_questionnaires
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
    demographic_vars
    + ["mansa_totaal.1"]
    + questionnaire_totals_regression_only   # Inspire_totaal_1, FR_totaal.1, honos_totaal.1
    + binary_questionnaires)                 # Inspire_binair.1, FR_binair.1
print("\nRunning Model 6...")
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
print("\n=== Comparison of Models 1–6 (leak-free) ===")
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
#OLS is in-sample baseline, de rest out-of-sample; niet direct 1-op-1 vergelijken.
#%%------------------------------------------------------------------------------------------------------------
#The third model is the best for our model and intention! So we go on with the third model from now on!
#Coefficient analysis for model 3
print("\nCoefficient importance for Model 3")
#Refit model 3 so we can extract coefficients properly
pipe3 = Pipeline(steps=[
    ("preprocess", build_preprocessor(model3_feats)),
    ("reg", LinearRegression())])
X3 = df_mansa_9_15.loc[valid_idx, model3_feats].copy()
y3 = y_all
pipe3.fit(X3.loc[train_idx], y3.loc[train_idx])
#Extract transformed column names
pre3 = pipe3.named_steps["preprocess"]
feature_names_3 = pre3.get_feature_names_out()
#Extract coefficients
coefs_3 = pipe3.named_steps["reg"].coef_
coef_df = pd.DataFrame({
    "feature": feature_names_3,
    "coef": coefs_3
}).sort_values("coef", key=abs, ascending=False)
coef_df.to_excel(os.path.join(models_dir, "model3_coefficients.xlsx"), index=False)
print(coef_df.head(20).round(3))
#Visualize
plt.figure(figsize=(8,10))
coef_df.head(20).sort_values("coef").plot(
    x="feature", y="coef", kind="barh", figsize=(8,10))
plt.title("Model 3 – Standardized coefficients (Top 20)")
plt.tight_layout()
plt.savefig(os.path.join(img_dir_1, "model3_coefplot_top20.png"), dpi=300)
plt.show()
plt.close()
#%%-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Check for multicollinearity (VIF)
#VIF op train (drop zero-variance kolommen en voeg constant toe)
#This block make us remove the "inkomsten_" and betaaldwerk.1 columns due to high multicolinearity with "betaaldwerk.1" en "dagbest_betaald.1" earlier in this code.
#Now we are checking it again on model 3 :)
#Recreate model3_X_train (raw input features before ColumnTransformer for VIF and they other stuff)
model3_feats = demographic_vars + ["mansa_totaal.1"] + questionnaire_totals_regression_only
model3_X_train = df_mansa_9_15.loc[train_idx, model3_feats].copy()
#VIF op train (drop zero-variance kolommen, drop NaNs, voeg constant toe)
Xv = model3_X_train.copy().apply(pd.to_numeric, errors="coerce")
#kolommen met maar 1 unieke waarde eruit
Xv = Xv.loc[:, Xv.nunique(dropna=True) > 1]
#rijen met NaN weg, anders geeft statsmodels MissingDataError
Xv = Xv.dropna(axis=0)
#constant toevoegen
Xv_const = sm.add_constant(Xv, has_constant="add")
#naar numpy (float) voor variance_inflation_factor
arr = Xv_const.to_numpy(dtype=float, copy=False)
vifs = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
vif3 = (
    pd.DataFrame({"feature": Xv_const.columns, "VIF": vifs})
    .query("feature!='const'")
    .sort_values("VIF", ascending=False))
vif3.to_excel(os.path.join(models_dir, "model3_vif_train.xlsx"), index=False)
print(vif3.head(20).round(2))
#%%--------------------------------------------------------------------------------------------------------------------
#I want to check for outliers in the third model = the model I am going to use for further modelling
X_m3_train = model3_X_train.copy() 
#Scale the data with MinMax
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_m3_train)
#Isolationforest
iso = IsolationForest(contamination=0.05, random_state=42)
labels = iso.fit_predict(X_scaled)        # -1 = outlier, 1 = inlier
#Attach flags and make a summary
flag = np.where(labels == -1, "Outlier", "Inlier")
print("\nIsolationForest (Model 3, TRAIN set only)")
unique, counts = np.unique(flag, return_counts=True)
print(dict(zip(unique, counts)))
print(f"≈ {(flag == 'Outlier').mean()*100:.1f}% flagged as outliers in the training fold")
#Save flagged original row indices
outlier_idx_train = X_m3_train.index[labels == -1]
out_path = os.path.join(models_dir, "model3_outlier_indices_train_only.csv")
pd.Series(outlier_idx_train, name="row_index").to_csv(out_path, index=False)
print(f"Saved train outlier indices to: {out_path}")
#Visualization
if {"mansa_totaal.1", "honos_totaal.1"}.issubset(df_mansa_9_15.columns):
    idx = X_m3_train.index
    plt.figure(figsize=(6,4))
    plt.scatter(
        df_mansa_9_15.loc[idx, "mansa_totaal.1"],
        df_mansa_9_15.loc[idx, "honos_totaal.1"],
        c=np.where(flag=="Outlier","red","blue"), alpha=0.7)
    plt.xlabel("MANSA T1"); plt.ylabel("HoNOS T1")
    plt.title("IsolationForest – Global Outliers (Model 3, train fold)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir_1, "Outlier detection model 3"), dpi=300, bbox_inches="tight")
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
#%%--------------------------------------------
#Helper: Columntransformer with MinMax-Scaling
def build_preprocessor_minmax(features: list[str], random_state: int) -> ColumnTransformer:
    feats_exist = [c for c in features if c in df_mansa_9_15.columns]
    if len(feats_exist) == 0:
        raise ValueError("No requested features found in dataframe.")
    cat_cols = [c for c in categorical_cols_global if c in feats_exist]
    bin_cols = [c for c in binary_cols_global if c in feats_exist]
    num_cols = [c for c in feats_exist if c not in cat_cols + bin_cols]
    print(f"\n[build_preprocessor_minmax] Using features: {feats_exist}")
    print(f"  Numeric: {num_cols}")
    print(f"  Binary:  {bin_cols}")
    print(f"  Categorical: {cat_cols}")
    numeric_transformer = Pipeline(steps=[
        ("imputer", IterativeImputer(
            random_state=random_state,
            max_iter=50,
            sample_posterior=True)),
        ("scaler", MinMaxScaler()),])
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("bin", binary_transformer, bin_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",)
    return preprocessor
#%%-----------------------------------
#Global setting for all CV models
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
N_OUTER_SPLITS = 10   
N_INNER_SPLITS = 5     
GROUP_COL = "Proefpersoonnummer"
ycol = "mansa_totaal.2"
valid_idx = df_mansa_9_15.index[df_mansa_9_15[ycol].notna()].to_numpy()
groups_all = df_mansa_9_15.loc[valid_idx, GROUP_COL].to_numpy()
def repeated_groupkfold_splits(
    valid_idx: np.ndarray,
    groups_all: np.ndarray,
    n_splits: int = N_OUTER_SPLITS,):
    unique_groups = np.unique(groups_all)
    n_repeats = len(SEEDS)
    for rep_idx in range(n_repeats):
        seed = SEEDS[rep_idx]
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_groups)
        rank = {g: i for i, g in enumerate(shuffled)}
        order = np.argsort([rank[g] for g in groups_all])
        vidx = valid_idx[order]
        garr = groups_all[order]
        cv = GroupKFold(n_splits=n_splits)
        for fold, (tr, te) in enumerate(cv.split(vidx, groups=garr), start=1):
            yield rep_idx + 1, fold, vidx[tr], vidx[te]
#%%--------------------------------------------------------------------------------------------------------
#Design matrix snapchat (for insight)
cross_validation_dir = p("cross_validation_results")
os.makedirs(cross_validation_dir, exist_ok=True)
ycol = "mansa_totaal.2"
features = demographic_vars + ["mansa_totaal.1"] + questionnaire_totals_regression_only
model3_feats = features.copy()  
mask = df_mansa_9_15[ycol].notna()
X_raw = df_mansa_9_15.loc[mask, model3_feats].copy()
y_all = df_mansa_9_15.loc[mask, ycol].astype(float)
preproc_snapshot = build_preprocessor_minmax(model3_feats, random_state=42)
X_proc = preproc_snapshot.fit_transform(X_raw, y_all)
try:
    feat_names = preproc_snapshot.get_feature_names_out()
except Exception:
    feat_names = [f"feat_{i}" for i in range(X_proc.shape[1])]
X_design = pd.DataFrame(X_proc, index=X_raw.index, columns=feat_names)
design_dir = cross_validation_dir
excel_path = os.path.join(design_dir, "Model3_design_after_MinMax_preprocessing.xlsx")
X_design.to_excel(excel_path, index=True, sheet_name="design_matrix")
print(f"\nSaved Model 3 design snapshot → {excel_path}")
#%%---------------------------------------------------------------------------------------------------
#XGBoost: repeated nested GroupKFold CV on Model 3 (10 seeds via SEEDS)
xgboost_cv_dir_2 = p("cross_validation_results", "xgboost_cv_results.2")
os.makedirs(xgboost_cv_dir_2, exist_ok=True)
#Doelvariabele en groepen
ycol = "mansa_totaal.2"
GROUP_COL = "Proefpersoonnummer"
valid_idx = df_mansa_9_15.index[df_mansa_9_15[ycol].notna()].to_numpy()
groups_all = df_mansa_9_15.loc[valid_idx, GROUP_COL].to_numpy()
N_OUTER_SPLITS = 10
N_INNER_SPLITS = 5
inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)
param_grid_xgb = {
    "xgb__n_estimators":     [200, 600],
    "xgb__learning_rate":    [0.03, 0.1],
    "xgb__max_depth":        [3, 6, 8],
    "xgb__min_child_weight": [1, 3],
    "xgb__subsample":        [0.8],
    "xgb__colsample_bytree": [0.8],
    "xgb__reg_lambda":       [0.0, 1.0],}
rows = []
best_params_list = []
importances_per_outer = []
SAVE_PREDS = True
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
    Xtr_raw = df_mansa_9_15.loc[outer_train_idx, model3_feats].copy()
    Xte_raw = df_mansa_9_15.loc[outer_test_idx,  model3_feats].copy()
    ytr     = df_mansa_9_15.loc[outer_train_idx, ycol].astype(float)
    yte     = df_mansa_9_15.loc[outer_test_idx,  ycol].astype(float)
    preproc = build_preprocessor_minmax(model3_feats, random_state=seed)
    Xtr_proc = preproc.fit_transform(Xtr_raw, ytr)
    Xte_proc = preproc.transform(Xte_raw)

    try:
        proc_feature_names = preproc.get_feature_names_out()
    except Exception:
        proc_feature_names = [f"feat_{i}" for i in range(Xtr_proc.shape[1])]

    Xtr = pd.DataFrame(Xtr_proc, index=outer_train_idx, columns=proc_feature_names)
    Xte = pd.DataFrame(Xte_proc, index=outer_test_idx,  columns=proc_feature_names)

    inner_groups = df_mansa_9_15.loc[outer_train_idx, GROUP_COL].to_numpy()

    pipe_xgb = Pipeline([
        ("xgb", XGBRegressor(
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0,
        ))
    ])

    gcv = GridSearchCV(
        estimator=pipe_xgb,
        param_grid=param_grid_xgb,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

    t0 = time.time()
    gcv.fit(Xtr, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_

    ypred = best_model.predict(Xte)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred) ** 0.5
    mae   = mean_absolute_error(yte, ypred)

    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2":   r2,
        "RMSE": rmse,
        "MAE":  mae,
    })
    best_params_list.append({
        "rep": rep,
        "fold": ofold,
        **gcv.best_params_,
    })

    xgb_step = best_model.named_steps["xgb"]
    importances_per_outer.append(
        pd.Series(xgb_step.feature_importances_, index=Xtr.columns)
          .sort_values(ascending=False)
          .rename(f"rep{rep:02d}_fold{ofold:02d}")
    )

    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"XGB_mm_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,
        )

    print(
        f"[XGB_mm rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
        f"seed={seed} | best={gcv.best_params_} | "
        f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}"
    )
    gc.collect()

res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2":   ["mean", "std"],
    "RMSE": ["mean", "std"],
    "MAE":  ["mean", "std"],
})

res_df.to_csv(os.path.join(xgboost_cv_dir_2, "xgb_mm_nestedcv_folds.csv"), index=False)
summ.to_csv(os.path.join(xgboost_cv_dir_2, "xgb_mm_nestedcv_summary.csv"))

if importances_per_outer:
    imp_df   = pd.concat(importances_per_outer, axis=1).fillna(0.0)
    imp_mean = imp_df.mean(axis=1).sort_values(ascending=False)
    imp_mean.to_csv(os.path.join(xgboost_cv_dir_2, "xgb_mm_feature_importances_mean.csv"))
else:
    imp_mean = pd.Series(dtype=float)

pd.DataFrame(best_params_list).to_csv(
    os.path.join(xgboost_cv_dir_2, "xgb_mm_bestparams_per_outer.csv"),
    index=False,
)

excel_path = os.path.join(xgboost_cv_dir_2, "xgb_mm_nestedcv_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ.to_excel(w, sheet_name="summary")
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    if not imp_mean.empty:
        imp_mean.rename("mean_importance").to_frame().to_excel(
            w, sheet_name="mean_importance"
        )

#%%
#kNN
kNN_cv_dir_2 = p("cross_validation_results", "kNN_cv_results.2")
os.makedirs(kNN_cv_dir_2, exist_ok=True)

param_grid_knn = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
    "knn__leaf_size": [20, 30, 40],
}

rows = []
best_params_list = []
SAVE_PREDS = True

inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)

pipe_knn = Pipeline([
    # geen extra scaler – features zijn al MinMax geschaald in preprocessor
    ("knn", KNeighborsRegressor())
])

for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx, groups_all, n_splits=N_OUTER_SPLITS):

    seed = SEEDS[rep - 1]
    out_dir = kNN_cv_dir_2

    np.save(os.path.join(out_dir, f"kNN_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"), outer_train_idx)
    np.save(os.path.join(out_dir, f"kNN_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),  outer_test_idx)

    Xtr_raw = df_mansa_9_15.loc[outer_train_idx, model3_feats].copy()
    Xte_raw = df_mansa_9_15.loc[outer_test_idx,  model3_feats].copy()
    ytr = df_mansa_9_15.loc[outer_train_idx, ycol].astype(float)
    yte = df_mansa_9_15.loc[outer_test_idx,  ycol].astype(float)

    preproc = build_preprocessor_minmax(model3_feats, random_state=seed)
    Xtr_proc = preproc.fit_transform(Xtr_raw, ytr)
    Xte_proc = preproc.transform(Xte_raw)

    try:
        feat_names = preproc.get_feature_names_out()
    except Exception:
        feat_names = [f"feat_{i}" for i in range(Xtr_proc.shape[1])]

    Xtr = pd.DataFrame(Xtr_proc, index=outer_train_idx, columns=feat_names)
    Xte = pd.DataFrame(Xte_proc, index=outer_test_idx,  columns=feat_names)

    inner_groups = df_mansa_9_15.loc[outer_train_idx, GROUP_COL].to_numpy()

    gcv = GridSearchCV(
        estimator=pipe_knn,
        param_grid=param_grid_knn,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    t0 = time.time()
    gcv.fit(Xtr, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_

    ypred = best_model.predict(Xte)
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
            index=False,
        )

    print(f"[kNN rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | best={gcv.best_params_} | R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()

res_df = pd.DataFrame(rows)
summ = res_df.agg({"R2": ["mean","std"], "RMSE": ["mean","std"], "MAE": ["mean","std"]})
res_df.to_csv(os.path.join(kNN_cv_dir_2, "knn_repeated_nestedcv_folds_2.csv"), index=False)
summ.to_csv(os.path.join(kNN_cv_dir_2, "knn_repeated_nestedcv_summary_2.csv"))
pd.DataFrame(best_params_list).to_csv(
    os.path.join(kNN_cv_dir_2, "knn_repeated_bestparams_per_outer_2.csv"),
    index=False,
)

excel_path = os.path.join(kNN_cv_dir_2, "knn_repeated_nestedcv_results_2.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ.to_excel(w, sheet_name="summary")
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)

print(f"\nSaved all kNN results → {excel_path}")
#%%---------------------------------------------------------------------------------------------------
# Random Forest: repeated nested GroupKFold CV on Model 3 (10 seeds via SEEDS)

random_forest_dir_2 = p("cross_validation_results", "random_forest.2")
os.makedirs(random_forest_dir_2, exist_ok=True)

param_grid_rf_m3 = {
    "rf__n_estimators":      [50, 100, 150],
    "rf__max_depth":         [None, 10, 20],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf":  [1, 2, 5],
    "rf__max_features":      ["sqrt", 0.5],
}

rows = []
best_params_list = []
importances_per_outer = []
SAVE_PREDS = True

inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)

for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):

    seed = SEEDS[rep - 1]
    out_dir = random_forest_dir_2

    # Save fold indices
    np.save(os.path.join(out_dir, f"RF_mm_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(out_dir, f"RF_mm_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)

    # Raw data
    Xtr_raw = df_mansa_9_15.loc[outer_train_idx, model3_feats].copy()
    Xte_raw = df_mansa_9_15.loc[outer_test_idx,  model3_feats].copy()
    ytr     = df_mansa_9_15.loc[outer_train_idx, ycol].astype(float)
    yte     = df_mansa_9_15.loc[outer_test_idx,  ycol].astype(float)

    # Leak-free preprocessing (MICE + OHE + MinMax), seed per repeat
    preproc = build_preprocessor_minmax(model3_feats, random_state=seed)
    Xtr_proc = preproc.fit_transform(Xtr_raw, ytr)
    Xte_proc = preproc.transform(Xte_raw)

    try:
        feat_names = preproc.get_feature_names_out()
    except Exception:
        feat_names = [f"feat_{i}" for i in range(Xtr_proc.shape[1])]

    Xtr = pd.DataFrame(Xtr_proc, index=outer_train_idx, columns=feat_names)
    Xte = pd.DataFrame(Xte_proc, index=outer_test_idx,  columns=feat_names)

    inner_groups = df_mansa_9_15.loc[outer_train_idx, GROUP_COL].to_numpy()

    # Random Forest pipeline (features al geschaald → geen extra scaler)
    pipe_rf = Pipeline([
        ("rf", RandomForestRegressor(
            random_state=seed,
            n_jobs=-1,
            oob_score=False)),
    ])

    # Inner-loop hyperparameter tuning
    gcv = GridSearchCV(
        estimator=pipe_rf,
        param_grid=param_grid_rf_m3,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

    t0 = time.time()
    gcv.fit(Xtr, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_

    # Outer-loop evaluation
    ypred = best_model.predict(Xte)
    r2    = r2_score(yte, ypred)
    rmse  = mean_squared_error(yte, ypred) ** 0.5
    mae   = mean_absolute_error(yte, ypred)

    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    })

    best_params_list.append({
        "rep": rep,
        "fold": ofold,
        **gcv.best_params_
    })

    # Feature importances per outer fold
    rf_step = best_model.named_steps["rf"]
    importances_per_outer.append(
        pd.Series(rf_step.feature_importances_, index=Xtr.columns)
          .sort_values(ascending=False)
          .rename(f"rep{rep:02d}_fold{ofold:02d}")
    )

    # Save predictions
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"RF_mm_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,
        )

    print(f"[RF_mm rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | best={gcv.best_params_} | "
          f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

    gc.collect()

# Aggregatie RF Model 3
res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2":   ["mean", "std"],
    "RMSE": ["mean", "std"],
    "MAE":  ["mean", "std"],
})

res_df.to_csv(os.path.join(random_forest_dir_2, "RF_mm_nestedcv_folds.csv"), index=False)
summ.to_csv(os.path.join(random_forest_dir_2, "RF_mm_nestedcv_summary.csv"))

# Gemiddelde feature importances over alle outer folds
if importances_per_outer:
    imp_df = pd.concat(importances_per_outer, axis=1).fillna(0.0)
    imp_mean = imp_df.mean(axis=1).sort_values(ascending=False)
    imp_mean.to_csv(os.path.join(random_forest_dir_2, "RF_mm_feature_importances_mean.csv"))
else:
    imp_mean = pd.Series(dtype=float)

pd.DataFrame(best_params_list).to_csv(
    os.path.join(random_forest_dir_2, "RF_mm_bestparams_per_outer.csv"),
    index=False,
)

excel_path = os.path.join(random_forest_dir_2, "RF_mm_nestedcv_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ.to_excel(w, sheet_name="summary")
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    if not imp_mean.empty:
        imp_mean.rename("mean_importance").to_frame().to_excel(
            w, sheet_name="mean_importance"
        )

print(f"\nSaved all RF_mm (Model 3) results → {excel_path}")
#%%---------------------------------------------------------------------------------------------------
# Random Forest – Model 2 (Demographics + MANSA_totaal.1)
# Nested GroupKFold CV with 10 seeds (SEEDS)

rf_model2_dir = p("cross_validation_results", "random_forest_model_2")
os.makedirs(rf_model2_dir, exist_ok=True)

model2_feats = demographic_vars + ["mansa_totaal.1"]

param_grid_rf_m2 = {
    "rf__n_estimators":      [50, 100, 150],
    "rf__max_depth":         [None, 10, 20],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf":  [1, 2, 5],
    "rf__max_features":      ["sqrt", 0.5],
}

rows = []
best_params_list = []
importances_per_outer = []
SAVE_PREDS = True

inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)

for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):

    seed = SEEDS[rep - 1]
    out_dir = rf_model2_dir

    # Save fold indices (for reproducibility)
    np.save(os.path.join(out_dir, f"RF_model2_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"), outer_train_idx)
    np.save(os.path.join(out_dir, f"RF_model2_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),  outer_test_idx)

    # Raw training/test samples
    Xtr_raw = df_mansa_9_15.loc[outer_train_idx, model2_feats].copy()
    Xte_raw = df_mansa_9_15.loc[outer_test_idx,  model2_feats].copy()
    ytr     = df_mansa_9_15.loc[outer_train_idx, ycol].astype(float)
    yte     = df_mansa_9_15.loc[outer_test_idx,  ycol].astype(float)

    # Leak-free preprocessing (MICE + OHE + MinMax)
    preproc = build_preprocessor_minmax(model2_feats, random_state=seed)
    Xtr_proc = preproc.fit_transform(Xtr_raw, ytr)
    Xte_proc = preproc.transform(Xte_raw)

    try:
        feat_names = preproc.get_feature_names_out()
    except Exception:
        feat_names = [f"feat_{i}" for i in range(Xtr_proc.shape[1])]

    Xtr = pd.DataFrame(Xtr_proc, index=outer_train_idx, columns=feat_names)
    Xte = pd.DataFrame(Xte_proc, index=outer_test_idx,  columns=feat_names)

    inner_groups = df_mansa_9_15.loc[outer_train_idx, GROUP_COL].to_numpy()

    # Random Forest pipeline
    pipe_rf = Pipeline([
        ("rf", RandomForestRegressor(
            random_state=seed,
            n_jobs=-1,
            oob_score=False)),
    ])

    # Inner loop tuning
    gcv = GridSearchCV(
        estimator=pipe_rf,
        param_grid=param_grid_rf_m2,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

    t0 = time.time()
    gcv.fit(Xtr, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_

    # Outer test evaluation
    ypred = best_model.predict(Xte)
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

    # Feature importances per fold
    rf_step = best_model.named_steps["rf"]
    importances_per_outer.append(
        pd.Series(rf_step.feature_importances_, index=Xtr.columns)
          .sort_values(ascending=False)
          .rename(f"rep{rep:02d}_fold{ofold:02d}")
    )

    #Save predictions
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"RF_model2_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,
        )

    print(f"[RF_model2 rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | best={gcv.best_params_} | "
          f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

    gc.collect()

# ===== Aggregation =====

res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2": ["mean","std"],
    "RMSE": ["mean","std"],
    "MAE": ["mean","std"],
})
summ_flat = summ.reset_index().rename(columns={"index": "metric"})

res_df.to_csv(os.path.join(rf_model2_dir, "RF_model2_nestedcv_folds.csv"), index=False)
summ_flat.to_csv(os.path.join(rf_model2_dir, "RF_model2_nestedcv_summary.csv"), index=False)

# Mean feature importances
if importances_per_outer:
    imp_df   = pd.concat(importances_per_outer, axis=1).fillna(0.0)
    imp_mean = imp_df.mean(axis=1).sort_values(ascending=False)
    imp_mean.to_csv(os.path.join(rf_model2_dir, "RF_model2_feature_importances_mean.csv"))
else:
    imp_mean = pd.Series(dtype=float)

pd.DataFrame(best_params_list).to_csv(
    os.path.join(rf_model2_dir, "RF_model2_bestparams_per_outer.csv"),
    index=False,
)

excel_path = os.path.join(rf_model2_dir, "RF_model2_nestedcv_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ_flat.to_excel(w, sheet_name="summary", index=False)
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
    if not imp_mean.empty:
        imp_mean.rename("mean_importance").to_frame().to_excel(
            w, sheet_name="mean_importance")

print(f"\nSaved all RF_model2 results → {excel_path}")
#%%---------------------------------------------------------------------------------------------------
# Linear SVR + RFE(5) – Model 3 feature set
# Nested GroupKFold CV with 10 seeds (SEEDS)

svr_rfe5_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
os.makedirs(svr_rfe5_dir, exist_ok=True)

importances_dir = os.path.join(svr_rfe5_dir, "selected_features_per_fold")
os.makedirs(importances_dir, exist_ok=True)

param_grid_svr = {
    "svr__C":       [0.01, 0.1, 1, 10, 30],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    "svr__tol":     [1e-4, 1e-3],
}

rows = []
best_params_list = []
SAVE_PREDS = True

inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)

for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):

    seed = SEEDS[rep - 1]
    out_dir = svr_rfe5_dir

    # indices opslaan voor reproduceerbaarheid
    np.save(os.path.join(out_dir, f"svr_linear_rfe5_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(out_dir, f"svr_linear_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)

    # ruwe data
    Xtr_raw = df_mansa_9_15.loc[outer_train_idx, model3_feats].copy()
    Xte_raw = df_mansa_9_15.loc[outer_test_idx,  model3_feats].copy()
    ytr     = df_mansa_9_15.loc[outer_train_idx, ycol].astype(float)
    yte     = df_mansa_9_15.loc[outer_test_idx,  ycol].astype(float)

    # leak-free preprocessing (MICE + OHE + MinMax)
    preproc = build_preprocessor_minmax(model3_feats, random_state=seed)
    Xtr_proc = preproc.fit_transform(Xtr_raw, ytr)
    Xte_proc = preproc.transform(Xte_raw)

    try:
        feat_names = preproc.get_feature_names_out()
    except Exception:
        feat_names = [f"feat_{i}" for i in range(Xtr_proc.shape[1])]

    Xtr = pd.DataFrame(Xtr_proc, index=outer_train_idx, columns=feat_names)
    Xte = pd.DataFrame(Xte_proc, index=outer_test_idx,  columns=feat_names)

    inner_groups = df_mansa_9_15.loc[outer_train_idx, GROUP_COL].to_numpy()

    # basis-SVR voor RFE (werkt op al geschaalde features)
    base_svr = LinearSVR(random_state=seed, max_iter=20000)

    pipe = Pipeline([
        ("rfe", RFE(estimator=base_svr, n_features_to_select=5, step=1)),
        ("svr", LinearSVR(random_state=seed, max_iter=20000)),
    ])

    gcv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_svr,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

    t0 = time.time()
    gcv.fit(Xtr, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_

    # RFE-top5 features voor deze outer fold opslaan
    support_mask = best_model.named_steps["rfe"].support_
    top5_feats = list(pd.Index(Xtr.columns)[support_mask])
    pd.DataFrame({"selected_features": top5_feats}).to_csv(
        os.path.join(importances_dir,
                     f"svr_rfe5_rep{rep:02d}_fold{ofold:02d}_features.csv"),
        index=False,
    )

    # outer test performance
    ypred = best_model.predict(Xte)
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

    # predictions opslaan
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"svr_linear_rfe5_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,
        )

    print(f"[SVR-linear RFE5 rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | best={gcv.best_params_} | "
          f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

    gc.collect()

# ===== Aggregatie =====

res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2":   ["mean","std"],
    "RMSE": ["mean","std"],
    "MAE":  ["mean","std"],
})

summ_flat = summ.reset_index().rename(columns={"index": "metric"})

res_df.to_csv(
    os.path.join(svr_rfe5_dir, "svr_linear_rfe5_repeated_nestedcv_folds.csv"),
    index=False,
)
summ_flat.to_csv(
    os.path.join(svr_rfe5_dir, "svr_linear_rfe5_repeated_nestedcv_summary.csv"),
    index=False,
)

excel_path = os.path.join(svr_rfe5_dir, "svr_linear_rfe5_repeated_nestedcv_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ_flat.to_excel(w, sheet_name="summary", index=False)
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)

print(f"\nSaved all LinearSVR RFE5 results → {excel_path}")
#%%---------------------------------------------------------------------------------------------------
# Linear Regression + RFE(5) op Model 3 feature set
# Repeated GroupKFold (10 seeds via SEEDS, N_OUTER_SPLITS folds)
linear_regression_baseline_rfe5 = p("cross_validation_results", "linear_regression_rfe5")
os.makedirs(linear_regression_baseline_rfe5, exist_ok=True)

importances_dir_rfe5 = os.path.join(linear_regression_baseline_rfe5, "selected_features_per_fold")
os.makedirs(importances_dir_rfe5, exist_ok=True)

rows = []
SAVE_PREDS = True

for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):

    seed = SEEDS[rep - 1]
    out_dir = linear_regression_baseline_rfe5

    # indices opslaan (reproduceerbaarheid)
    np.save(os.path.join(out_dir, f"linreg_rfe5_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(out_dir, f"linreg_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)

    # ruwe data voor deze outer-fold
    Xtr_raw = df_mansa_9_15.loc[outer_train_idx, model3_feats].copy()
    Xte_raw = df_mansa_9_15.loc[outer_test_idx,  model3_feats].copy()
    ytr     = df_mansa_9_15.loc[outer_train_idx, ycol].astype(float)
    yte     = df_mansa_9_15.loc[outer_test_idx,  ycol].astype(float)

    # leak-free preprocessing (MICE + MinMax + OHE) binnen de outer-train
    preproc = build_preprocessor_minmax(model3_feats, random_state=seed)
    Xtr_proc = preproc.fit_transform(Xtr_raw, ytr)
    Xte_proc = preproc.transform(Xte_raw)

    # feature-namen na ColumnTransformer
    try:
        feat_names = preproc.get_feature_names_out()
    except Exception:
        feat_names = [f"feat_{i}" for i in range(Xtr_proc.shape[1])]

    Xtr = pd.DataFrame(Xtr_proc, index=outer_train_idx, columns=feat_names)
    Xte = pd.DataFrame(Xte_proc, index=outer_test_idx,  columns=feat_names)

    # RFE(5) op de geschaalde design-matrix (alleen op outer-train)
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
    rfe.fit(Xtr, ytr)

    top5_feats = list(pd.Index(Xtr.columns)[rfe.support_])

    # geselecteerde features per rep/fold bewaren
    pd.DataFrame({"selected_features": top5_feats}).to_csv(
        os.path.join(importances_dir_rfe5,
                     f"linreg_rfe5_rep{rep:02d}_fold{ofold:02d}_features.csv"),
        index=False,
    )

    #train/test beperken tot top-5 features
    Xtr_sel = Xtr[top5_feats].copy()
    Xte_sel = Xte[top5_feats].copy()

    #Lineaire regressie op de 5 geselecteerde features
    model = LinearRegression()
    t0 = time.time()
    model.fit(Xtr_sel, ytr)
    ypred = model.predict(Xte_sel)

    #metrics op outer-test
    r2   = r2_score(yte, ypred)
    rmse = mean_squared_error(yte, ypred)**0.5
    mae  = mean_absolute_error(yte, ypred)

    rows.append({
        "rep": rep,
        "outer_fold": ofold,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
    })

    # predictions opslaan
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": yte.values,
            "y_pred": ypred,
        }).to_csv(
            os.path.join(out_dir, f"linreg_rfe5_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,
        )

    print(f"[LinReg-RFE5 rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
          f"seed={seed} | R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

    gc.collect()

# aggregatie over alle reps × folds
res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2":   ["mean", "std"],
    "RMSE": ["mean", "std"],
    "MAE":  ["mean", "std"],
})

res_df.to_csv(
    os.path.join(linear_regression_baseline_rfe5, "linreg_rfe5_folds.csv"),
    index=False,
)
summ.to_csv(
    os.path.join(linear_regression_baseline_rfe5, "linreg_rfe5_summary.csv"),
)

excel_path = os.path.join(linear_regression_baseline_rfe5, "linreg_rfe5_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ.to_excel(w, sheet_name="summary")

print(f"\nSaved → {excel_path}")
print("\nLinearRegression(Model3 + RFE5) summary:")
print(summ.round(3))
#%%---------------------------------------------------------------------------------------------------
# Elastic Net (Nested CV, 10 seeds, Model 3 feature set)
elastic_net_dir = p("cross_validation_results", "elastic_net_cv_results")
os.makedirs(elastic_net_dir, exist_ok=True)

param_grid_en = {
    "elasticnet__alpha":    [0.0005, 0.001, 0.01, 0.1, 1.0, 10.0],
    "elasticnet__l1_ratio": [0.1, 0.5, 0.7, 0.9, 1.0],
}

rows = []
best_params_list = []
SAVE_PREDS = True

inner_cv = GroupKFold(n_splits=N_INNER_SPLITS)

for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS):

    seed = SEEDS[rep - 1]

    # save outer indices
    np.save(os.path.join(elastic_net_dir,
                         f"EN_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
            outer_train_idx)
    np.save(os.path.join(elastic_net_dir,
                         f"EN_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
            outer_test_idx)

    # raw outer data
    Xtr_raw = df_mansa_9_15.loc[outer_train_idx, model3_feats].copy()
    Xte_raw = df_mansa_9_15.loc[outer_test_idx,  model3_feats].copy()
    ytr     = df_mansa_9_15.loc[outer_train_idx, ycol].astype(float)
    yte     = df_mansa_9_15.loc[outer_test_idx,  ycol].astype(float)

    # leak-free preprocessing: MICE + MinMax + OHE
    preproc = build_preprocessor_minmax(model3_feats, random_state=seed)
    Xtr_proc = preproc.fit_transform(Xtr_raw, ytr)
    Xte_proc = preproc.transform(Xte_raw)

    try:
        feat_names = preproc.get_feature_names_out()
    except:
        feat_names = [f"feat_{i}" for i in range(Xtr_proc.shape[1])]

    Xtr = pd.DataFrame(Xtr_proc, index=outer_train_idx, columns=feat_names)
    Xte = pd.DataFrame(Xte_proc, index=outer_test_idx,  columns=feat_names)

    inner_groups = df_mansa_9_15.loc[outer_train_idx, GROUP_COL].to_numpy()

    # Elastic Net pipeline — scaler removed (already scaled in preprocessor)
    pipe_en = Pipeline([
        ("elasticnet", ElasticNet(random_state=seed, max_iter=20000)),
    ])

    gcv = GridSearchCV(
        estimator=pipe_en,
        param_grid=param_grid_en,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

    t0 = time.time()
    gcv.fit(Xtr, ytr, groups=inner_groups)
    best_model = gcv.best_estimator_

    # predict outer fold
    ypred = best_model.predict(Xte)

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
          f"seed={seed} | best={gcv.best_params_} | "
          f"R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

    gc.collect()

# aggregate over 10×10 = 100 outer test evaluations
res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2":   ["mean", "std"],
    "RMSE": ["mean", "std"],
    "MAE":  ["mean", "std"],
})

res_df.to_csv(os.path.join(elastic_net_dir, "elastic_net_nestedcv_folds.csv"),
              index=False)
summ.to_csv(os.path.join(elastic_net_dir, "elastic_net_nestedcv_summary.csv"))

pd.DataFrame(best_params_list).to_csv(
    os.path.join(elastic_net_dir, "elastic_net_bestparams_per_outer.csv"),
    index=False,
)

excel_path = os.path.join(elastic_net_dir, "elastic_net_nestedcv_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ.to_excel(w, sheet_name="summary")
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)

print(f"\nSaved all ElasticNet nested CV results → {excel_path}")
print(summ.round(3))
#%%----------------------------------------------------------------------------------------------------------------------------
#Making the comparison of these second run of cross validations + random forest model 2 + linear regression cross-validation baseline
base_dir = p("cross_validation_results")
out_dir = os.path.join(base_dir, "comparison_all_models_2")
os.makedirs(out_dir, exist_ok=True)
models = {
    "Random Forest":   os.path.join(base_dir, "random_forest.2",
                                    "RF_mm_nestedcv_summary.csv"),
    "XGBoost":         os.path.join(base_dir, "xgboost_cv_results.2",
                                    "xgb_mm_nestedcv_summary.csv"),
    "kNN":             os.path.join(base_dir, "kNN_cv_results.2",
                                    "knn_repeated_nestedcv_summary_2.csv"),
    "Linear SVR":      os.path.join(base_dir, "svr_linear_cv_results.2_rfe5",
                                    "svr_linear_rfe5_repeated_nestedcv_summary.csv"),
    "Linear Regression (M3)": os.path.join(base_dir, "linear_regression_rfe5",
                                           "linreg_rfe5_summary.csv"),
    "Random Forest (Model 2)": os.path.join(base_dir, "random_forest_model_2",
                                            "RF_model2_nestedcv_summary.csv"),
    "Elastic Net": os.path.join(base_dir, "elastic_net_cv_results",
                                "elastic_net_nestedcv_summary.csv"),}
rows = []
for model_name, path in models.items():
    if os.path.exists(path):
        wide = pd.read_csv(path, index_col=0)
        df = (
            wide.T
            .reset_index()
            .rename(columns={"index": "metric",
                             "mean": "mean",
                             "std": "sd"}))
        df["Model"] = model_name
        rows.append(df)
    else:
        print(f" NOT FOUND: {model_name}: {path}")
all_results = pd.concat(rows, ignore_index=True)
comparison = all_results.pivot_table(
    index="metric", columns="Model", values="mean")
sd_table = all_results.pivot_table(
    index="metric", columns="Model", values="sd")
comparison_with_sd = (comparison.round(3).astype(str) + " ± " + sd_table.round(3).astype(str))
rank_rows = {}
if "R2" in comparison.index:
    rank_rows["R2"] = comparison.loc["R2"].rank(ascending=False).astype(int)
if "RMSE" in comparison.index:
    rank_rows["RMSE"] = comparison.loc["RMSE"].rank(ascending=True).astype(int)
if "MAE" in comparison.index:
    rank_rows["MAE"] = comparison.loc["MAE"].rank(ascending=True).astype(int)
rank_df = pd.DataFrame(rank_rows)
out_excel = os.path.join(out_dir, "comparison_summary_ML_models_2.xlsx")
with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    all_results.to_excel(writer, sheet_name="raw", index=False)
    comparison.to_excel(writer, sheet_name="mean_only")
    comparison_with_sd.to_excel(writer, sheet_name="mean±sd")
    rank_df.to_excel(writer, sheet_name="ranks (1=best)")
print("\nComparison saved →", out_excel)
print("\nMEANS:")
print(comparison.round(3))
print("\nRANKS:")
print(rank_df)
#%%------------------------------------------------------------------------------------------
#Make visualisations
#Output directory
save_dir = p("cross_validation_results", "comparison_all_models_2")
#Load comparison results
comp_path = os.path.join(save_dir, "comparison_summary_ML_models_2.xlsx")
mean_df = pd.read_excel(comp_path, sheet_name="mean_only", index_col=0).T  # models in rows
sd_with_text = pd.read_excel(comp_path, sheet_name="mean±sd", index_col=0).T
sd_df = sd_with_text.applymap(lambda x: float(str(x).split("±")[1].strip()))
#Colors per model
colors = {
    "Linear Regression (M3)": "#1f77b4",
    "Linear SVR": "#ff7f0e",
    "Elastic Net": "#5a645e",
    "Random Forest": "#2ca02c",
    "Random Forest (Model 2)": "#d62728",
    "XGBoost": "#9467bd",
    "kNN": "#8c564b",}
#Shorter labels for the overview
short_labels = {
    "Linear Regression (M3)": "LinReg (M3)",
    "Linear SVR": "Linear SVR",
    "Elastic Net": "Elastic Net",
    "Random Forest": "RF",
    "Random Forest (Model 2)": "RF (M2)",
    "XGBoost": "XGBoost",
    "kNN": "kNN",}
models = mean_df.index.tolist()
x = np.arange(len(models))
#Barplot
plt.figure(figsize=(10, 5))
means = mean_df["R2"].values
errors = sd_df["R2"].values * 1.96  # 95% CI
bar_colors = [colors[m] for m in models]
plt.bar(x, means, yerr=errors, capsize=5, edgecolor="black", color=bar_colors)
plt.ylabel("R²")
plt.title("Model Performance: R² (higher = better)")
plt.xticks(x, [short_labels[m] for m in models], rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "plot_R2_models.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "plot_R2_models.pdf"))
plt.show()
#Barplot RMSE
plt.figure(figsize=(10, 5))
means = mean_df["RMSE"].values
errors = sd_df["RMSE"].values * 1.96
plt.bar(x, means, yerr=errors, capsize=5, edgecolor="black", color=bar_colors)
plt.ylabel("RMSE")
plt.title("Model Performance: RMSE (lower = better)")
plt.xticks(x, [short_labels[m] for m in models], rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "plot_RMSE_models.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "plot_RMSE_models.pdf"))
plt.show()
#Radarplot
metrics = ["R2", "RMSE", "MAE"]
#Normalise
norm_df = mean_df.copy()
norm_df["R2"] = (norm_df["R2"] - norm_df["R2"].min()) / (mean_df["R2"].max() - mean_df["R2"].min())
norm_df["RMSE"] = 1 - (mean_df["RMSE"] - mean_df["RMSE"].min()) / (mean_df["RMSE"].max() - mean_df["RMSE"].min())
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
#Extra visualisation?
base_dir = p("cross_validation_results")

# mapping: modelnaam → path naar folds-resultaat
fold_files = {
    "Elastic Net": os.path.join(base_dir, "elastic_net_cv_results", "elastic_net_nestedcv_folds.csv"),
    "Linear Regression (M3)": os.path.join(base_dir, "linear_regression_rfe5", "linreg_rfe5_folds.csv"),
    "Linear SVR": os.path.join(base_dir, "svr_linear_cv_results.2_rfe5", "svr_linear_rfe5_repeated_nestedcv_folds.csv"),
    "Random Forest": os.path.join(base_dir, "random_forest.2", "RF_mm_nestedcv_folds.csv"),
    "Random Forest (Model 2)": os.path.join(base_dir, "random_forest_model_2", "RF_model2_nestedcv_folds.csv"),
    "XGBoost": os.path.join(base_dir, "xgboost_cv_results.2", "xgb_mm_nestedcv_folds.csv"),
    "kNN": os.path.join(base_dir, "kNN_cv_results.2", "knn_repeated_nestedcv_folds_2.csv"),
}

all_models = []
for model_name, path in fold_files.items():
    if not os.path.exists(path):
        print("Niet gevonden:", model_name, path)
        continue

    df = pd.read_csv(path)  # bevat rep, outer_fold, R2, RMSE, MAE

    # per seed (rep) gemiddelden
    grp = (
        df.groupby("rep")[["R2", "RMSE", "MAE"]]
        .mean()
        .reset_index()
    )
    grp["Model"] = model_name
    all_models.append(grp)

per_seed = pd.concat(all_models, ignore_index=True)

# lang → breed
# We willen een tabel per seed zoals jouw screenshot
output_path = os.path.join(base_dir, "comparison_all_models_per_seed.xlsx")
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for rep in sorted(per_seed["rep"].unique()):
        rep_df = per_seed[per_seed["rep"] == rep]

        table = rep_df.pivot_table(
            index="Model",
            values=["MAE", "R2", "RMSE"]
        ).T  # zodat metrics in de rijen komen zoals bij jou

        table.to_excel(writer, sheet_name=f"seed_{rep}")

print("10 tabellen opgeslagen in:", output_path)

#%%--------------------------------------------------------------------------
#Linear SVR performs constantly the best. This is the model we chose!
#Add uncertainty estimation using repeated GroupKFold predictions to see the uncertainty per patient
GROUP_COL = "Proefpersoonnummer"
svr_cv_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
pred_prefix = "svr_linear_rfe5"  
idx_prefix = "svr_linear_rfe5"    
unc_threshold = 1.5
#Get all prediction files
pred_files = sorted(glob(os.path.join(svr_cv_dir, f"{pred_prefix}_rep*_outer*_preds.csv")))
if not pred_files:
    raise FileNotFoundError(f"No prediction files found under {svr_cv_dir}")
pred_rows = []
for pred_path in pred_files:
    match = re.search(r"rep(\d+)_outer(\d+)_preds", os.path.basename(pred_path))
    if match is None:
        continue
    rep = int(match.group(1))
    outer_fold = int(match.group(2))
    #Load test indices to retrieve patient IDs
    idx_path = os.path.join(svr_cv_dir, f"{idx_prefix}_rep{rep:02d}_outer_test_idx_{outer_fold:02d}.npy")
    preds = pd.read_csv(pred_path)
    test_idx = np.load(idx_path)
    patient_ids = df_mansa_9_15.loc[test_idx, GROUP_COL].to_numpy()
    #Add repetition, fold, index and patient ID to the dataframe
    preds["rep"] = rep
    preds["outer_fold"] = outer_fold
    preds["patient_index"] = test_idx
    preds[GROUP_COL] = patient_ids
    pred_rows.append(preds)
#Combine all predictions into one dataframe + Saving
all_preds = pd.concat(pred_rows, ignore_index=True)
unc_dir = os.path.join(svr_cv_dir, "uncertainty_analysis")
os.makedirs(unc_dir, exist_ok=True)
all_preds.to_csv(os.path.join(unc_dir, f"{pred_prefix}_all_preds.csv"), index=False)
#Calculate uncertainty metrics per patient
uncertainty_df = (
    all_preds.groupby(GROUP_COL)
    .agg(
        n_predictions=("y_pred", "size"),
        y_true_mean=("y_true", "mean"),
        y_pred_mean=("y_pred", "mean"),
        y_pred_std=("y_pred", "std"),).reset_index())
uncertainty_df["is_uncertain"] = uncertainty_df["y_pred_std"] > unc_threshold
uncertainty_df.to_csv(os.path.join(unc_dir, f"{pred_prefix}_uncertainty.csv"), index=False)
#Summary and numbers
pct_uncertain = 100 * uncertainty_df["is_uncertain"].mean()
n_patients = uncertainty_df[GROUP_COL].nunique()
n_uncertain = (uncertainty_df["is_uncertain"]).sum()
print(f"{pct_uncertain:.1f}% of patients have Linear SVR prediction SD > {unc_threshold}")
print(f"→ This corresponds to {n_uncertain} out of {n_patients} patients")
print(f"Uncertainty stats – patients: {n_patients}")
print(f"y_pred_std median={uncertainty_df['y_pred_std'].median():.3f}, "
      f"75th pct={uncertainty_df['y_pred_std'].quantile(0.75):.3f}, "
      f"90th pct={uncertainty_df['y_pred_std'].quantile(0.90):.3f}")
#Histogram
plt.figure(figsize=(6, 4))
sns.histplot(uncertainty_df["y_pred_std"].dropna(), bins=30, color="#1f77b4")
plt.axvline(unc_threshold, color="red", linestyle="--", label=f"SD = {unc_threshold}")
plt.xlabel("Prediction SD across folds")
plt.ylabel("Count")
plt.title("Linear SVR prediction uncertainty")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(unc_dir, f"{pred_prefix}_uncertainty_hist.png"), dpi=300)
#Scatterplot
plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    uncertainty_df["y_true_mean"],
    uncertainty_df["y_pred_mean"],
    c=uncertainty_df["y_pred_std"],
    cmap="viridis",
    s=45,)
plt.axline((0, 0), slope=1, color="gray", linestyle="--")
plt.xlabel("Mean observed score")
plt.ylabel("Mean predicted score")
plt.title("Linear SVR per-patient prediction stability")
cbar = plt.colorbar(scatter)
cbar.set_label("Prediction SD")
plt.tight_layout()
plt.savefig(os.path.join(unc_dir, f"{pred_prefix}_uncertainty_scatter.png"), dpi=300)
plt.show()
plt.close()
#Checking what these models do without the uncertain patients in the linear SVR
#Kies de drempel die je al gebruikt (is_uncertain zit al in uncertainty_df)
uncertain_ids = set(uncertainty_df.loc[uncertainty_df["is_uncertain"], GROUP_COL])
filtered_preds = all_preds[~all_preds[GROUP_COL].isin(uncertain_ids)].copy()
def metrics(df, label):
    r2   = r2_score(df["y_true"], df["y_pred"])
    rmse = mean_squared_error(df["y_true"], df["y_pred"]) ** 0.5
    mae  = mean_absolute_error(df["y_true"], df["y_pred"])
    print(f"{label}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, n={len(df)}")
print("=== Robustness check: exclude high-uncertainty patients ===")
metrics(all_preds, "All patients")
metrics(filtered_preds, "Excluding uncertain patients")
print(f"Removed {len(uncertain_ids)} patients out of {uncertainty_df[GROUP_COL].nunique()}")
#%%-------------------------------------------------------
#I want to test it for random forest to see if this explains why this model performs worse
#Add uncertainty estimation using repeated GroupKFold predictions to see the uncertainty per patient
GROUP_COL = "Proefpersoonnummer"
random_forest_cv_dir = p("cross_validation_results", "random_forest.2")
pred_prefix = "RF_mm"        
idx_prefix  = "RF_mm"  
unc_threshold = 1.5
pred_files = sorted(
    glob(os.path.join(random_forest_cv_dir,
                      f"{pred_prefix}_rep*_outer*_preds.csv")))
if not pred_files:
    raise FileNotFoundError(f"No prediction files found under {random_forest_cv_dir}")
pred_rows = []
for pred_path in pred_files:
    match = re.search(r"rep(\d+)_outer(\d+)_preds",
                      os.path.basename(pred_path))
    if match is None:
        continue
    rep = int(match.group(1))
    outer_fold = int(match.group(2))
    idx_path = os.path.join(
        random_forest_cv_dir,
        f"{idx_prefix}_rep{rep:02d}_outer_test_idx_{outer_fold:02d}.npy")
    preds = pd.read_csv(pred_path)
    test_idx = np.load(idx_path)
    if len(test_idx) != len(preds):
        raise ValueError(
            f"Length mismatch rep{rep} fold{outer_fold}: "
            f"idx {len(test_idx)} vs preds {len(preds)}")
    patient_ids = df_mansa_9_15.loc[test_idx, GROUP_COL].to_numpy()
    preds["rep"] = rep
    preds["outer_fold"] = outer_fold
    preds["patient_index"] = test_idx
    preds[GROUP_COL] = patient_ids
    pred_rows.append(preds)
#Combine all predictions into one dataframe + Saving
all_preds = pd.concat(pred_rows, ignore_index=True)
unc_dir = os.path.join(random_forest_cv_dir, "uncertainty_analysis")
os.makedirs(unc_dir, exist_ok=True)
all_preds.to_csv(os.path.join(unc_dir, f"{pred_prefix}_all_preds.csv"), index=False)
#Calculate uncertainty metrics per patient
uncertainty_df = (
    all_preds.groupby(GROUP_COL)
    .agg(
        n_predictions=("y_pred", "size"),
        y_true_mean=("y_true", "mean"),
        y_pred_mean=("y_pred", "mean"),
        y_pred_std=("y_pred", "std"),).reset_index())
uncertainty_df["is_uncertain"] = uncertainty_df["y_pred_std"] > unc_threshold
uncertainty_df.to_csv(os.path.join(unc_dir, f"{pred_prefix}_uncertainty.csv"), index=False)
#Summary and numbers
pct_uncertain = 100 * uncertainty_df["is_uncertain"].mean()
n_patients = uncertainty_df[GROUP_COL].nunique()
n_uncertain = (uncertainty_df["is_uncertain"]).sum()
print(f"{pct_uncertain:.1f}% of patients have Random Forest prediction SD > {unc_threshold}")
print(f"→ This corresponds to {n_uncertain} out of {n_patients} patients")
print(f"Uncertainty stats – patients: {n_patients}")
print(f"y_pred_std median={uncertainty_df['y_pred_std'].median():.3f}, "
      f"75th pct={uncertainty_df['y_pred_std'].quantile(0.75):.3f}, "
      f"90th pct={uncertainty_df['y_pred_std'].quantile(0.90):.3f}")
#Histogram
plt.figure(figsize=(6, 4))
sns.histplot(uncertainty_df["y_pred_std"].dropna(), bins=30, color="#1f77b4")
plt.axvline(unc_threshold, color="red", linestyle="--", label=f"SD = {unc_threshold}")
plt.xlabel("Prediction SD across folds")
plt.ylabel("Count")
plt.title("Random Forest prediction uncertainty")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(unc_dir, f"{pred_prefix}_uncertainty_hist.png"), dpi=300)
#Scatterplot
plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    uncertainty_df["y_true_mean"],
    uncertainty_df["y_pred_mean"],
    c=uncertainty_df["y_pred_std"],
    cmap="viridis",
    s=45,)
plt.axline((0, 0), slope=1, color="gray", linestyle="--")
plt.xlabel("Mean observed score")
plt.ylabel("Mean predicted score")
plt.title("Random Forest per-patient prediction stability")
cbar = plt.colorbar(scatter)
cbar.set_label("Prediction SD")
plt.tight_layout()
plt.savefig(os.path.join(unc_dir, f"{pred_prefix}_uncertainty_scatter.png"), dpi=300)
plt.show()
plt.close()
#Checking what these models do without the uncertain patients in the random forest thing!
uncertain_ids = set(uncertainty_df.loc[uncertainty_df["is_uncertain"], GROUP_COL])
filtered_preds = all_preds[~all_preds[GROUP_COL].isin(uncertain_ids)].copy()
def metrics(df, label):
    r2   = r2_score(df["y_true"], df["y_pred"])
    rmse = mean_squared_error(df["y_true"], df["y_pred"]) ** 0.5
    mae  = mean_absolute_error(df["y_true"], df["y_pred"])
    print(f"{label}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, n={len(df)}")
print("Robustness check: exclude high-uncertainty patients")
metrics(all_preds, "All patients")
metrics(filtered_preds, "Excluding uncertain patients")
print(f"Removed {len(uncertain_ids)} patients out of {uncertainty_df[GROUP_COL].nunique()}")
#%%------------------------------------------------------------------------------------------------------
#EERST ERROR ANALYSIS OP BEIDE
#Using SHAP and error analysis on Random Forest to see if the data is really linear? AND ON LINEAR SVR
#That would explain why the linear models work the best 
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
    if rep is not None and fold is not None:
        dfp["rep"] = rep
        dfp["outer_fold"] = fold
    pred_rows.append(dfp)
all_preds_svr = pd.concat(pred_rows, ignore_index=True)
all_preds_svr["resid"] = all_preds_svr["y_true"] - all_preds_svr["y_pred"]
#Residuals vs predicted
plt.figure(figsize=(6, 4))
sns.scatterplot(data=all_preds_svr, x="y_pred", y="resid", alpha=0.25, s=20, color="#1f77b4")
sns.regplot(data=all_preds_svr, x="y_pred", y="resid", scatter=False, lowess=True, color="red")
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Linear SVR: residuals vs predicted")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "svr_linear_rfe5_residuals_vs_pred.png"), dpi=300)
#Residual distribution
plt.figure(figsize=(6, 4))
sns.histplot(all_preds_svr["resid"], bins=40, kde=True, color="#1f77b4")
plt.axvline(0, color="k", linestyle="--")
plt.xlabel("Residual")
plt.title("Linear SVR: residual distribution")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "svr_linear_rfe5_residual_hist.png"), dpi=300)
# QQ-plot
plt.figure(figsize=(5, 5))
sps.probplot(all_preds_svr["resid"], dist="norm", plot=plt)
plt.title("Linear SVR: residual QQ-plot")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "svr_linear_rfe5_residual_qq.png"), dpi=300)
plt.show()
plt.close()
#There is a bit hetereoscadisty but not too much, so it is respectable
#%%------------------------------------------------------------------------------------
#Residual analysis for Random Forest
#Paths
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
#Residual distribution
plt.figure(figsize=(6, 4))
sns.histplot(all_preds_rf["resid"], bins=40, kde=True, color="#1f77b4")
plt.axvline(0, color="k", linestyle="--")
plt.xlabel("Residual")
plt.title("Random Forest: residual distribution")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "rf_mm_residual_hist.png"), dpi=300)
#Making the QQ-plot
plt.figure(figsize=(5, 5))
sps.probplot(all_preds_svr["resid"], dist="norm", plot=plt)
plt.title("Random Forest: residual QQ-plot")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "rf_mm_residual_qq.png"), dpi=300)
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------
#Trying SHAP on the chosen Linear SVR model to see the features and their influence
#SHAP on the chosen Linear SVR model (final RFE=5 version)
img_dir = p("data_images_MANSA_9_15_months")
os.makedirs(img_dir, exist_ok=True)
svr_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
#Best hyperparameters laden uit de Excel met best_params
bestparams_path = os.path.join(svr_dir, "svr_linear_rfe5_repeated_nestedcv_results.xlsx")
best_df = pd.read_excel(bestparams_path, sheet_name="best_params")
#Mode per kolom 
mode_params = best_df.mode().iloc[0].to_dict()
svr_kwargs = {k.replace("svr__", ""): v for k, v in mode_params.items() if k.startswith("svr__")}
svr_kwargs.setdefault("random_state", 42)
svr_kwargs.setdefault("max_iter", 20000)
#Building the design matrix with the preprocessing
ycol = "mansa_totaal.2"
full_idx = df_mansa_9_15.index[df_mansa_9_15[ycol].notna()].to_numpy()
X_raw = df_mansa_9_15.loc[full_idx, model3_feats].copy()
y_all = df_mansa_9_15.loc[full_idx, ycol].astype(float).copy()
preproc = build_preprocessor_minmax(model3_feats, random_state = 42)
X_proc = preproc.fit_transform(X_raw, y_all)
try:
    feat_names = preproc.get_feature_names_out()
except Exception:
    feat_names = [f"feat_{i}" for i in range(X_proc.shape[1])]
X_design = pd.DataFrame(X_proc, index=full_idx, columns=feat_names)
y = y_all.values
#Fitting final LinearSVR fitten op de volledig gepreprocesseerde matrix (geen extra scaler!)
svr = LinearSVR(**svr_kwargs)
svr.fit(X_design, y)
# 4) Sample voor SHAP (max 800 rijen)
n_sample = min(800, len(X_design))
rng = np.random.default_rng(0)
sample_idx = rng.choice(len(X_design), size=n_sample, replace=False)
X_sample = X_design.iloc[sample_idx, :]
#SHAP explainer for a lineair model
explainer = shap.LinearExplainer(svr, X_design.values)
shap_values_exp = explainer(X_sample.values)
shap_values = getattr(shap_values_exp, "values", shap_values_exp)
#Visualizing SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_sample.values,
                  feature_names=X_design.columns, show=False)
plt.tight_layout()
out_path = os.path.join(img_dir, "svr_linear_rfe5_shap_summary.png")
plt.savefig(out_path, dpi=300)
print("SHAP summary saved to:", out_path)
#Top-3 features + dependence plots
mean_abs = np.abs(shap_values).mean(axis=0)
top_features = X_design.columns[np.argsort(mean_abs)[::-1][:3]]
print("Top-3 SHAP features (SVR):", list(top_features))
for feat in top_features:
    plt.figure()
    shap.dependence_plot(
        feat,
        shap_values,
        X_sample.values,
        feature_names=X_design.columns,
        show=False,)
    plt.tight_layout()
    out_dep = os.path.join(img_dir, f"svr_linear_rfe5_shap_dependence_{feat}.png")
    plt.savefig(out_dep, dpi=300)
    print("Saved:", out_dep)
plt.show()
plt.close()
#%%-------------------------------------------------------------------
#Trying SHAP on Random Forest to see if it discovers some non-linear patterns
img_dir = p("data_images_MANSA_9_15_months")
os.makedirs(img_dir, exist_ok=True)
rf_dir = p("cross_validation_results", "random_forest.2")
#Load RF best hyperparameters 
bestparams_path = os.path.join(rf_dir, "RF_mm_nestedcv_results.xlsx")
best_df = pd.read_excel(bestparams_path, sheet_name="best_params")
mode_params = best_df.mode().iloc[0].to_dict()
rf_kwargs = {
    k.replace("rf__", ""): v
    for k, v in mode_params.items()
    if k.startswith("rf__")}
rf_kwargs.setdefault("random_state", 42)
rf_kwargs.setdefault("n_jobs", -1)
#Convert ints
for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
    if k in rf_kwargs and pd.notna(rf_kwargs[k]):
        rf_kwargs[k] = int(rf_kwargs[k])
#Build full design matrix using SAME preprocessing as CV
ycol = "mansa_totaal.2"
full_idx = df_mansa_9_15.index[df_mansa_9_15[ycol].notna()].to_numpy()
X_raw = df_mansa_9_15.loc[full_idx, model3_feats].copy()
y_all = df_mansa_9_15.loc[full_idx, ycol].astype(float).copy()
#Preprocess exactly like CV
preproc = build_preprocessor_minmax(model3_feats,random_state = 42)
X_proc = preproc.fit_transform(X_raw, y_all)
try:
    feat_names = preproc.get_feature_names_out()
except Exception:
    feat_names = [f"feat_{i}" for i in range(X_proc.shape[1])]
X_full = pd.DataFrame(X_proc, index=full_idx, columns=feat_names)
y = y_all.values
#Fit final Random Forest
rf = RandomForestRegressor(**rf_kwargs)
rf.fit(X_full, y)
#Sample max 800 rows for SHAP
n_sample = min(800, len(X_full))
sample = X_full.sample(n=n_sample, random_state=0)
#SHAP Tree explainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(sample)
#Visualizing the summary plot
plt.figure()
shap.summary_plot(shap_values, sample,
                  feature_names=X_full.columns, show=False)
plt.tight_layout()
out_sum = os.path.join(img_dir, "rf_thirdtry_shap_summary.png")
plt.savefig(out_sum, dpi=300)
print("SHAP summary saved to:", out_sum)
#Top-3 most important features
mean_abs = np.abs(shap_values).mean(axis=0)
top_feats = sample.columns[np.argsort(mean_abs)[::-1][:3]]
print("Top-3 SHAP features (RF):", list(top_feats))
#Visualizing dependence plots
for feat in top_feats:
    plt.figure()
    shap.dependence_plot(
        feat, shap_values, sample,
        feature_names=X_full.columns, show=False)
    plt.tight_layout()
    out_dep = os.path.join(img_dir, f"rf_thirdtry_shap_dependence_{feat}.png")
    plt.savefig(out_dep, dpi=300)
    print("Saved:", out_dep)
plt.show()
plt.close()
#%%------------------------------------------------------------------------------------------
#Because of the fact that SHAP is not really useful for linear models, I am going to show the model through the coefficients and bootstrapping
#VOLGENS CHAT IS DIT OOK NIET DE JUISTE MANIER:
#Bootstrap-CIs voor SVR-coefs zijn “approximate”
#Technisch: SVR is geen lineair regressiemodel met normale fouten, dus je CIs zijn empirische, niet analytische CIs.
#Dit is niet fout, maar je kunt in de thesis 1 zin opnemen dat het non-parametrische bootstrap CIs van een marginaal model zijn, en geen “echte” klassieke 95% CIs.
img_dir = p("data_images_MANSA_9_15_months")
os.makedirs(img_dir, exist_ok=True)
svr_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
feat_dir = os.path.join(svr_dir, "selected_features_per_fold")
#Hyperparameters (modus from nested CV)
best_xlsx = os.path.join(svr_dir, "svr_linear_rfe5_repeated_nestedcv_results.xlsx")
best_df = pd.read_excel(best_xlsx, sheet_name="best_params")
mode_params = best_df.mode().iloc[0].to_dict()
svr_kwargs = {
    k.replace("svr__", ""): v
    for k, v in mode_params.items()
    if k.startswith("svr__")}
svr_kwargs.setdefault("random_state", 42)
svr_kwargs.setdefault("max_iter", 20000)
print("Final LinearSVR hyperparameters (Model 5E):", svr_kwargs)
#Top 5 features
feat_files = glob(os.path.join(feat_dir, "svr_rfe5_rep*_fold*_features.csv"))
if not feat_files:
    raise FileNotFoundError(f"Geen RFE5 feature-bestanden gevonden in {feat_dir}")
freq = Counter()
for fpath in feat_files:
    feats = pd.read_csv(fpath)["selected_features"].dropna()
    freq.update(feats)
freq_df = (
    pd.DataFrame({"feature": list(freq.keys()),
                  "selected_count": list(freq.values())})
    .sort_values("selected_count", ascending=False)
    .reset_index(drop=True))
top5 = freq_df.head(5)["feature"].tolist()
print("RFE Top-5 features (globaal):", top5)
#Building the Design-matrix
ycol = "mansa_totaal.2"
mask = df_mansa_9_15[ycol].notna()
full_idx = df_mansa_9_15.index[mask].to_numpy()
X_raw = df_mansa_9_15.loc[full_idx, model3_feats].copy()
y_all = df_mansa_9_15.loc[full_idx, ycol].astype(float).copy()
preproc = build_preprocessor_minmax(model3_feats, random_state = 42)
X_proc = preproc.fit_transform(X_raw, y_all)
try:
    feat_names = preproc.get_feature_names_out()
except Exception:
    feat_names = [f"feat_{i}" for i in range(X_proc.shape[1])]

X_full = pd.DataFrame(X_proc, index=full_idx, columns=feat_names)
y_vec = y_all.to_numpy()
#Only keeping the top 5 features
missing_top5 = [f for f in top5 if f not in X_full.columns]
if missing_top5:
    raise ValueError(f"Top-5 features niet gevonden in design-matrix: {missing_top5}")
X = X_full[top5].copy()
print("Final Linear SVR dataset (Top-5) shape:", X.shape, y_vec.shape)
#Fitting final model on all the data
svr_final_pipe = make_pipeline(LinearSVR(**svr_kwargs))
svr_final_pipe.fit(X, y_vec)
coefs_scaled = svr_final_pipe.named_steps["linearsvr"].coef_
coef_dir = p("cross_validation_results", "svr_linear_final_top5")
os.makedirs(coef_dir, exist_ok=True)
coef_df = (
    pd.DataFrame({
        "feature": X.columns,
        "scaled_coef": coefs_scaled,
        "abs_scaled_coef": np.abs(coefs_scaled),
    })
    .sort_values("abs_scaled_coef", ascending=False))
coef_path = os.path.join(coef_dir, "svr_linear_final_top5_scaled_coefs.csv")
coef_df.to_csv(coef_path, index=False)
print("Saved scaled coefficients to:", coef_path)
#Bootstrapping
n_boot = 500
boot_coefs = np.zeros((n_boot, X.shape[1]))
rng = np.random.default_rng(42)
for b in range(n_boot):
    idx = rng.integers(0, len(X), len(X))
    X_b = X.iloc[idx]
    y_b = y_vec[idx]
    pipe_b = make_pipeline(LinearSVR(**svr_kwargs))
    pipe_b.fit(X_b, y_b)
    boot_coefs[b, :] = pipe_b.named_steps["linearsvr"].coef_
coef_mean = boot_coefs.mean(axis=0)
ci_low = np.percentile(boot_coefs, 2.5, axis=0)
ci_high = np.percentile(boot_coefs, 97.5, axis=0)
boot_df = (
    pd.DataFrame({
        "feature": X.columns,
        "scaled_coef_mean": coef_mean,
        "ci_2_5": ci_low,
        "ci_97_5": ci_high,
        "abs_scaled_coef_mean": np.abs(coef_mean),
    })
    .sort_values("abs_scaled_coef_mean", ascending=False))
boot_path = os.path.join(
    coef_dir, "svr_linear_final_top5_scaled_coefs_bootstrap_ci.csv")
boot_df.to_csv(boot_path, index=False)
print("Saved bootstrap CIs to:", boot_path)
#Barplot
err_low = boot_df["scaled_coef_mean"] - boot_df["ci_2_5"]
err_high = boot_df["ci_97_5"] - boot_df["scaled_coef_mean"]
plt.figure(figsize=(6, 8))
plt.barh(
    boot_df["feature"],
    boot_df["scaled_coef_mean"],
    xerr=[err_low, err_high],
    color="#1f77b4",
    ecolor="black",
    capsize=3,)
plt.axvline(0, color="black", linewidth=1)
plt.gca().invert_yaxis()
plt.xlabel("Scaled coefficient (mean over bootstrap)")
plt.title("Final Linear SVR (Top-5) – feature effects (95% CI)")
plt.tight_layout()
plot_path = os.path.join(
    img_dir, "svr_linear_final_top5_scaled_coefs_bootstrap_barplot_ci.png")
plt.savefig(plot_path, dpi=300)
print("Saved barplot with CIs to:", plot_path)
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------------------------------------------------
#Subquestion 2
#Making a seperate folder with the results for subquestion 2
subq2_dir = p("subquestion2")
os.makedirs(subq2_dir, exist_ok=True)
#Zelfde single train-test split als bij de OLS-baselines
ycol = "mansa_totaal.2"
valid_idx = df_mansa_9_15.index[df_mansa_9_15[ycol].notna()].to_numpy()
train_idx, test_idx = train_test_split(
    valid_idx,
    test_size=0.2,
    random_state=42,
    shuffle=True,)
print("Subquestion 2 split sizes → train:", len(train_idx), "test:", len(test_idx))
#Item-level predictors of Model 5
Xv2 = df_mansa_9_15.loc[train_idx, model5_feats].copy()
Xv2 = Xv2.apply(pd.to_numeric, errors="coerce")
Xv2 = Xv2.loc[:, Xv2.nunique(dropna=True) > 1]
Xv2_drop = Xv2.dropna()
print("Shape used for VIF (rows, cols):", Xv2_drop.shape)
#Calculating VIF
Xv2_const = sm.add_constant(Xv2_drop, has_constant="add")
arr = Xv2_const.to_numpy(dtype=float, copy=False)
vifs = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
vif_all = (
    pd.DataFrame({"feature": Xv2_const.columns, "VIF": vifs})
      .query("feature != 'const'")
      .sort_values("VIF", ascending=False))
#Save VIF-table
vif_path = os.path.join(subq2_dir, "model5_vif_train_all_features.xlsx")
vif_all.to_excel(vif_path, index=False)
print("Saved Model 5 VIF (all features) to:", vif_path)
print(vif_all.head(20).round(2))
#Correlation-heatmap
plt.figure(figsize=(14, 12))
corr = Xv2_drop.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
plt.title("Correlation Heatmap – All Model 5 Predictors (train design matrix)")
plt.tight_layout()
heatmap_path = os.path.join(subq2_dir, "model5_all_features_corr_heatmap_train.png")
plt.savefig(heatmap_path, dpi=300)
print("Saved heatmap to:", heatmap_path)
plt.show()
plt.close()
#%%---------------------------------------------------------------------------------------------------------------------------
#This made me wonder if model5 would perform good without the MANSA_totaal.1 so I wanted to test that.
#Define features
model5b_feats = demographic_vars + separate_questionnaires
print("\nRunning Model 5b (without MANSA_totaal.1)...")
#Fit + evaluate using the SAME OLS helper as Models 1–5
res5b = fit_eval_linear_model(
    model_name="Model 5b: Demographics + all questionnaire items (NO MANSA_totaal.1)",
    features=model5b_feats,
    plot_filename="linear_regression_model_5b_actual_vs_predicted.png",
    design_prefix="model5b_design_matrix",)
#Extract metrics
r2_reg_5b   = res5b["r2"]
rmse_reg_5b = res5b["rmse"]
mae_reg_5b  = res5b["mae"]
print("\nModel 5b completed.")
print(f"  R²   = {r2_reg_5b:.3f}")
print(f"  RMSE = {rmse_reg_5b:.3f}")
print(f"  MAE  = {mae_reg_5b:.3f}")
#IT DID NOT IMPROVE MODEL 3 SO THAT IS NICE! THE SUBQUESTIONS ALONE DO NOT MAKE A GOOD MODEL
#%%-------------------------------------------------------------------------------------------------------------------------------------------
#The VIF showed that especially the MANSA items are very correlated at eachother AND we know that MANSA_totaal.1 
#is the strongest predictor. I want to see what happens to model5 if we remove the separate MANSA items and only leave the total in it
model5c_feats = demographic_vars + ["mansa_totaal.1"] + inspire_items + fr_items + honos_items_1
res5c = fit_eval_linear_model(
    model_name="Model 5c: Demographics + MANSA_totaal.1 + Inspire/FR/HoNOS items (no MANSA items)",
    features=model5c_feats,
    plot_filename="linear_regression_model_5c_actual_vs_predicted.png",
    design_prefix="model5c_design_matrix",)
r2_reg_5c   = res5c["r2"]
rmse_reg_5c = res5c["rmse"]
mae_reg_5c  = res5c["mae"]
print("\nModel 5c completed.")
print(f"  R²   = {r2_reg_5c:.3f}")
print(f"  RMSE = {rmse_reg_5c:.3f}")
print(f"  MAE  = {mae_reg_5c:.3f}")
#Also did not improve model 3 --> the totals still tell more than the separate
#%%-----------------------------------------------------------------------------------------------------------------------------
#Improve Model 5 by using subscales instead of all individual items
#Goal: Reduce overfitting, dimensionality and noise by grouping questionnaire items into meaningful subscores.
#This can improve generalizability in repeated GroupKFold cross-validation.
#The subscales:
#MANSA: 3 QoL domains (life, social, self) using ONLY Likert items
#HoNOS: symptom, social, behaviour
#INSPIRE: relationship, collaboration
#FR: coping, hope
df_mansa_9_15 = df_mansa_9_15.copy()
def mean_subscale(df, cols, max_missing_ratio=0.5):
    vals = df[cols].apply(pd.to_numeric, errors="coerce")  # force numeric
    allowed_missing = int(np.floor(len(cols) * max_missing_ratio))
    valid = vals.isna().sum(axis=1) <= allowed_missing
    return vals.mean(axis=1, skipna=True).where(valid, np.nan)
#MANSA scales (Likert only)
mansa_life_idx   = [i for i in [1, 2, 3, 4, 5, 6]    if i in mansa_likert_items]
mansa_social_idx = [i for i in [7, 8, 9, 10, 11]     if i in mansa_likert_items]
mansa_self_idx   = [i for i in [12, 13, 14, 15, 16]  if i in mansa_likert_items]
mansa_life_cols   = [f"MANSA_PH_{i}.1" for i in mansa_life_idx]
mansa_social_cols = [f"MANSA_PH_{i}.1" for i in mansa_social_idx]
mansa_self_cols   = [f"MANSA_PH_{i}.1" for i in mansa_self_idx]
df_mansa_9_15["mansa_life_1"]   = mean_subscale(df_mansa_9_15, mansa_life_cols)
df_mansa_9_15["mansa_social_1"] = mean_subscale(df_mansa_9_15, mansa_social_cols)
df_mansa_9_15["mansa_self_1"]   = mean_subscale(df_mansa_9_15, mansa_self_cols)
#HoNOS subscales
honos_psych_cols    = honos_items_1[0:4]
honos_social_cols   = honos_items_1[4:8]
honos_behavior_cols = honos_items_1[8:12]
df_mansa_9_15["honos_psych_1"]    = mean_subscale(df_mansa_9_15, honos_psych_cols)
df_mansa_9_15["honos_social_1"]   = mean_subscale(df_mansa_9_15, honos_social_cols)
df_mansa_9_15["honos_behavior_1"] = mean_subscale(df_mansa_9_15, honos_behavior_cols)
#INSPIRE subscales
inspire_relationship_cols = inspire_items[0:2]
inspire_collab_cols       = inspire_items[2:5]
df_mansa_9_15["inspire_relationship_1"] = mean_subscale(df_mansa_9_15, inspire_relationship_cols)
df_mansa_9_15["inspire_collab_1"]       = mean_subscale(df_mansa_9_15, inspire_collab_cols)
#FR subscales
fr_coping_cols = fr_items[0:1]   
fr_hope_cols   = fr_items[1:3]   
df_mansa_9_15["fr_coping_1"] = mean_subscale(df_mansa_9_15, fr_coping_cols)
df_mansa_9_15["fr_hope_1"]   = mean_subscale(df_mansa_9_15, fr_hope_cols)
print("Subscales added. New shape:", df_mansa_9_15.shape)
subscale_cols = [
    "mansa_life_1", "mansa_social_1", "mansa_self_1",
    "honos_psych_1", "honos_social_1", "honos_behavior_1",
    "inspire_relationship_1", "inspire_collab_1",
    "fr_coping_1", "fr_hope_1"]
print("Created subscale columns:", subscale_cols)
model5d_feats = demographic_vars + ["mansa_totaal.1"] + subscale_cols
res5d = fit_eval_linear_model(
    model_name="Model 5d: Demographics + MANSA_totaal.1 + subscales",
    features=model5d_feats,
    plot_filename="linear_regression_model_5d_actual_vs_predicted.png",
    design_prefix="model5d_design_matrix",)
r2_reg_5d   = res5d["r2"]
rmse_reg_5d = res5d["rmse"]
mae_reg_5d  = res5d["mae"]
print("\nModel 5d results:")
print(f"  R²   = {r2_reg_5d:.3f}")
print(f"  RMSE = {rmse_reg_5d:.3f}")
print(f"  MAE  = {mae_reg_5d:.3f}")
print(f"  mean_pred = {res5d['mean_pred']:.2f}")
#%%--------------------------------------------------------------------------------------------------------------
#Trying model 5E which is 5D but - separate MANSA items, as MANSA_totaal.1 is in it = high multicolinearity
subscale_cols_no_mansa = [
    "honos_psych_1", "honos_social_1", "honos_behavior_1",
    "inspire_relationship_1", "inspire_collab_1",
    "fr_coping_1", "fr_hope_1",]
model5e_feats = demographic_vars + ["mansa_totaal.1"] + subscale_cols_no_mansa
print("Model 5E features:", model5e_feats)
res5e = fit_eval_linear_model(
    model_name="Model 5E: Demographics + MANSA_totaal.1 + subscales (excl. MANSA)",
    features=model5e_feats,
    plot_filename="linear_regression_model_5e_actual_vs_predicted.png",
    design_prefix="model5e_design_matrix",)
r2_reg_5e    = res5e["r2"]
rmse_reg_5e  = res5e["rmse"]
mae_reg_5e   = res5e["mae"]
mean_pred_5e = res5e["mean_pred"]
print("\nModel 5E results:")
print(f"  R²   = {r2_reg_5e:.3f}")
print(f"  RMSE = {rmse_reg_5e:.3f}")
print(f"  MAE  = {mae_reg_5e:.3f}")
print(f"  mean_pred = {mean_pred_5e:.2f}")
#Although model 5E is performing the best, it is not outperforming model 3 in the linear regression.
#%%-------------------------------------------------------------------------------------------------------------
#Run same 10×5 repeated GroupKFold CV for LinearRegression, RF, XGB on model 5E
#Het beste van deze modellen nog in de beste linear SVR gooien, trying it for my own sanity
SEEDS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
N_REPEATS = 10
N_OUTER_SPLITS = 10
#Features: demographics + MANSA_totaal.1 + HoNOS/INSPIRE/FR subscales (geen MANSA-subscales)
svr5e_dir = os.path.join(subq2_dir, "svr_linear_model5e_rfe5")
out_dir = svr5e_dir
os.makedirs(out_dir, exist_ok=True)
importances_dir = os.path.join(out_dir, "selected_features_per_fold")
os.makedirs(importances_dir, exist_ok=True)
GROUP_COL = "Proefpersoonnummer"
ycol = "mansa_totaal.2"
#All valid rows
valid_idx = df_mansa_9_15.index[df_mansa_9_15[ycol].notna()].to_numpy()
groups_all = df_mansa_9_15.loc[valid_idx, GROUP_COL].to_numpy()
print("Using Model 5E feature set:", model5e_feats)
#Repeated GroupKFold (10 folds × 5 repeats)
def repeated_groupkfold_splits(valid_idx, groups_all, n_splits=10, n_repeats=10, seeds=None):
    unique_groups = np.unique(groups_all)
    for rep in range(1, n_repeats + 1):
        rep_seed = seeds[rep - 1]   # kies de juiste seed voor deze repeat
        rng = np.random.default_rng(rep_seed)
        # shuffle groepen
        shuffled_groups = rng.permutation(unique_groups)
        # maak mapping
        rank = {g: i for i, g in enumerate(shuffled_groups)}
        order = np.argsort([rank[g] for g in groups_all])
        vidx = valid_idx[order]
        garr = groups_all[order]
        cv = GroupKFold(n_splits=n_splits)
        for fold, (tr, te) in enumerate(cv.split(vidx, groups=garr), start=1):
            yield rep, fold, vidx[tr], vidx[te]
#Inner CV (group-aware)
inner_cv = GroupKFold(n_splits=5)
param_grid = {
    "svr__C":       [0.01, 0.1, 1, 10, 30],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    "svr__tol":     [1e-4, 1e-3],}
rows = []
best_params_list = []
SAVE_PREDS = True
np.random.seed(42)
#Outer loop
for rep, ofold, outer_train_idx, outer_test_idx in repeated_groupkfold_splits(
        valid_idx=valid_idx,
        groups_all=groups_all,
        n_splits=N_OUTER_SPLITS,
        n_repeats=N_REPEATS,
        seeds=SEEDS):

    seed = SEEDS[rep - 1]  


    np.save(
        os.path.join(out_dir, f"svr_model5e_rfe5_rep{rep:02d}_outer_train_idx_{ofold:02d}.npy"),
        outer_train_idx,
    )
    np.save(
        os.path.join(out_dir, f"svr_model5e_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy"),
        outer_test_idx,
    )

    X_train_raw = df_mansa_9_15.loc[outer_train_idx, model5e_feats].copy()
    X_test_raw  = df_mansa_9_15.loc[outer_test_idx,  model5e_feats].copy()
    y_train = df_mansa_9_15.loc[outer_train_idx, ycol].astype(float)
    y_test  = df_mansa_9_15.loc[outer_test_idx, ycol].astype(float)

    preproc = build_preprocessor_minmax(model5e_feats, random_state=seed)
    Xtr_mat = preproc.fit_transform(X_train_raw, y_train)
    Xte_mat = preproc.transform(X_test_raw)


    try:
        feat_names = preproc.get_feature_names_out()
    except:
        feat_names = [f"feat_{i}" for i in range(Xtr_mat.shape[1])]

    Xtr = pd.DataFrame(Xtr_mat, index=outer_train_idx, columns=feat_names)
    Xte = pd.DataFrame(Xte_mat, index=outer_test_idx,  columns=feat_names)

    rfe_est = LinearSVR(random_state=seed, max_iter=20000)
    rfe = RFE(estimator=rfe_est, n_features_to_select=5, step=1)
    rfe.fit(Xtr, y_train)
    top5_feats = Xtr.columns[rfe.support_].tolist()

    pd.DataFrame({"selected_features": top5_feats}).to_csv(
        os.path.join(
            importances_dir,
            f"svr_model5e_rfe5_rep{rep:02d}_fold{ofold:02d}_features.csv",
        ),
        index=False,
    )

    Xtr_sel = Xtr[top5_feats].copy()
    Xte_sel = Xte[top5_feats].copy()

    inner_groups = df_mansa_9_15.loc[outer_train_idx, GROUP_COL].to_numpy()

    pipe = Pipeline([("svr", LinearSVR(random_state=seed, max_iter=20000))])

    gcv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )

    t0 = time.time()
    gcv.fit(Xtr_sel, y_train, groups=inner_groups)
    best_model = gcv.best_estimator_

    y_pred = best_model.predict(Xte_sel)

    r2   = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae  = mean_absolute_error(y_test, y_pred)
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
    if SAVE_PREDS:
        pd.DataFrame({
            "rep": rep,
            "outer_fold": ofold,
            "y_true": y_test.values,
            "y_pred": y_pred,
        }).to_csv(
            os.path.join(out_dir, f"svr_model5e_rfe5_rep{rep:02d}_outer{ofold:02d}_preds.csv"),
            index=False,)
    print(
        f"[SVR Model5E RFE5 rep {rep} fold {ofold}] {time.time()-t0:.1f}s | "
        f"best={gcv.best_params_} | R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    gc.collect()
#Saving
res_df = pd.DataFrame(rows)
summ = res_df.agg({
    "R2":   ["mean", "std"],
    "RMSE": ["mean", "std"],
    "MAE":  ["mean", "std"],})
res_df.to_csv(
    os.path.join(out_dir, "svr_model5e_rfe5_repeated_nestedcv_folds.csv"),
    index=False,)
summ.to_csv(
    os.path.join(out_dir, "svr_model5e_rfe5_repeated_nestedcv_summary.csv"),)
pd.DataFrame(best_params_list).to_csv(
    os.path.join(out_dir, "svr_model5e_rfe5_repeated_bestparams_per_outer.csv"),
    index=False,)
excel_path = os.path.join(out_dir, "svr_model5e_rfe5_repeated_nestedcv_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
    res_df.to_excel(w, sheet_name="fold_metrics", index=False)
    summ.to_excel(w, sheet_name="summary")
    pd.DataFrame(best_params_list).to_excel(w, sheet_name="best_params", index=False)
#%%-------------------------------------------------------------------------------------------------------------------------------------------
#Summary tabel of all the OLS regression models (3 + 5(....E))
ols_summary = pd.DataFrame([
    {
        "Model": "Model 3 (baseline)",
        "R2": r2_reg_3,
        "RMSE": rmse_reg_3,
        "MAE": mae_reg_3},
    {
        "Model": "Model 5",
        "R2": r2_reg_5,
        "RMSE": rmse_reg_5,
        "MAE": mae_reg_5},
    {
        "Model": "Model 5b",
        "R2": r2_reg_5b,
        "RMSE": rmse_reg_5b,
        "MAE": mae_reg_5b},
    {
        "Model": "Model 5c",
        "R2": r2_reg_5c,
        "RMSE": rmse_reg_5c,
        "MAE": mae_reg_5c},
    {
        "Model": "Model 5d",
        "R2": r2_reg_5d,
        "RMSE": rmse_reg_5d,
        "MAE": mae_reg_5d},
    {
        "Model": "Model 5e",
        "R2": r2_reg_5e,
        "RMSE": rmse_reg_5e,
        "MAE": mae_reg_5e}])
#Rond netjes af
ols_summary = ols_summary.round(3)
print(ols_summary)
#Save to Excel
ols_summary_path = os.path.join(subq2_dir, "OLS_baseline_comparison_models3_5_5b_5c_5d_5e.xlsx")
ols_summary.to_excel(ols_summary_path, index=False)
print("Saved OLS baseline comparison to:", ols_summary_path)
#Visualize
order = ["Model 3 (baseline)", "Model 5", "Model 5b", "Model 5c", "Model 5d", "Model 5e"]
ols_summary["Model"] = pd.Categorical(ols_summary["Model"], categories=order, ordered=True)
ols_summary = ols_summary.sort_values("Model")
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=ols_summary, x="Model", y="R2")
plt.ylabel("R² (higher is better)")
plt.xlabel("")
plt.title("OLS regression baselines – comparison of R²")
plt.ylim(0, 1)  # eventueel iets van 0–1 voor context
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
ols_plot_path = os.path.join(subq2_dir, "OLS_baseline_comparison_R2_models3_5_5b_5c_5d_5e.png")
plt.savefig(ols_plot_path, dpi=300, bbox_inches="tight")
print("Saved OLS baseline R² plot to:", ols_plot_path)
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------------------------------------------------------------------------
#Summary tabel of nested CV model 3 + best performing model 5 nested CV 
nested_summary_dir = p("subquestion2")
os.makedirs(nested_summary_dir, exist_ok=True)
def load_nested_summary(path, model_name):
    df = pd.read_csv(path)
    idx_col = df.columns[0]          
    mean_row = df[df[idx_col] == "mean"].iloc[0]
    std_row  = df[df[idx_col] == "std"].iloc[0]
    return {
        "Model": model_name,
        "R2_mean":   mean_row["R2"],
        "R2_sd":     std_row["R2"],
        "RMSE_mean": mean_row["RMSE"],
        "RMSE_sd":   std_row["RMSE"],
        "MAE_mean":  mean_row["MAE"],
        "MAE_sd":    std_row["MAE"],}
#Model 3 (hoofdspecificatie)
model3_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
model3_path = os.path.join(model3_dir, "svr_linear_rfe5_repeated_nestedcv_summary.csv")
m3 = load_nested_summary(model3_path, "Model 3")
#Model 5E (best performing 5-variant)
model5e_dir = p("subquestion2", "svr_linear_model5e_rfe5")
model5e_path = os.path.join(model5e_dir, "svr_model5e_rfe5_repeated_nestedcv_summary.csv")
m5e = load_nested_summary(model5e_path, "Model 5E")
nested_compare = pd.DataFrame([m3, m5e]).round(3)
print(nested_compare)
out_path = os.path.join(nested_summary_dir, "NestedCV_comparison_Model3_vs_Model5E.xlsx")
nested_compare.to_excel(out_path, index=False)
print("Saved nested CV comparison to:", out_path)
#Plot alleen de mean R²
plt.figure(figsize=(6, 4))
plt.bar(nested_compare["Model"], nested_compare["R2_mean"])
plt.ylabel("Mean R² (nested CV)")
plt.title("Nested CV – Model 3 vs Model 5E")
plt.ylim(0, max(nested_compare["R2_mean"]) + 0.05)
plt.grid(axis="y", alpha=0.3)
plot_path = os.path.join(nested_summary_dir, "NestedCV_R2_Model3_vs_Model5E.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved R² plot to:", plot_path)
#%%--------------------------------------------------------------------------------------------
#Feature influence: SVR top-5 with 95% CI
coef_dir = p("cross_validation_results", "svr_linear_final_top5")
boot_ci_path = os.path.join(
    coef_dir,
    "svr_linear_final_top5_scaled_coefs_bootstrap_ci.csv")
boot_df = pd.read_csv(boot_ci_path)
#Kies de kolommen die je in de thesis-tabel wilt tonen
feat_table = (
    boot_df[["feature", "scaled_coef_mean", "ci_2_5", "ci_97_5", "abs_scaled_coef_mean"]]
    .sort_values("abs_scaled_coef_mean", ascending=False)
    .round(3))
print(feat_table)
feat_table_path = os.path.join(subq2_dir, "SVR_top5_feature_effects_with_95CI.xlsx")
feat_table.to_excel(feat_table_path, index=False)
print("Saved SVR top-5 feature effect table to:", feat_table_path)
#%%---------------------------------------------------------------------------------------------------------------------------
#Compact bootstrapping of top 5 features of model 5E
#Directories
svr5e_dir = p("subquestion2", "svr_linear_model5e_rfe5")
feat_dir  = os.path.join(svr5e_dir, "selected_features_per_fold")
coef_dir  = p("subquestion2", "model5e_feature_effects")
os.makedirs(coef_dir, exist_ok=True)
#Hyperparameters
best_df = pd.read_csv(
    os.path.join(svr5e_dir, "svr_model5e_rfe5_repeated_bestparams_per_outer.csv"))
mode_params = best_df.mode().iloc[0]
svr_kwargs = {
    "C":           mode_params["svr__C"],
    "epsilon":     mode_params["svr__epsilon"],
    "tol":         mode_params["svr__tol"],
    "random_state": 42,
    "max_iter":    20000,}
print("Final LinearSVR hyperparameters (Model 5E):", svr_kwargs)
#RFE top 5-features
feat_files = glob(os.path.join(feat_dir, "svr_model5e_rfe5_rep*_fold*_features.csv"))
if not feat_files:
    raise FileNotFoundError(f"Geen RFE-bestanden gevonden in: {feat_dir}")
freq = Counter()
for fpath in feat_files:
    feats = pd.read_csv(fpath)["selected_features"].dropna()
    freq.update(feats)
top5 = (
    pd.DataFrame({"feature": list(freq.keys()), "count": list(freq.values())})
    .sort_values("count", ascending=False)
    .head(5)["feature"]
    .tolist())
print("Model 5E RFE – chosen top 5-features:", top5)
#building design matrix
ycol = "mansa_totaal.2"
mask = df_mansa_9_15[ycol].notna()
full_idx = df_mansa_9_15.index[mask].to_numpy()
#Raw data
X_raw = df_mansa_9_15.loc[full_idx, model5e_feats].copy()
y_all = df_mansa_9_15.loc[full_idx, ycol].astype(float).copy()
#Preprocessing
preproc = build_preprocessor_minmax(model5e_feats, random_state = 42)
X_proc = preproc.fit_transform(X_raw, y_all)
try:
    feat_names = preproc.get_feature_names_out()
except Exception:
    feat_names = [f"feat_{i}" for i in range(X_proc.shape[1])]
X_full = pd.DataFrame(X_proc, index=full_idx, columns=feat_names)
y_vec = y_all.to_numpy()
missing_top5 = [f for f in top5 if f not in X_full.columns]
if missing_top5:
    raise ValueError(f"Top-5 features niet gevonden in design-matrix: {missing_top5}")
X = X_full[top5].copy()
print("Final Model 5E dataset for bootstrap:", X.shape, y_vec.shape)
#Bootstrapping
n_boot = 500
boot_coefs = np.zeros((n_boot, X.shape[1]))
rng = np.random.default_rng(42)
for b in range(n_boot):
    idx = rng.integers(0, len(X), len(X))
    X_b = X.iloc[idx]
    y_b = y_vec[idx]
    pipe_b = make_pipeline(LinearSVR(**svr_kwargs))
    pipe_b.fit(X_b, y_b)
    boot_coefs[b, :] = pipe_b.named_steps["linearsvr"].coef_
coef_mean = boot_coefs.mean(axis=0)
ci_low    = np.percentile(boot_coefs,  2.5, axis=0)
ci_high   = np.percentile(boot_coefs, 97.5, axis=0)
boot_df = (
    pd.DataFrame({
        "feature": X.columns,
        "scaled_coef_mean": coef_mean,
        "ci_2_5": ci_low,
        "ci_97_5": ci_high,
        "abs_scaled_coef_mean": np.abs(coef_mean),})
    .sort_values("abs_scaled_coef_mean", ascending=False))
boot_path = os.path.join(coef_dir, "model5e_scaled_coefs_bootstrap_ci.csv")
boot_df.to_csv(boot_path, index=False)
print("Saved bootstrap CIs →", boot_path)
#Visualization
plt.figure(figsize=(6, 8))
err_low  = boot_df["scaled_coef_mean"] - boot_df["ci_2_5"]
err_high = boot_df["ci_97_5"] - boot_df["scaled_coef_mean"]
plt.barh(
    boot_df["feature"],
    boot_df["scaled_coef_mean"],
    xerr=[err_low, err_high],
    capsize=3,)
plt.axvline(0, color="black", linewidth=1)
plt.gca().invert_yaxis()
plt.xlabel("Scaled coefficient (bootstrap mean)")
plt.title("Model 5E – Top-5 feature effects (95% CI)")
plt.tight_layout()
plot_path = os.path.join(coef_dir, "model5e_top5_feature_effects_CI.png")
plt.savefig(plot_path, dpi=300)
print("Saved Model 5E feature influence plot →", plot_path)
plt.show()
plt.close()
#%%-------------------------------------------------------------------------------------------------
#Subquestion 3 To what extent does predictive performance differ across demographic subgroups (e.g., gender, age)?
#Making a seperate folder with the results for subquestion 3
subq3_dir = p("subquestion3")
os.makedirs(subq3_dir, exist_ok=True)
#Helperfunction for the fairness computation
def rmse(y_t, y_p):
    return mean_squared_error(y_t, y_p) ** 0.5
metrics = {
    "R2": r2_score,
    "RMSE": rmse,
    "MAE": mean_absolute_error,}
def run_fairness_for_feature(fair_df, feature_col, name, prefix, subq3_dir, saveplots=True):
    feature_series = fair_df[feature_col]
    #The rows
    mask = feature_series.notna()
    y_true = fair_df.loc[mask, "y_true"]
    y_pred = fair_df.loc[mask, "y_pred"]
    sensitive = feature_series.loc[mask].astype("string")
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,)
    print(f"\nFairness by {name} ({prefix})")
    print(mf.by_group)
    print("Overall:")
    print(mf.overall)
    print(f"\n{name} ({prefix}) – differences (best vs worst group)")
    print(mf.difference())
    print(f"\n=== {name} ({prefix}) – ratios (worst / best group)")
    print(mf.ratio())
    if saveplots:
        ax = mf.by_group[["R2", "RMSE", "MAE"]].plot(
            kind="bar",
            subplots=True,
            layout=(1, 3),
            figsize=(12, 4),
            legend=False,
            title=[
                f"R² by {name} ({prefix})",
                f"RMSE by {name} ({prefix})",
                f"MAE by {name} ({prefix})",],
            rot=0,)
        plt.tight_layout()
        fname = f"fairness_{prefix}_{name.replace(' ', '_')}.png"
        plt.savefig(os.path.join(subq3_dir, fname), dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
    return mf
#%%-----------------------------------------------------------------------------------------------------------------------------------
#Using fairlearn on the different demographic subgroups on the best model (= model 3 with Linear SVR + RFE(5))
#Model3!
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
    idx_path = os.path.join(
        svr_rfe5_dir,
        f"svr_linear_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy")
    test_idx = np.load(idx_path)
    if len(test_idx) != len(df_pred):
        raise ValueError(f"Length mismatch rep {rep}, fold {ofold}")
    df_pred["rep"] = rep
    df_pred["outer_fold"] = ofold
    df_pred["idx"] = test_idx
    all_rows.append(df_pred)
preds_all = pd.concat(all_rows, ignore_index=True)
print("Model 3 fairness predictions loaded:", preds_all.shape)
#Merge with demographics
meta_cols = ["Age", "geslacht_GegevensAfname"]
if "leefsituatie.1" in df_mansa_9_15.columns:
    meta_cols.append("leefsituatie.1")
if "opleiding_nieuw" in df_mansa_9_15.columns:
    meta_cols.append("opleiding_nieuw")
meta = df_mansa_9_15[meta_cols].copy()
meta["idx"] = meta.index
fair_df = preds_all.merge(meta, on="idx", how="left")
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
        2: "Samenwonend met derden",
        3: "Onzelfstandig",
        4: "Dakloos/anders",}
    fair_df["leefsituatie_label"] = fair_df["leefsituatie.1"].map(leef_map)
#Education labels
if "opleiding_nieuw" in fair_df.columns:
    fair_df["opleiding_label"] = fair_df["opleiding_nieuw"]
#Fairness runs
mf_gender_3 = run_fairness_for_feature(
    fair_df, "geslacht_label", "gender", "model3", subq3_dir)
mf_age_3 = run_fairness_for_feature(
    fair_df, "age_group", "age group", "model3", subq3_dir)
if "leefsituatie_label" in fair_df.columns:
    mf_leef_3 = run_fairness_for_feature(
        fair_df, "leefsituatie_label", "living situation", "model3", subq3_dir)
if "opleiding_label" in fair_df.columns:
    mf_opleiding_3 = run_fairness_for_feature(
        fair_df, "opleiding_label", "education level", "model3", subq3_dir)
#%%-------------------------------------------------------------------------------------------
#Fairness for Model 5E (SVR + RFE5, Model 5E feature set)
model5e_dir = p("subquestion2", "svr_linear_model5e_rfe5")
pred_files = sorted(
    glob_module.glob(os.path.join(
        model5e_dir,
        "svr_model5e_rfe5_rep??_outer??_preds.csv")))
all_rows = []
pattern = re.compile(r".*rep(\d+)_outer(\d+)_preds\.csv$")
for fp in pred_files:
    m = pattern.match(fp)
    if not m:
        continue
    rep = int(m.group(1))
    ofold = int(m.group(2))
    df_pred = pd.read_csv(fp)
    idx_path = os.path.join(
        model5e_dir,
        f"svr_model5e_rfe5_rep{rep:02d}_outer_test_idx_{ofold:02d}.npy")
    test_idx = np.load(idx_path)
    if len(test_idx) != len(df_pred):
        raise ValueError(f"Mismatch in fold {rep}-{ofold}")
    df_pred["rep"] = rep
    df_pred["outer_fold"] = ofold
    df_pred["idx"] = test_idx
    all_rows.append(df_pred)
preds5e = pd.concat(all_rows, ignore_index=True)
print("Model 5E fairness predictions:", preds5e.shape)
#Merge with demographics
meta_cols = ["Age", "geslacht_GegevensAfname"]
if "leefsituatie.1" in df_mansa_9_15.columns:
    meta_cols.append("leefsituatie.1")
if "opleiding_nieuw" in df_mansa_9_15.columns:
    meta_cols.append("opleiding_nieuw")
meta = df_mansa_9_15[meta_cols].copy()
meta["idx"] = meta.index
fair5e = preds5e.merge(meta, on="idx", how="left")
fair5e = fair5e.dropna(subset=["y_true", "y_pred", "Age", "geslacht_GegevensAfname"])
#Age-groepen
fair5e["age_group"] = pd.cut(
    fair5e["Age"],
    bins=[17, 25, 55, 80],
    labels=["<25", "25–55", ">55"],
    include_lowest=True,)
#Sex labels
fair5e["geslacht_label"] = fair5e["geslacht_GegevensAfname"].map(geslacht_map)
#living situation labels
if "leefsituatie.1" in fair5e.columns:
    leef_map = {
        1: "Zelfstandig",
        2: "Samenwonend met derden",
        3: "Onzelfstandig",
        4: "Dakloos/anders",}
    fair5e["leefsituatie_label"] = fair5e["leefsituatie.1"].map(leef_map)
#Eduation labels
if "opleiding_nieuw" in fair5e.columns:
    fair5e["opleiding_label"] = fair5e["opleiding_nieuw"]
#Fairness-runs voor Model 5E
mf_gender_5e = run_fairness_for_feature(
    fair5e, "geslacht_label", "gender", "model5e", subq3_dir)
mf_age_5e = run_fairness_for_feature(
    fair5e, "age_group", "age group", "model5e", subq3_dir)
if "leefsituatie_label" in fair5e.columns:
    mf_leef_5e = run_fairness_for_feature(
        fair5e, "leefsituatie_label", "living situation", "model5e",subq3_dir)
if "opleiding_label" in fair5e.columns:
    mf_opleiding_5e = run_fairness_for_feature(
        fair5e, "opleiding_label", "education level", "model5e", subq3_dir)
#%%-------------------------------------------------------------------------------------
#Subquestion 4: To what extent does the best-performing model generalize to patients withfollow-up intervals outside the 9–15 month range?
#Making a seperate folder with the results for subquestion 4
#Making a different map for this subquestion
subq4_dir = p("subquestion 4")
os.makedirs(subq4_dir, exist_ok=True)
#%%-------------------------------------------------------------------------------------------------------------------------------------------
#Load best hyperparameters of model3
svr_dir = p("cross_validation_results", "svr_linear_cv_results.2_rfe5")
best_df = pd.read_excel(
    os.path.join(svr_dir, "svr_linear_rfe5_repeated_nestedcv_results.xlsx"),
    sheet_name="best_params",)
mode_params = best_df.mode(numeric_only=False).iloc[0].to_dict()
svr_kwargs = {
    k.replace("svr__", ""): v
    for k, v in mode_params.items()
    if k.startswith("svr__")}
svr_kwargs.setdefault("random_state", 42)
svr_kwargs.setdefault("max_iter", 20000)
print("Final Model 3 params (Subq4):", svr_kwargs)
ycol = "mansa_totaal.2"
#Train model 9-15
train_idx = df_mansa_9_15.index[df_mansa_9_15[ycol].notna()].to_numpy()
X_train_raw = df_mansa_9_15.loc[train_idx, model3_feats].copy()
X_train_raw = X_train_raw.replace({pd.NA: np.nan})
X_train_raw = X_train_raw.apply(pd.to_numeric, errors="coerce")
y_train = df_mansa_9_15.loc[train_idx, ycol].astype(float)
#MinMax-preprocessor
preproc = build_preprocessor_minmax(model3_feats, random_state = 42)
X_train_proc = preproc.fit_transform(X_train_raw, y_train)
try:
    feat_names = preproc.get_feature_names_out()
except Exception:
    feat_names = [f"feat_{i}" for i in range(X_train_proc.shape[1])]
X_train = pd.DataFrame(X_train_proc, index=train_idx, columns=feat_names)
#Applying RFE(5)
base_svr = LinearSVR(random_state=42, max_iter=20000)
rfe = RFE(estimator=base_svr, n_features_to_select=5, step=1)
X_train_sel = rfe.fit_transform(X_train, y_train)
selected_mask = rfe.support_
selected_feats = list(pd.Index(X_train.columns)[selected_mask])
print("Model 3 – final top-5 features (Subq4):", selected_feats)
#Final Linear SVR on subset 9-15
svr_final = LinearSVR(**svr_kwargs)
svr_final.fit(X_train_sel, y_train)
#Testset = only MANSA-paren outside 9–15 months
df_out = df_final[
    (df_final["mansa_totaal.1"].notna()) &
    (df_final["mansa_totaal.2"].notna()) &
    (~df_final["month_diff_1_and_2"].between(9, 15, inclusive="both"))].copy()
print("Out-of-range test set shape:", df_out.shape)
df_out = df_out.replace({pd.NA: np.nan})
X_test_raw = df_out[model3_feats].copy()
X_test_raw = X_test_raw.apply(pd.to_numeric, errors="coerce")
y_test = df_out[ycol].astype(float)
#Preprocessing on the wider set
X_test_proc = preproc.transform(X_test_raw)
X_test = pd.DataFrame(X_test_proc, index=df_out.index, columns=feat_names)
#RFE 5 features
X_test_sel = X_test[selected_feats].copy()
y_pred = svr_final.predict(X_test_sel)
#Performance outside 9-15 month range
r2_out = r2_score(y_test, y_pred)
rmse_out = mean_squared_error(y_test, y_pred) ** 0.5
mae_out = mean_absolute_error(y_test, y_pred)
print("\n Out-of-range performance (Model 3)")
print(f"R²   = {r2_out:.3f}")
print(f"RMSE = {rmse_out:.3f}")
print(f"MAE  = {mae_out:.3f}")
#Save the scaled dataset to excel 
full_out = X_test.copy()             
full_out["y_true"] = y_test.values
full_out["y_pred"] = y_pred
full_csv  = os.path.join(subq4_dir, "model3_out_of_range_design_after_MinMax_FULL.csv")
full_xlsx = os.path.join(subq4_dir, "model3_out_of_range_design_after_MinMax_FULL.xlsx")
full_out.to_csv(full_csv, index_label="index")
with pd.ExcelWriter(full_xlsx, engine="openpyxl") as w:
    full_out.to_excel(w, sheet_name="out_of_range_full", index=True)
#Scatterplot observed vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="black", linewidth=0.5)
lo = min(y_test.min(), y_pred.min())
hi = max(y_test.max(), y_pred.max())
plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
plt.xlabel("Observed MANSA_totaal.2 (out-of-range)")
plt.ylabel("Predicted MANSA_totaal.2")
plt.title("Model 3 – predictions outside 9–15 months")
plt.grid(alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(subq4_dir, "model3_out_of_range_actual_vs_predicted.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%------------------------------------------------------------------------------------------------------------------------------------------
#Seeing if there is a difference between 0-8 months and 15-54 months
def rmse(y_t, y_p):
    return mean_squared_error(y_t, y_p) ** 0.5
subq4_dir = p("subquestion 4")
df_subq4 = pd.read_csv(
    os.path.join(subq4_dir, "model3_out_of_range_design_after_MinMax_FULL.csv"))
#Splitting on month difference
df_short = df_subq4[df_subq4["num__month_diff_1_and_2"] < 0].copy() 
df_long  = df_subq4[df_subq4["num__month_diff_1_and_2"] > 1].copy()  
print("SHORT follow-up:", df_short.shape)
print("LONG  follow-up:", df_long.shape)
#Performance per groep
for name, df_part in [("SHORT (<9)", df_short), ("LONG (>15)", df_long)]:
    y_t = df_part["y_true"]
    y_p = df_part["y_pred"]
    print(f"\n{name}")
    print("  R²   =", r2_score(y_t, y_p))
    print("  RMSE =", rmse(y_t, y_p))
    print("  MAE  =", mean_absolute_error(y_t, y_p))
#Visualizing
df_short["residual"] = df_short["y_pred"] - df_short["y_true"]
df_long["residual"]  = df_long["y_pred"] - df_long["y_true"]
#Dataframe
box_df = pd.DataFrame({
    "residual": pd.concat([df_short["residual"], df_long["residual"]]),
    "group": (["<9 months"] * len(df_short)) + ([" >15 months"] * len(df_long))})
plt.figure(figsize=(10, 5))
sns.boxplot(data=box_df, x="group", y="residual", palette="pastel")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Model 3 – Residuals for short vs long follow-up intervals")
plt.xlabel("Follow-up group")
plt.ylabel("Residual (predicted − observed)")
plt.tight_layout()
fig_box = os.path.join(subq4_dir, "model3_residual_boxplots_short_vs_long.png")
plt.savefig(fig_box, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
#%%----------------------------------------------------------------------------------------------------------
