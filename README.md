# psychosis-qol-thesis
This repository contains all code developed and used for the Master’s thesis:

“Predicting One-Year Follow-Up Outcomes in Patients with Psychotic Disorders.”

The project investigates the prediction of one-year follow-up quality-of-life outcomes in patients with psychotic disorders using supervised machine-learning methods. All analyses are conducted on the Altrecht Psychosis Prognosis Prediction (PPP) dataset.

The code implements a fully group-aware 10×10 repeated nested GroupKFold cross-validation framework, ensuring strict separation between training, validation, and test data at the patient level. The modelling pipeline includes data preprocessing, imputation, feature selection, and hyperparameter optimisation within the inner cross-validation loops. Multiple machine-learning models are evaluated and compared using out-of-sample performance metrics.

The primary outcome of interest is quality of life at follow-up, measured using the Manchester Short Assessment of Quality of Life (MANSA).

Running the code requires access to the encrypted Altrecht PPP dataset. Due to data confidentiality, ethical approval constraints, and privacy regulations, the dataset is not included in this repository.
