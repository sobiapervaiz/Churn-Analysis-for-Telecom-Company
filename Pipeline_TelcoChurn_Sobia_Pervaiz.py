import re
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_extraction.text import CountVectorizer

def pipline_Churn(df2, Train, scaling_data=True, Handling_Outliers=True,
             Handling_Missing_Values_Not_DT=True, Handling_Missing_Values_DT=False,
             encoding=True, is_training_data=True):
    """
    df2: dataframe to clean and prepare for modeling or deployment
    Train: dataframe used for training models (used to fit all transformers)

    Handling_Missing_Values_Not_DT: default True
        - Handling Missing Values for Non-Decision Tree Algorithms
        - For DT, Random Forest, Extra Trees, AdaBoost, XGBoost, LightGBM, CatBoost → set False
    Handling_Missing_Values_DT: default False
        - Handling Missing Values for Decision Tree Algorithms
        - For DT, Random Forest, Extra Trees, AdaBoost → set True

    is_training_data: default True
        - For test data or new application data → set False

    # For distance based algorithm and those rely on gradient descent based optimisation such as Linear Regression, Logistic Regression, SVM, Neural Networks:
    scaling_data=True,
    Handling_Outliers=True,
    Handling_Missing_Values_Not_DT=True,
    Handling_Missing_Values_DT=False,
    encoding=True

    # For DECISION TREES, RANDOM FOREST, and Extra Trees
    scaling_data=False,
    Handling_Outliers=False, #tree splits are not affected by outliers
    Handling_Missing_Values_Not_DT=False,
    Handling_Missing_Values_DT=True, ## DT, RF, Extra Trees do NOT handle NaN — sentinel fill (-9999999) used
    encoding=True

    # For ADABOOST
    scaling_data=False,
    Handling_Outliers=True, ## unlike RF/DT, AdaBoost is sensitive to outliers via sample reweighting
    Handling_Missing_Values_Not_DT=False,
    Handling_Missing_Values_DT=True,
    encoding=True

    # For XGBOOST and LIGHTGBM
    scaling_data=False,
    Handling_Outliers=False,
    Handling_Missing_Values_Not_DT=False,
    Handling_Missing_Values_DT=False,  # XGBOOST, LIGHTGBM, and CATBOOST handle missing values internally
    encoding=True

    # For CATBOOST
    scaling_data=False,
    Handling_Outliers=False,
    Handling_Missing_Values_Not_DT=False,
    Handling_Missing_Values_DT=False,  # XGBOOST, LIGHTGBM, and CATBOOST handle missing values internally
    encoding=False  # CATBOOST handle categorical internally
    """
    df = df2.copy()  # prevents unexpected side effects


    ##############################################################
    # 1. Duplicates & ID Columns
    ##############################################################
    df = df.drop_duplicates()

    # dropping features that are only for identification/irrelevant/may cause data leakage
    # safe drop — won't crash if columns are already absent (e.g. on test data)
    df.drop(columns=[c for c in ['CustomerID','Count', 'Country', 'State','Zip Code','Lat Long', 'Latitude','Longitude',  'Churn Label', 'Churn Reason', 'Churn_Score', 'Churn Score', 'CLTV'] if c in df.columns], inplace=True)

    ##############################################################
    # 2. Basic Cleaning in 'Total Charges'
    ##############################################################
    #Step1: replace ' ' with NaN
    df['Total Charges'].replace(' ',np.nan, inplace=True)
    
    # Step 2: replcae NaN with median of the columns
    df['Total Charges'].fillna(df['Total Charges'].median(), inplace=True)

    # Converts Total Charges to numeric before the train/test split because the raw file stores it as text.

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    Train["Total Charges"] = pd.to_numeric(Train["Total Charges"], errors="coerce")

    # Standardize text category labels by removing accidental spaces.
    cat_cols_clean = df.select_dtypes(exclude='number').columns

    for col in cat_cols_clean:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        Train[col] = Train[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Clip impossible negative values to 0.
    for col in ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]:
        df[col] = df[col].clip(lower=0)
        Train[col] = Train[col].clip(lower=0) # will convert all values less than 0 to 0
    

    ##############################################################
    # 3. Drop rows missing ALL values
    ##############################################################
    df.dropna(how='all', inplace=True)


    ##############################################################
    # 4. Missing Value Imputation
    ##############################################################
    if Handling_Missing_Values_Not_DT:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()

        # Categorical: fill with mode from Train
        for col in cat_cols:
            if df[col].isna().any():
                df[col + "_missing"] = df[col].isna().astype(int)
            mode_val = Train[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

        # Numeric: MICE — fit on Train only to avoid leakage
        mice = IterativeImputer(max_iter=10, random_state=42, min_value=0)

        if is_training_data:
            mice.fit(Train[numeric_cols])
            joblib.dump(mice, 'mice_imputer.pkl') #Save the trained state to disk
        else:
            mice = joblib.load('mice_imputer.pkl') # Load the exact model used during training

        # create missing indicators column-by-column BEFORE imputation
        for col in numeric_cols:
            if df[col].isna().any():
                df[col + "_missing"] = df[col].isna().astype(int)
        df[numeric_cols] = mice.transform(df[numeric_cols])

    if Handling_Missing_Values_DT:
        for col in df.select_dtypes(include='number').columns:
            if df[col].isna().any():
                df[col + "_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(9999999)

        for col in df.select_dtypes(exclude='number').columns:
            df[col] = df[col].fillna("Unknown")
     ##############################################################
    # 5. Outlier Handling (numeric columns only)
    ##############################################################
    if Handling_Outliers:
        # Iterate over df's numeric columns, excluding _missing indicators added in step 4
        # Outlier bounds (Q1, Q3, IQR) are always computed from Train to avoid leakage
        cols_to_check = [
            c for c in df.select_dtypes(include='number').columns
            if not c.endswith('_missing')
        ]

        for col in cols_to_check:
            Q1 = Train[col].quantile(0.25)
            Q3 = Train[col].quantile(0.75)
            IQR = Q3 - Q1
            LOF = Q1 - 3 * IQR  # Lower Outer Fence
            UOF = Q3 + 3 * IQR  # Upper Outer Fence
            df[col + "_flag"] = df[col].apply(lambda x: 1 if x < LOF or x > UOF else 0)
            df[col] = df[col].clip(lower=LOF, upper=UOF)

  

  

    ##############################################################
    # 6. Feature Engineering
    ##############################################################

    
    #**************Mapping of Binary Columns**************** 
    
    #### 'Gender': 'Male':0, 'Female':1
    df['Gender']=df['Gender'].map({'Male':0,
                    'Female':1})

    ### Senior Citizen: 'No':0, 'Yes':1
    df['Senior Citizen']=df['Senior Citizen'].map({'No':0,
                                                    'Yes':1})
    ### Partner: 'No':0, 'Yes':1
    df['Partner']=df['Partner'].map({'No':0, 
                                      'Yes':1})

    ### Dependents: 'No':0, 'Yes':1
    df['Dependents']=df['Dependents'].map({'No':0,
                                            'Yes':1})

    ### Phone Service: 'Yes':1,'No':0
    df['Phone Service']=df['Phone Service'].map({'No':0,
                           'Yes':1})

    ### Multiple Lines:'No':0,'Yes':1,'No phone service':2
    df['Multiple Lines']=df['Multiple Lines'].map({'No':0, 
                                                     'Yes':1,
                                                    'No phone service':2})

    ### Internet Service: 'No':0, 'Fiber optic':1, 'DSL':2
    df['Internet Service']=df['Internet Service'].map({'No': 0,
                                                'Fiber optic':1,
                                                'DSL':2})

    ### Online Security: 'No':0, 'Yes':1, 'No internet service':2
    df['Online Security']=df['Online Security'].map({'No':0,
                            'Yes':1,
                            'No internet service':2})

    ### 'Online Backup':'No':0, 'Yes':1,'No internet service':2
    df['Online Backup']=df['Online Backup'].map({'No':0,
                           'Yes':1,
                           'No internet service':2})

    ### Device Protection: 'No':0, 'Yes':1, 'No internet service':2
    df['Device Protection']=df['Device Protection'].map({'No':0,
                                                          'Yes':1,
                                                          'No internet service':2})

    ### Tech Support: 'No':0, 'Yes':1, 'No internet service':2
    df['Tech Support']=df['Tech Support'].map({'No':0,
                                                'Yes':1,
                                                'No internet service':2})

    ### Streaming TV:'No':0, 'Yes':1, 'No internet service':2
    df['Streaming TV']=df['Streaming TV'].map({'No':0,
                                               'Yes':1,
                                               'No internet service':2})

    ### Streaming Movies:'No':0, 'Yes':1, 'No internet service':2
    df['Streaming Movies']=df['Streaming Movies'].map({'No':0,
                                'Yes':1,
                                'No internet service':2})

    ### 	Contract: 'Month-to-month':0,'One year':1,'Two year':2
    df['Contract']=df['Contract'].map({'Month-to-month':0,
                                        'One year':1,
                                        'Two year':2})

    ### Paperless Billing: 'No':0, 'Yes':1
    df['Paperless Billing']=df['Paperless Billing'].map({'No':0,
                                                          'Yes':1})

    ### Payment Method: 'Mailed check':0,'Electronic check':1, 
    #'Bank transfer (automatic)':2,'Credit card (automatic)':3
    df['Payment Method']=df['Payment Method'].map({'Mailed check':0,
                                                     'Electronic check':1,
                                                    'Bank transfer (automatic)':2,
                                                   'Credit card (automatic)':3 })

    

    # Get the median Total charges for cities
    median_charges_city=df.groupby('City')['Total Charges'].transform('median')

    # Calculate Total Charges ratio relative to the City
    df['TotalCharge_CityRatio']=df['Total Charges']/median_charges_city

    # Average charge per tenure month and total-charge-to-CLTV ratio
    # are computed BEFORE log-transforming source columns.
    df["ACT"] = df["Total Charges"] / (df["Tenure Months"] + 1e-3)
    df["TCR"] = df["Total Charges"] / (df["CLTV"] + 1e-3)

    # Defensive cleanup
    df["ACT"] = df["ACT"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    df["TCR"] = df["TCR"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)

    # Log-transform ratio features
    df["ACT"] = np.log1p(df["ACT"])  # log1p(x)=log(x+1)
    df["TCR"] = np.log1p(df["TCR"])

    # Log-transform skewed monetary columns — log1p(x) = log(x+1), safe for 0 values
    for col in ["Monthly Charges", "Total Charges", "CLTV"]:
        df[col] = np.log1p(df[col])
 
     ##############################################################
    # 7. Scaling (numeric columns only)
    ##############################################################
    if scaling_data:
        # Exclude binary indicator/flag columns — scaling adds no benefit
        # and makes the values harder to interpret (0/1 becomes e.g. -0.5 / 2.3)
        numeric_cols = [
            c for c in df.select_dtypes(include='number').columns
            if not c.endswith('_missing') and not c.endswith('_flag')
        ]
        scaler = RobustScaler()  # RobustScaler uses Median & IQR — not sensitive to outliers

        if is_training_data:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            joblib.dump(scaler, 'scaler.pkl')
        else:
            scaler = joblib.load('scaler.pkl')
            df[numeric_cols] = scaler.transform(df[numeric_cols])

   

     ##############################################################
    # 8. Encoding (categorical columns only)
    ##############################################################
    # if encoding:
  
    #     from sklearn.preprocessing import TargetEncoder
    #     # 2. Setup your target encoding parameters
    #     te_cols = ['City'] 
    #     te = TargetEncoder(smooth="auto", cv=5)
    
    #     if is_training_data:
    #         # --- TRAINING PHASE ---
    #         te_array = te.fit_transform(df[te_cols], y)
    #         joblib.dump(te, 'te.pkl')
    #     else:
    #         # --- INFERENCE PHASE ---
    #         te = joblib.load('te.pkl')
    #         te_array = te.transform(df[te_cols])
    
    #     # 3. Rebuild the DataFrame with the encoded City column
    #     df_te = pd.DataFrame(te_array, columns=te_cols, index=df.index)
        
    #     # Drop the original string 'City' column and attach the encoded numeric version
    #     df = df.drop(columns=te_cols)
    #     df = pd.concat([df, df_te], axis=1)

    if encoding:
        
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()
        num_cols = df.select_dtypes(include='number').columns.tolist()


        # OHE is fitted on df's categorical columns after all prior cleaning steps
        # This is correct as long as is_training_data=True is called with the full training set
        # handle_unknown='ignore' silently zero-encodes unseen categories at inference —
        # intentional: new categories contribute nothing to the model output
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        if is_training_data:
            ohe_array = ohe.fit_transform(df[cat_cols])
            joblib.dump(ohe, 'ohe.pkl')

            
        else:
            ohe = joblib.load('ohe.pkl')
            ohe_array = ohe.transform(df[cat_cols])


        ohe_names = ohe.get_feature_names_out(cat_cols)
        df_cat = pd.DataFrame(ohe_array, columns=ohe_names, index=df.index)
        df = pd.concat([df[num_cols], df_cat], axis=1)
    ##############################################################
    # 9. Column Name Cleanup
    ##############################################################
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[<]", "lt", regex=True)
        .str.replace(r"[>]", "gt", regex=True)
        .str.replace(r"[+]", "plus", regex=True)
        .str.replace(" ", "_")
    )
    df.columns = [re.sub(r'.*?__', '', c) for c in df.columns]

    return df




"""
##############################################################
# USAGE GUIDE
##############################################################

# Separate train and test BEFORE calling dataprep
# Always separate features and target BEFORE dataprep — never pass Loan Status into the pipeline

X_train = df_train.drop(columns=['Loan Status'])
y_train = df_train['Loan Status']

X_test = df_test.drop(columns=['Loan Status'])
y_test = df_test['Loan Status']


# ── Logistic Regression / SVM / Neural Network ──────────────
X_train_clean = dataprep(
    df2=X_train,
    Train=X_train,         # Train = itself during training
    scaling_data=True,
    Handling_Outliers=True,
    Handling_Missing_Values_Not_DT=True,
    Handling_Missing_Values_DT=False,
    encoding=True,
    is_training_data=True
)
y_train_scaled = y_train.loc[X_train_scale.index]
model.fit(X_train_clean, y_train)
joblib.dump(model, 'model.pkl')

X_test_clean = dataprep(
    df2=X_test,
    Train=X_train,         # always pass the original training data — never refit on test
    scaling_data=True,
    Handling_Outliers=True,
    Handling_Missing_Values_Not_DT=True,
    Handling_Missing_Values_DT=False,
    encoding=True,
    is_training_data=False
)
model = joblib.load('model.pkl')

## If you used pd.get_dummies() instead of OHE — then yes, reindex is essential 
X_test_clean = X_test_clean.reindex(columns=X_train_clean.columns, fill_value=0)
predictions = model.predict(X_test_clean)


# ── Decision Trees / Random Forest / Extra Trees ─────────────
X_train_clean = dataprep(X_train, X_train,
    scaling_data=False, Handling_Outliers=False,
    Handling_Missing_Values_Not_DT=False, Handling_Missing_Values_DT=True,
    encoding=True, is_training_data=True)


# ── AdaBoost ─────────────────────────────────────────────────
X_train_clean = dataprep(X_train, X_train,
    scaling_data=False, Handling_Outliers=True,
    Handling_Missing_Values_Not_DT=False, Handling_Missing_Values_DT=True,
    encoding=True, is_training_data=True)


# ── XGBoost / LightGBM ───────────────────────────────────────
X_train_clean = dataprep(X_train, X_train,
    scaling_data=False, Handling_Outliers=False,
    Handling_Missing_Values_Not_DT=False, Handling_Missing_Values_DT=False,
    encoding=True, is_training_data=True)


# ── CatBoost ─────────────────────────────────────────────────
X_train_clean = dataprep(X_train, X_train,
    scaling_data=False, Handling_Outliers=False,
    Handling_Missing_Values_Not_DT=False, Handling_Missing_Values_DT=False,
    encoding=False,        # CatBoost handles categorical internally
    is_training_data=True)
"""