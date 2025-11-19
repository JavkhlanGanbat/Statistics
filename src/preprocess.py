from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Data cleaning

# We read all the data from the CSV file and drop the fnlwgt variable afterwards.
# income_>50K is the target, and every other variable is split between numerical and categorical

# Numerical: age = 67, capital-gain = 99999 etc.
# Using StandardScaler, we compute x = (x - μ) / σ where μ is the mean and σ is the standard deivation.
# We do this to avoid bias towards bigger numbers. Without scaling, something like capital gain = 99999 would
# influence the model more than age = 67, due to its larger absolute value.

# Categorical: education → ['HS-grad', 'Bachelors', 'Masters', 'Doctorate'] etc.  
# Using OneHotEncoder, for every variable with a finite number of possible categorical values, we convert them
# to binary vectors so that only the true value has the value 1, and all others have 0.
# Example: If a person has a Doctorate, then [education_HS-grad=0, education_Bachelors=0, education_Masters=0, education_Doctorate=1] 

def build_preprocessor():

    numeric_features = [
        "age", "educational-num", "capital-gain",
        "capital-loss", "hours-per-week"
    ]

    categorical_features = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race",
        "gender", "native-country"
    ]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
