from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Өгөгдөл цэвэрлэх
    # CSV файлаас өгөгдлийг уншаад дараа нь fnlwgt хувьсагчийг хасна.
    # income_>50K нь зорилтот хувьсагч бөгөөд бусад хувьсагчийг тоон болон
    # чанарын гэж хоёр ангилна.

# Тоон хувьсагчид: age = 67, capital-gain = 99999 гэх мэт.
    # StandardScaler ашиглан x = (x - μ) / σ томьёогоор хувиргана. Энд μ нь дундаж,
    # σ нь стандарт хазайлт.
    # Ингэснээр аливаа хувьсагчийн абсолют утга нь том байсан ч эцсийн үр дүнд хэт их нөлөөлөхгүй. 
    # Үгүй бол age = 67-тэй харьцуулахад capital-gain = 99999 нь
    # абсолют утга ихтэй тул модельд илүү нөлөөтэй байж магадгүй.

# Чанарын хувьсагчид: education → ['HS-grad', 'Bachelors', 'Masters', 'Doctorate'] гэх мэт.
    # OneHotEncoder ашиглан боломжит чанарын утгуудыг бүгдийг нь хоёртын вектор болгон хувиргана.
    # Ингэхдээ тухайн мөрт тохирох утга 1, бусад нь 0 байна.
    # Жишээлбэл Doctorate зэрэгтэй бол
    # [education_HS-grad=0, education_Bachelors=0, education_Masters=0, education_Doctorate=1]

# CSV файл дотор байсан мөр дараах хэлбэрт орно
    # [ 2.231, 2.400, 13.366, -0.218, 1.667,
    #   1,0,0,  0,0,0,1,  0,1,0,  1,0,0,  0,1,0,  1,0,0,  1,0,  1,0,0 ]

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
