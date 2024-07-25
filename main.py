from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    df["y"] = df[Config.CLASS_COL]
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return  df

def load_data_for_type3():
    df = get_input_data()
    df["y"] = df[Config.TYPE_COLS[2]]
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return  df

def load_data_for_type4():
    df = get_input_data()
    df["y"] = df[Config.TYPE_COLS[2]]
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    return model_predict(data, df, name)

def type_prediction(df):
    df = preprocess_data(df)
   
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(name)
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        type_score , predictions=perform_modelling(data, group_df, name)
        return type_score, predictions


if __name__ == '__main__':
    df=load_data()
    df_for_type3=load_data_for_type3()
    df_for_type4=load_data_for_type4()
    print("Prediction Type 2")
    type2 , prediction2=type_prediction(df)
    print("Prediction Type 3")
    type3 , prediction3=type_prediction(df_for_type3)
    print("Prediction Type 4")
    type4 , prediction4=type_prediction(df_for_type4)
    print(f"Prediction of type 2= {type2}")
    average=(type2+type3+type4)/3
    print(f"average = {average}")
    print(prediction2)
    
    
