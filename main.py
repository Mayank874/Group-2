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
                                                                                   
def load_data_for_type3(prediction_from_type2):                                               #  Teking the predicting Values of type 2 in the type 3
    df = get_input_data()                                                  
    df["y2"]=prediction_from_type2
    df["y"] = df[Config.TYPE_COLS[1]]
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return  df

def load_data_for_type4(prediction_from_type3):                                               #  Teking the predicting Values of type 3 in the type 4
    df = get_input_data()
    df["y3"]=prediction_from_type3
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
    # Loading data for type 2 
    df=load_data()


    print("Prediction Type 2")    
    type2 , prediction2 =type_prediction(df)     # predicting the model for type 2 
    prediction2=pd.Series(prediction2)          # converting the predicting into series
    print(type2)
    
    print("Prediction Type 3")
    df_for_type3=load_data_for_type3(prediction2)         # predicting the model for type 3 
    type3 , prediction3=type_prediction(df_for_type3)     # converting the predicting into series
    prediction3=pd.Series(prediction3) 

    print("Prediction Type 4")
    df_for_type4=load_data_for_type4(prediction3)          # predicting the model for type 4
    type4 , prediction4=type_prediction(df_for_type4)      # converting the predicting into series
    
    
    average=(type2+type3+type4)/3
    print(f"Average Accuracy Achieved by the Model  = {average}")
    
