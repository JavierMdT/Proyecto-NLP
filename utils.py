from tqdm import tqdm
import pandas as pd
from spacy.language import Language

def procesar_matrices(df:pd.DataFrame,
                      nlp:Language):
    '''
    Procesa los datos para generar las 4 tokenizaciones distintas.
    
    1. Tokenizacion simple 
    2. Tokenizacion sin puntuaciones
    3. Tokenizacion sin stopwords
    4. Tokenización sin ambas cosas
    
    Args:
        df: Dataframe a procesar
        nlp: Modelo tokenizador de spacy a utilizar 
    '''
    
    m1, m2, m3, m4 = [], [], [], []
    
    # nlp.pipe procesa los textos en lote de forma eficiente
    for doc in tqdm(nlp.pipe(df.astype(str), batch_size=500), total=len(df), desc="Tokenizando"):
        # 1. Solo tokenizado (separado por espacios)
        m1.append(" ".join([t.text.lower() for t in doc]))
        
        # 2. Sin puntuaciones
        m2.append(" ".join([t.text.lower() for t in doc if not t.is_punct]))
        
        # 3. Sin stopwords
        m3.append(" ".join([t.text.lower() for t in doc if not t.is_stop]))
        
        # 4. Sin puntuaciones ni stopwords
        m4.append(" ".join([t.text.lower() for t in doc if not t.is_punct and not t.is_stop]))
        
    return pd.Series(m1), pd.Series(m2), pd.Series(m3), pd.Series(m4)