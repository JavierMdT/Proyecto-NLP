from tqdm import tqdm
import pandas as pd
from spacy.language import Language
import torch
import torch.nn as nn

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




import torch
import torch.nn as nn


class TextClassifierRNN(nn.Module):
    """Recurrent Neural Network.

    Puede ser o LSTM o GRU en funcion del argumento del constructor.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        output_dim: int,
        mode: str,
        num_layers: int,
        embedding_dim: int,
        pretrained_weights: torch.Tensor = None,
    ):
        """Constructor de la clase.

        Args:
            vocab_size: tamaño del vocabulario de datos
            hidden_dim: dimension oculta
            output_dim: numero de clases
            mode: 'lstm' o 'gru'
            num_layers: numero de capas para la RNN
            embedding_dim: dimension del embedding
            pretrained_weights: tensor con los pesos preentrenados (GloVe/FastText)
        """
        super(TextClassifierRNN, self).__init__()

        self.mode = mode.lower()

        # Capa de embedding si es necesaria
        if pretrained_weights is not None:
            # Carga los pesos estáticos y evita que se actualicen en el entrenamiento
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=True
            )
        else:
            # Crea un embedding aleatorio que sí se entrenará desde cero
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Capa recurrente
        if self.mode == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        elif self.mode == "gru":
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError("El parámetro 'mode' debe ser 'lstm' o 'gru'")

        # Clasificador
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):

        # Solo si se especifica
        embedded = self.embedding(text)

        # RNN forward pass
        if self.mode == "lstm":
            output, (hidden, _) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)

        # Tomamos el último estado oculto del último paso temporal
        last_hidden = hidden[-1]

        # Clasificar
        return self.fc(last_hidden)