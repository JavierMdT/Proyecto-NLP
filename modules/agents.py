from langchain_ollama import OllamaLLM
from typing import Callable
from typing import Any, Dict, List

######################### AGENTES #########################

# Inicializamos el modelo de lenguaje de forma global dentro del módulo de utilidades
llm = OllamaLLM(model="gemma2", temperature=0.0)

# Prompt retrieval
RETRIEVAL_DECISION_PROMPT = """Analyze the following news title and determine if it is necessary to retrieve additional context or descriptions from the database to accurately classify its topic, or if the title is completely self-explanatory.

News Title:
"{title}"

Respond strictly with a single digit:
- Write 1 if the title is ambiguous, incomplete, or absolutely requires additional context to determine its exact topic.
- Write 0 if the title is totally clear, explicit, and can be perfectly classified without reading anything else.

Response (0 or 1):"""

# Prompt reformular
REPHRASE_PROMPT = """
You are an expert search engine optimizer. Your task is to rephrase and expand the following news title into a clean query optimized for information retrieval (Vector Search and BM25). 

Extract key terms, remove fluff words, and add relevant synonyms or broader domain terms that are likely to appear in the news description.

Original Title:
"{title}"

Respond strictly with the rephrased query or keywords. Do not include any introductions, explanations, quotes, or markdown formatting.
Optimized Query:"""

# Prompt router
ROUTER_PROMPT = """
Analyze the following query state consisting of a news title and its retrieved context (if available). Your task is to select the most effective text retrieval/matching strategy for final classification.

State Data:
{state_text}

Choose the strategy based on these technical criteria:
- Choose 0 (Semantic) if the query relies heavily on abstract concepts, high-level context, or implicit meaning.
- Choose 1 (BM25) if the query contains highly specific keywords, unique names, exact product models, numbers, or acronyms.
- Choose 2 (Hybrid) if the query shares a mix of both conceptual ambiguity and specific keywords, requiring a balanced combination.

Respond strictly with a single digit: 0, 1, or 2. Do not include any introductions, explanations, or additional text.
Strategy Code:"""

# Prompt para clasificar 
CLASSIFICATION_PROMPT = """
You are an expert news classifier. Your task is to classify the following news item into one of these 4 categories:
1: World
2: Sports
3: Business
4: Sci/Tech

Analyze the title and the final context provided to make an accurate prediction.

Title:
"{title}"

Final Context:
{final_context}

Respond strictly with a single integer digit (1, 2, 3, or 4). Do not include any explanations, introduction, markdown formatting, or punctuation.
Category Code:"""

def agent_retrieval(title: str):
    """Agente de control que decide condicionalmente si aplicar RAG o no.

    Args:
        title (str): Consulta original (columna title de test).

    Returns:
        dict: Diccionario estructurado con el título y la decisión de control.
    """
    # Construir el prompt inyectando la consulta actual
    prompt = RETRIEVAL_DECISION_PROMPT.format(title=title)

    # Invocar al LLM y limpiar la salida para asegurar el formato numérico
    response = llm.invoke(prompt).strip()

    # Lógica condicional basada en la respuesta determinista (0 o 1)
    if "1" in response:
        # Retornamos el diccionario indicando que sí requiere proceso de recuperación
        return {"title": title, "necesita_recuperar": True}

    else:
        # No se requiere retrieval: marcamos la bandera de control como False
        return {"title": title, "necesita_recuperar": False}
       
def agent_rephraser(state_dict: dict):
    """Agente que toma el diccionario de estado y reformula el título para mejorar la búsqueda.

    Args:
        state_dict (dict): Diccionario de estado actual (contiene 'title' y opcionalmente 'retrieval').

    Returns:
        dict: El mismo diccionario de estado pero con la clave 'title' actualizada con la versión reformulada.
    """
    # Extraer el título original del diccionario de estado
    original_title = state_dict["title"]

    # Construir el prompt para el LLM
    prompt = REPHRASE_PROMPT.format(title=original_title)

    # Invocar al LLM y limpiar espacios en blanco sobrantes
    rephrased_title = llm.invoke(prompt).strip()

    # Actualizar el diccionario de estado mutando únicamente el valor de 'title'
    state_dict["title"] = rephrased_title

    return state_dict

def agent_router(state_dict):
    """Agente condicional que decide la estrategia de recuperación óptima (Semántico, BM25 o Híbrido).

    Args:
        state_dict (dict): Diccionario de estado actual (contiene 'title' y opcionalmente 'retrieval').

    Returns:
        dict: El mismo diccionario de estado con la nueva clave 'method' asignada (0, 1 o 2).
    """
    # Formatear los datos del estado actual para el texto del prompt
    state_text = f"Title: {state_dict['title']}"

    # Construir el prompt e invocar al LLM
    prompt = ROUTER_PROMPT.format(state_text=state_text)
    response = llm.invoke(prompt).strip()

    # Extraer el código numérico de la respuesta (por seguridad, buscamos el primer dígito válido)
    method_code = None
    for char in response:
        if char in ["0", "1", "2"]:
            method_code = int(char)
            break

    # Si por alguna razón el LLM falla en el formato, asignamos el método híbrido (2) por defecto
    if method_code is None:
        method_code = 2

    # Inyectar la decisión de control en el diccionario de estado con la clave 'method'
    state_dict["method"] = method_code

    return state_dict

def agent_execution_router(
    state_dict: Dict[str, Any],
    chroma_retriever: Any,
    bm25_retriever: Any,
    hybrid_retriever_fn: Any,
    k: int = 3) -> Dict[str, Any]:
    """
    Agente ejecutor que lee el estado de control y delega el flujo a la tool correcta.
    """
    
    # Por defecto híbrido si no existe
    method = state_dict.get("method", 2)  

    if method == 0:
        # Ejecuta la estrategia semántica
        return tool_semantic_retrieval(state_dict, chroma_retriever, k=k)
    elif method == 1:
        # Ejecuta la estrategia BM25
        return tool_bm25_retrieval(state_dict, bm25_retriever, k=k)
    elif method == 2:
        # Ejecuta la estrategia híbrida
        return tool_hybrid_retrieval(state_dict, hybrid_retriever_fn, k=k)

    return state_dict


######################### TOOLS #########################

def tool_semantic_retrieval(state_dict: Dict[str, Any], 
                            chroma_retriever: Any, 
                            k: int = 3) -> Dict[str, Any]:
    
    """
    Tool que ejecuta una búsqueda puramente semántica (Chroma) usando el título reformulado.
    """
    query = state_dict["title"]
    docs = chroma_retriever.invoke(query)[:k]
    # Guardamos o extendemos el contexto recuperado en el estado
    state_dict["final_context"] = " ".join([doc.page_content for doc in docs])
    return state_dict


def tool_bm25_retrieval(state_dict: Dict[str, Any]
                        ,bm25_retriever: Any, 
                        k: int = 3) -> Dict[str, Any]:
    """
    Tool que ejecuta una búsqueda puramente léxica (BM25) usando el título reformulado.
    """
    query = state_dict["title"]
    docs = bm25_retriever.invoke(query)[:k]
    state_dict["final_context"] = " ".join([doc.page_content for doc in docs])
    return state_dict


def tool_hybrid_retrieval(state_dict: Dict[str, Any], 
                          hybrid_retriever_fn: Any, 
                          k: int = 3) -> Dict[str, Any]:
    """
    Tool que ejecuta la búsqueda híbrida unificada (RRF) usando el título reformulado.
    """
    query = state_dict["title"]
    docs = hybrid_retriever_fn(query, k=k)
    state_dict["final_context"] = " ".join([doc.page_content for doc in docs])
    return state_dict


def tool_classifier(state_dict: Dict[str, Any]) -> int:
    """
    Tool externa de clasificación que genera la predicción final.

    Args:
        state_dict (dict): Diccionario con el estado completo del agente.

    Returns:
        int: Código numérico de la categoría predicha (1, 2, 3 o 4).
    """
    title = state_dict["title"]
    
    # Si el flujo decidió no recuperar información, el contexto estará vacío
    final_context = state_dict.get("final_context", "No additional context provided.")

    # Construir el prompt de clasificación inyectando las variables del estado
    prompt = CLASSIFICATION_PROMPT.format(title=title, final_context=final_context)

    # Invocar al LLM
    response = llm.invoke(prompt).strip()

    # Limpieza estricta para asegurar el retorno de un número entero válido
    predicted_class = None
    for char in response:
        if char in ["1", "2", "3", "4"]:
            predicted_class = int(char)
            break

    # Clase por defecto (por ejemplo, 1: World) si el modelo da una salida inesperada
    if predicted_class is None:
        predicted_class = 1

    return predicted_class