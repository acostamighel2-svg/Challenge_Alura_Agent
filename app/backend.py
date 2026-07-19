import os
import pandas as pd
import cohere
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

def obtener_cliente_cohere():
    """Inicializa de forma segura el cliente de Cohere usando st.secrets y lo guarda en la sesión."""
    if "co_client" not in st.session_state:
        try:
            api_key = st.secrets["COHERE_API_KEY"]
            st.session_state.co_client = cohere.Client(api_key)
        except KeyError:
            st.error("❌ No se encontró la clave 'COHERE_API_KEY' en los secretos de Streamlit.")
            return None
        except Exception as e:
            st.error(f"❌ Error al conectar con Cohere: {e}")
            return None

        return st.session_state.co_client

    def cargar_base_predeterminada():
        """Busca y procesa el archivo CSV por defecto usando rutas absolutas seguras."""
        if "bloques_evento" in st.session_state and st.session_state.bloques_evento:
            return

        # Obtiene la ruta de la carpeta raíz del proyecto ('event-answers') de forma dinámica
        ruta_actual = os.path.dirname(os.path.abspath(__file__))  # carpeta 'app'
        ruta_raiz = os.path.dirname(ruta_actual)  # carpeta 'event-answers'

        # Construye la ruta absoluta hacia 'data/documentacion_eventos.csv'
        ruta_predeterminada = os.path.join(ruta_raiz, "data", "documentacion_eventos.csv")

        if os.path.exists(ruta_predeterminada):
            try:
                # Procesamos el documento de forma silenciosa al arranque
                bloques = procesar_documento(ruta_predeterminada)
                embeddings = generar_embeddings_contexto(bloques)

                st.session_state.bloques_evento = bloques
                st.session_state.embeddings_evento = embeddings
                st.session_state.nombre_evento = "VivaTech 2026"
            except Exception as e:
                st.session_state.nombre_evento = "Error al Cargar Base"
                st.error(f"⚠️ Error al procesar el archivo automático: {e}")
        else:
            # Si no lo encuentra, dejamos el estado explícito para saber qué falló
            st.session_state.nombre_evento = "Documento No Encontrado"

    def procesar_documento(ruta_archivo):
        """Detecta la extensión del archivo y extrae su contenido."""
        extension = os.path.splitext(ruta_archivo)[1].lower()
        bloques_texto = []

        if extension == '.pdf':
            loader = PyPDFLoader(ruta_archivo)
            paginas = loader.load()
            for pagina in paginas:
                if pagina.page_content.strip():
                    bloques_texto.append(pagina.page_content.strip())

        elif extension == '.csv':
            df = pd.read_csv(ruta_archivo, encoding='latin1')
            if isinstance(df, pd.DataFrame):
                for index, fila in df.iterrows():
                    categoria = fila.get('Categoria', 'General')
                    pregunta = fila.get('Pregunta_Clave', '')
                    respuesta = fila.get('Respuesta_Oficial', '')

                    bloque = f"Categoría: {categoria} | Pregunta: {pregunta} | Respuesta: {respuesta}"
                    bloques_texto.append(bloque)
            else:
                raise ValueError("Error al procesar la estructura del archivo CSV.")
        else:
            raise ValueError("Formato de archivo no soportado. Por favor usa un archivo PDF o CSV.")

        return bloques_texto

    def generar_embeddings_contexto(bloques_contexto):
        """Calcula los embeddings de todos los bloques del documento."""
        co = obtener_cliente_cohere()
        if not co or not bloques_contexto:
            return None

        response_contexto = co.embed(
            texts=bloques_contexto,
            model="embed-multilingual-v3.0",
            input_type="search_document"
        )
        return response_contexto.embeddings

def buscar_respuesta_semantica(pregunta, bloques_contexto, embeddings_contexto):
    """Encuentra el bloque de datos más relevante basado en la pregunta."""
    co = obtener_cliente_cohere()
    if not co or not bloques_contexto or not embeddings_contexto:
        return None

    response_pregunta = co.embed(
        texts=[pregunta],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    embedding_pregunta = response_pregunta.embeddings[0]

    mejor_similitud = -1
    mejor_bloque = ""

    for bloque, embedding_doc in zip(bloques_contexto, embeddings_contexto):
        similitud = sum(p * d for p, d in zip(embedding_pregunta, embedding_doc))
        if similitud > mejor_similitud:
            mejor_similitud = similitud
            mejor_bloque = bloque

    if mejor_similitud > 0.3:
        return mejor_bloque
    return None

def generar_respuesta_conversacional(pregunta, contexto_recuperado, historial_cohere=[]):
    """Usa el LLM de Cohere pasándole el historial de la conversación para mantener el hilo."""
    co = obtener_cliente_cohere()
    if not co:
        return "Cliente de Cohere no inicializado."

    prompt_sistema = (
        "Eres un asistente de soporte virtual humano, simpático, directo y conciso para un evento tecnológico.\n\n"
        "REGLAS DE OBLIGATORIO CUMPLIMIENTO:\n"
        "1. SI EL USUARIO DA RESPUESTAS CORTAS, AFIRMACIONES, NEGACIONES O EXCLAMACIONES (ej: 'no', 'sí', 'ok', 'huaoo', 'gracias'): "
        "Mantén el hilo de la conversación de forma natural basándote en el historial de chat (ej: 'Entendido, ¿hay algo más en lo que te pueda colaborar?', '¡Perfecto! Si te surge alguna duda aquí estaré.'). NO inventes información del evento ni uses el contexto si no viene al caso.\n"
        "2. SI EL USUARIO HACE UNA PREGUNTA REAL SOBRE EL EVENTO: Responde usando EXCLUSIVAMENTE el Contexto de Referencia provisto. Sé directo y ve al grano.\n"
        "3. SI PREGUNTAN ALGO QUE NO ESTÁ EN EL CONTEXTO: Di amablemente que no cuentas con esa información específica en tus registros oficiales."
    )

    # Si no hay contexto relevante o la pregunta es un monosílabo/afirmación, ignoramos el contexto erróneo
    contexto_texto = contexto_recuperado if contexto_recuperado else "No hay contexto relevante para esta interacción."
    prompt_usuario = f"Contexto de Referencia:\n{contexto_texto}\n\nPregunta actual del Usuario: {pregunta}"

    try:
        # Le pasamos chat_history para que recuerde de qué venían hablando
        respuesta = co.chat(
            model="command-r-08-2024",
            message=prompt_usuario,
            preamble=prompt_sistema,
            chat_history=historial_cohere,
            temperature=0.2
        )
        return respuesta.text
    except Exception as e:
        return f"Error al generar la respuesta con el LLM de Cohere: {e}"