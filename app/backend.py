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
