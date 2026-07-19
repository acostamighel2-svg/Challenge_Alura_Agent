import streamlit as st
import os
from backend import responder_pregunta, procesar_documento, generar_embeddings_contexto, cargar_base_predeterminada

st.set_page_config(page_title="Asistente de Eventos", page_icon="🤖", layout="wide")

# CARGA AUTOMÁTICA DE LA BASE PREDETERMINADA
cargar_base_predeterminada()

if "nombre_evento" not in st.session_state:
    st.session_state.nombre_evento = "Documento No Cargado"

    #INTERFAZ GRÁFICA (UI)

# 1. BARRA LATERAL: Configuración y carga manual de otros eventos
with st.sidebar:
    st.header("⚙️ Configuración del Evento")
    st.write(f"**Base actual:** {st.session_state.nombre_evento}")
    st.markdown("---")

    # Permitir al usuario subir un nuevo archivo si desea cambiar de evento
    archivo_subido = st.file_uploader("Subir otra base de conocimiento (PDF o CSV)", type=["pdf", "csv"])

    if archivo_subido is not None:
        try:
            # Guardar temporalmente el archivo subido para procesarlo
            os.makedirs("temp", exist_ok=True)
            ruta_temporal = os.path.join("temp", archivo_subido.name)
            with open(ruta_temporal, "wb") as f:
                f.write(archivo_subido.getbuffer())

            with st.spinner("Procesando nuevo documento..."):
                bloques = procesar_documento(ruta_temporal)
                embeddings = generar_embeddings_contexto(bloques)

                # Actualizar las variables de la sesión
                st.session_state.bloques_evento = bloques
                st.session_state.embeddings_evento = embeddings

                # Definir el título limpio quitándole la extensión .csv o .pdf
                titulo_limpio = os.path.splitext(archivo_subido.name)[0].replace("_", " ").title()
                st.session_state.nombre_evento = titulo_limpio

            st.success(f"✅ ¡{titulo_limpio} cargado con éxito!")
            # Borrar el archivo temporal
            os.remove(ruta_temporal)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

# 2. CUERPO PRINCIPAL: Título Dinámico del Evento
st.title(f"🤖 Asistente Virtual para eventos: {st.session_state.nombre_evento}")
st.subheader("Hazme cualquier pregunta sobre el evento basado en la documentación oficial")

# 3. HISTORIAL DE CHAT: Inicializar si no existe
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {"role": "assistant",
         "content": f"¡Hola! Soy tu asistente para el evento **{st.session_state.nombre_evento}**. ¿En qué te puedo colaborar hoy?"}
    ]
# Mostrar los mensajes existentes en la interfaz
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.write(mensaje["content"])

# 4. CAPTURA Y PROCESAMIENTO DE PREGUNTAS
if pregunta_usuario := st.chat_input("Escribe tu pregunta aquí..."):

    # Mostrar la pregunta del usuario en pantalla inmediatamente
    with st.chat_message("user"):
        st.write(pregunta_usuario)

    # Guardar la pregunta en el historial
    st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})

    # Generar la respuesta usando el backend
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                respuesta_agente = responder_pregunta(pregunta_usuario)
                st.write(respuesta_agente)
            except Exception as e:
                respuesta_agente = f"Lo siento, ocurrió un error al procesar tu solicitud: {e}"
                st.error(respuesta_agente)

    # Guardar la respuesta del asistente en el historial
    st.session_state.mensajes.append({"role": "assistant", "content": respuesta_agente})
