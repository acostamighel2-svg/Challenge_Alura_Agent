#  🤖 - Asistente Virtual para eventos

**Event Answers** es una aplicación web interactiva diseñada para responder preguntas de forma precisa sobre la agenda, logística y documentación de eventos oficiales, utilizando técnicas avanzadas de **RAG (Retrieval-Augmented Generation)**. 

La aplicación carga de manera automática una base de datos predeterminada y permite a los organizadores o usuarios interactuar con un agente inteligente que comprende el contexto y mantiene el historial de la conversación.

---

## 🏗️ Arquitectura de la Solución

El proyecto implementa una arquitectura basada en el patrón RAG para garantizar respuestas verídicas y libres de alucinaciones:

1. **Ingesta de Datos y Preprocesamiento:** Los documentos oficiales (en formato CSV) se cargan y procesan utilizando componentes de **LangChain**.
2. **Generación de Embeddings:** El texto procesado se convierte en vectores numéricos de alta densidad utilizando el modelo multilingüe de **Cohere**.
3. **Búsqueda Semántica (Retrieval):** Cuando el usuario realiza una pregunta, el sistema busca los fragmentos de información más relevantes y similares dentro del contexto del evento.
4. **Generación de Respuesta (Augmentation):** El modelo de lenguaje procesa la pregunta combinada con el contexto recuperado para formular una respuesta fluida, exacta y coherente.

---

## 🛠️ Tecnologías y Herramientas Utilizadas

* **Lenguaje:** Python
* **Interfaz de Usuario:** Streamlit (para el diseño del chat interactivo y la barra lateral).
* **Framework RAG:** LangChain (para la carga, segmentación y manejo de documentos).
* **Modelo de IA y Embeddings:** API de Cohere (utilizando modelos optimizados para contextos multilingües).
* **Manejo de Datos:** Pandas

---

## 🚀 Instrucciones para Ejecutar el Proyecto

Sigue estos pasos para clonar y ejecutar la aplicación en tu entorno local de forma segura:

### 1. Clonar el repositorio
```bash
git clone <URL_DE_TU_REPOSITORIO_GITHUB>
cd Challenge-Alura-Agente
```
### Paso 2: Configurar las Variables de Entorno (Secretos)
Por seguridad, las credenciales y llaves de API nunca deben subirse al repositorio. 

1. Crea una carpeta llamada `.streamlit` en la raíz de tu proyecto.
2. Dentro de esa carpeta, crea un archivo llamado `secrets.toml`.
3. Agrega tu clave de API de Cohere utilizando la siguiente estructura:
```toml
COHERE_API_KEY = "tu_api_key_real_de_cohere_aqui"
```
### Paso 3: Instalar las Dependencias

Asegúrate de tener un entorno de Python activo e instala los paquetes requeridos ejecutando el siguiente comando en la terminal:
```bash
pip install streamlit langchain langchain-community cohere pandas pypdf
```
### Paso 4: Ejecutar la Aplicación

Lanza el servidor local de Streamlit ejecutando el módulo principal desde la raíz de tu proyecto:
```bash
streamlit run app/main.py
```
Una vez ejecutado, el navegador abrirá automáticamente la interfaz web en la dirección http://localhost:8501

## 💡 Ejemplos de Interacción con el Agente

El agente está entrenado para responder preguntas con base en la documentación oficial disponible en data/documentacion_eventos.csv.

Ejemplos de Preguntas que el Agente puede Responder:

Ejemplos de Preguntas que el Agente puede Responder:

* ¿Cuándo se realiza VivaTech 2026?
* ¿VivaTech es accesible para personas con movilidad reducida?
* ¿Qué innovaciones se presentaron como estreno mundial en 2026?
* ¿Cómo compro una entrada para VivaTech?
* ¿Cuánto cuesta el pase de estudiante?



