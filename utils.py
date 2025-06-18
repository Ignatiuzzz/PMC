# === MÓDULO DE IMPORTACIONES NECESARIAS ===
import tensorflow as tf
import json
import os
import re
import spacy

# === CARGA DEL MODELO LINGÜÍSTICO EN ESPAÑOL (spaCy) ===
nlp = spacy.load("es_core_news_sm")

# === CLASE DE PREPROCESAMIENTO PARA EL MODELO DE NLP ===
class PreprocesadorPNL:
    def __init__(self):
        # Tokenizador de Keras con token especial para palabras fuera de vocabulario
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
        # Lista que contendrá los títulos únicos de las intenciones
        self.titulos_unicos = []

    # === FUNCIÓN PRINCIPAL PARA CARGAR Y PREPROCESAR LOS DATOS ===
    def cargar_datos(self, ruta):
        # Verifica si el archivo existe
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta}")

        # Carga del archivo JSON
        with open(ruta, 'r', encoding='utf-8') as f:
            datos = json.load(f)

        # ✅ Conversión desde el formato tipo chatbot {"intents": [...]} a lista plana
        if isinstance(datos, dict) and "intents" in datos:
            datos_transformados = []
            for intent in datos["intents"]:
                tag = intent.get("tag", "desconocido")
                for patron in intent.get("patterns", []):
                    datos_transformados.append({
                        "frase": patron,
                        "titulo": tag
                    })
            datos = datos_transformados

        # ✅ Validación: cada entrada debe tener 'frase' y 'titulo'
        if not all('frase' in d and 'titulo' in d for d in datos):
            raise ValueError("Cada entrada debe tener 'frase' y 'titulo'")

        # Preprocesamiento textual y extracción de etiquetas
        frases = [self._preprocesar_texto(d['frase']) for d in datos]
        titulos = [d['titulo'] for d in datos]

        # Almacena los títulos únicos (etiquetas de intención)
        self.titulos_unicos = sorted(set(titulos))
        # Crea los vectores de salida (clases)
        y = [self.titulos_unicos.index(t) for t in titulos]

        # Entrena el tokenizer con las frases preprocesadas
        self.tokenizer.fit_on_texts(frases)
        # Convierte el texto a vectores binarios
        X = self.tokenizer.texts_to_matrix(frases, mode='binary')

        return X, y

    # === CONVERSIÓN DE UNA FRASE A VECTOR PARA PREDICCIÓN ===
    def texto_a_vector(self, frase):
        frase_preprocesada = self._preprocesar_texto(frase)
        return self.tokenizer.texts_to_matrix([frase_preprocesada], mode='binary')

    # === CONVERSIÓN DE UN ÍNDICE A UNA CLASE/TÍTULO ===
    def indice_a_titulo(self, indice):
        if 0 <= indice < len(self.titulos_unicos):
            return self.titulos_unicos[indice]
        return "desconocido"

    # === GUARDAR EL TOKENIZER COMO JSON EN DISCO ===
    def guardar_tokenizer(self, path):
        tokenizer_json = self.tokenizer.to_json()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    # === CARGAR EL TOKENIZER DESDE UN ARCHIVO JSON ===
    def cargar_tokenizer(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo del tokenizer: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    # === PREPROCESAMIENTO BÁSICO: LIMPIEZA, LEMATIZACIÓN Y FILTRADO ===
    def _preprocesar_texto(self, texto):
        texto = texto.lower().strip()
        # Elimina signos de puntuación, conservando letras y números con tildes
        texto = re.sub(r'[^\w\sáéíóúüñ]', '', texto)
        doc = nlp(texto)
        # Lematiza y filtra stopwords/puntuaciones
        palabras = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(palabras)
