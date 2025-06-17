import tensorflow as tf
import json
import os
import re
import spacy

nlp = spacy.load("es_core_news_sm")

class PreprocesadorPNL:
    def __init__(self):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
        self.titulos_unicos = []

    def cargar_datos(self, ruta):
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta}")

        with open(ruta, 'r', encoding='utf-8') as f:
            datos = json.load(f)

        # ✅ Convertir desde formato "intents" si aplica
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

        # ✅ Validar estructura de los datos
        if not all('frase' in d and 'titulo' in d for d in datos):
            raise ValueError("Cada entrada debe tener 'frase' y 'titulo'")

        frases = [self._preprocesar_texto(d['frase']) for d in datos]
        titulos = [d['titulo'] for d in datos]

        self.titulos_unicos = sorted(set(titulos))
        y = [self.titulos_unicos.index(t) for t in titulos]

        self.tokenizer.fit_on_texts(frases)
        X = self.tokenizer.texts_to_matrix(frases, mode='binary')
        return X, y

    def texto_a_vector(self, frase):
        frase_preprocesada = self._preprocesar_texto(frase)
        return self.tokenizer.texts_to_matrix([frase_preprocesada], mode='binary')

    def indice_a_titulo(self, indice):
        if 0 <= indice < len(self.titulos_unicos):
            return self.titulos_unicos[indice]
        return "desconocido"

    def guardar_tokenizer(self, path):
        tokenizer_json = self.tokenizer.to_json()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)

    def cargar_tokenizer(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo del tokenizer: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    def _preprocesar_texto(self, texto):
        texto = texto.lower().strip()
        texto = re.sub(r'[^\w\sáéíóúüñ]', '', texto)
        doc = nlp(texto)
        palabras = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(palabras)
