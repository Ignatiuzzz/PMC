# === MÓDULO DE IMPORTACIONES ===
import tensorflow as tf
import os
import numpy as np
import dateparser
import re
import spacy
import parsedatetime
from utils import PreprocesadorPNL
from datetime import datetime, timedelta

# === MÓDULO DE CONFIGURACIÓN DE NLP ===
nlp = spacy.load("es_core_news_sm")

DIAS_SEMANA = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6
}

# === CLASE PRINCIPAL DEL MODELO ===
class ModeloPNL:
    # --- Inicialización del modelo y preprocesador ---
    def __init__(self):
        self.modelo = None
        self.pre = PreprocesadorPNL()

    # --- Entrenamiento del modelo neuronal con los datos de intención ---
    def entrenar(self, ruta_datos, epochs=30):
        X, y = self.pre.cargar_datos(ruta_datos)
        X = np.array(X)
        y = np.array(y)

        self.modelo = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.pre.titulos_unicos), activation='softmax')
        ])

        self.modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.modelo.fit(X, y, epochs=epochs, batch_size=4)

        os.makedirs("modelos", exist_ok=True)
        self.modelo.save("modelos/modelo.keras")
        self.pre.guardar_tokenizer("modelos/tokenizer.json")

    # --- Carga del modelo y tokenizer entrenado ---
    def cargar_modelo(self):
        self.modelo = tf.keras.models.load_model("modelos/modelo.keras")
        self.pre.cargar_tokenizer("modelos/tokenizer.json")

    # --- Predicción general: categoría, fecha y título limpio ---
    def predecir(self, frase, umbral_confianza=0.5):
        if self.modelo is None:
            self.cargar_modelo()

        X = self.pre.texto_a_vector(frase)
        pred = self.modelo.predict(X)[0]
        index = pred.argmax()
        score = pred[index]
        categoria = self.pre.indice_a_titulo(index)

        fecha = self._interpretar_fecha(frase)
        if not fecha:
            raise ValueError("No se pudo interpretar la fecha.")

        titulo_modelo = self._extraer_titulo(frase, categoria)
        titulo_spacy = self._extraer_titulo_spacy(frase)

        if score >= umbral_confianza and len(titulo_modelo.split()) >= 2:
            titulo_final = titulo_modelo
        elif len(titulo_spacy.split()) >= 2:
            titulo_final = titulo_spacy
        else:
            titulo_final = titulo_modelo if score >= umbral_confianza else titulo_spacy

        # --- Eliminación forzada de am/pm y derivados del título ---
        titulo_final = re.sub(r'\b(a\.?m\.?|p\.?m\.?|am|pm)\b', '', titulo_final, flags=re.IGNORECASE).strip()

        return {
            'titulo': titulo_final,
            'fecha': fecha.isoformat()
        }

    # === MÓDULO DE INTERPRETACIÓN DE FECHAS ===
    def _interpretar_fecha(self, texto):
        cal = parsedatetime.Calendar()
        texto = texto.lower().strip()
        ahora = datetime.now()
        fecha_base = ahora

        # --- Detección de expresiones relativas ---
        if "pasado mañana" in texto:
            fecha_base += timedelta(days=2)
            texto = texto.replace("pasado mañana", "")
        elif "mañana" in texto:
            fecha_base += timedelta(days=1)
            texto = texto.replace("mañana", "")
        elif "hoy" in texto:
            fecha_base = ahora
            texto = texto.replace("hoy", "")
        elif "esta tarde" in texto:
            hora = 16
            texto = texto.replace("esta tarde", "")
        elif "esta noche" in texto:
            hora = 20
            texto = texto.replace("esta noche", "")

        # --- Extracción y normalización de la hora ---
        hora = 9
        minuto = 0
        hora_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.?|p\.m\.?|de la mañana|de la tarde|de la noche)?', texto)
        if hora_match:
            hora = int(hora_match.group(1))
            minuto = int(hora_match.group(2)) if hora_match.group(2) else 0
            periodo = hora_match.group(3)
            if periodo:
                periodo = periodo.replace(".", "")
            if periodo in ["pm", "de la tarde", "de la noche"] and hora < 12:
                hora += 12
            elif periodo in ["am", "de la mañana"] and hora == 12:
                hora = 0

        # --- Interpretación de día de la semana ---
        for dia_nombre, dia_index in DIAS_SEMANA.items():
            if dia_nombre in texto:
                hoy_idx = ahora.weekday()
                dias_para_siguiente = (dia_index - hoy_idx + 7) % 7
                dias_para_siguiente = dias_para_siguiente or 7
                futura_fecha = ahora + timedelta(days=dias_para_siguiente)
                return datetime(futura_fecha.year, futura_fecha.month, futura_fecha.day, hora, minuto)

        # --- Fallbacks con dateparser y parsedatetime ---
        fecha = dateparser.parse(
            texto,
            settings={
                'PREFER_DATES_FROM': 'future',
                'RETURN_AS_TIMEZONE_AWARE': False,
                'RELATIVE_BASE': fecha_base.replace(hour=0, minute=0, second=0, microsecond=0)
            },
            languages=['es']
        )
        if fecha:
            return datetime(fecha.year, fecha.month, fecha.day, hora, minuto)

        tiempo, status = cal.parseDT(texto, sourceTime=fecha_base)
        if status:
            return tiempo.replace(hour=hora, minute=minuto)

        return None

    # === MÓDULO DE EXTRACCIÓN DEL TÍTULO CON PATRONES PERSONALIZADOS ===
    def _extraer_titulo(self, frase, categoria):
        frase = frase.lower().strip()

        # --- Limpieza profunda de contexto y horario ---
        patrones_remover = [
            r'\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?|am|pm|de la mañana|de la tarde|de la noche)?\b',
            r'\bpor la mañana\b', r'\ben la mañana\b',
            r'\bpor la tarde\b', r'\ben la tarde\b',
            r'\bpor la noche\b', r'\ben la noche\b',
            r'\ba\.?m\.?\b', r'\bp\.?m\.?\b', r'\bam\b', r'\bpm\b',
            r'\b(hoy|mañana|pasado mañana|esta tarde|esta noche|lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo)\b',
            r'\b(a las|la|las|el|de|a)\b',
            r'\b(debo|tengo que|quiero|necesito|me toca|recuerda|recordar|no olvidar|por favor|debería|anotar|voy a)\b'
        ]
        for patron in patrones_remover:
            frase = re.sub(patron, '', frase)

        frase = re.sub(r'\s+', ' ', frase).strip()

        # --- Patrones por intención ---
        patrones = {
            "llamar": r"(llamar(?: a)? [\w\sáéíóú]+)",
            "estudiar": r"(estudiar(?: [\w\sáéíóú]+)?)",
            "reunión": r"(reunión(?: con)? [\w\sáéíóú]+)",
            "pasear": r"(sacar a pasear(?: [\w\sáéíóú]+)|pasear(?: [\w\sáéíóú]+))",
            "cita": r"(cita(?: con)? [\w\sáéíóú]+)",
            "examen": r"(examen(?: de)? [\w\sáéíóú]+)",
            "comprar": r"(comprar(?: [\w\sáéíóú]+)+)",
            "visitar": r"(visitar(?: a)? [\w\sáéíóú]+)",
            "ir": r"(ir(?: a)? [\w\sáéíóú]+)",
            "hacer": r"(hacer(?: [\w\sáéíóú]+)+)",
            "recordar": r"(recordar(?: [\w\sáéíóú]+)+)",
            "ver": r"(ver(?: [\w\sáéíóú]+)+)",
            "enviar": r"(enviar(?: [\w\sáéíóú]+)+)",
            "asistir": r"(asistir(?: a)? [\w\sáéíóú]+)",
            "buscar": r"(buscar(?: [\w\sáéíóú]+)+)",
            "leer": r"(leer(?: [\w\sáéíóú]+)+)",
            "terminar": r"(terminar(?: [\w\sáéíóú]+)+)",
            "entrenar": r"(entrenar(?: [\w\sáéíóú]+)?)",
            "recoger": r"(recoger(?: [\w\sáéíóú]+)+)",
            "entregar": r"(entregar(?: [\w\sáéíóú]+)+)",
            "subir": r"(subir(?: [\w\sáéíóú]+)+)",
            "bajar": r"(bajar(?: [\w\sáéíóú]+)+)",
            "preparar": r"(preparar(?: [\w\sáéíóú]+)+)",
            "organizar": r"(organizar(?: [\w\sáéíóú]+)+)",
            "conversar": r"(conversar(?: con)? [\w\sáéíóú]+)",
            "revisar": r"(revisar(?: [\w\sáéíóú]+)+)"
        }

        patron = patrones.get(categoria)
        if patron:
            match = re.search(patron, frase)
            if match:
                return match.group(1).strip()

        return frase.strip()

    # === MÓDULO DE EXTRACCIÓN DEL TÍTULO CON SPACY COMO RESPALDO ===
    def _extraer_titulo_spacy(self, frase):
        doc = nlp(frase)
        entidades_temporales = set()
        for ent in doc.ents:
            if ent.label_ in ("DATE", "TIME"):
                for token in ent:
                    entidades_temporales.add(token.text.lower())

        palabras_fecha = {
            "lunes", "martes", "miércoles", "miercoles", "jueves", "viernes",
            "sábado", "sabado", "domingo", "hoy", "mañana", "pasado", "pasado mañana",
            "esta", "tarde", "noche", "am", "pm", "de", "a", "las", "la", "el"
        }

        descartes = entidades_temporales.union(palabras_fecha)
        patron_hora = re.compile(r'^\d{1,2}(:\d{2})?$')

        tokens_utiles = []
        for token in doc:
            txt = token.text.lower()
            if txt in descartes or patron_hora.match(txt) or (txt.isdigit() and 0 <= int(txt) <= 24):
                continue
            tokens_utiles.append(token.text)

        return " ".join(tokens_utiles).strip()
