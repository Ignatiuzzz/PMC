from flask import Flask, request, jsonify
from flask_cors import CORS  # 👈 Importa esto
from modelo_pnl import ModeloPNL

app = Flask(__name__)
CORS(app)  # 👈 Habilita CORS para todas las rutas

modelo = ModeloPNL()

@app.route('/procesar', methods=['POST'])
def procesar():
    datos = request.json
    frase = datos.get('frase', '')

    try:
        resultado = modelo.predecir(frase)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'mensaje': '🧠 Servicio PNL activo'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
