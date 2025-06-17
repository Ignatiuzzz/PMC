from modelo_pnl import ModeloPNL

if __name__ == "__main__":
    modelo = ModeloPNL()
    modelo.entrenar("datos/ejemplos.json", epochs=150)
    print("✅ Entrenamiento finalizado y modelo guardado.")


modelo = ModeloPNL()
resultado = modelo.predecir("comprar regalo para mamá el sabado a las 10 de la noche")
print(resultado)
