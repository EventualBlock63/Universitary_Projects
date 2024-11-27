# Importación de librerías necesarias para el proyecto
import serial  # Para la comunicación serial
import time  # Para funciones de tiempo
from time import sleep  # Para pausas de tiempo específicas
import speech_recognition as sr  # Para reconocimiento de voz
import numpy as np  # Para operaciones numéricas y matrices
import librosa  # Para análisis de audio
import os  # Para operaciones de sistema y manipulación de archivos
import keyboard  #  Para hacer que un programa en Python espere a que se presione una tecla

# Inicializar el puerto serial para comunicación con dispositivos externos (ej. Arduino)
ser = serial.Serial('COM4', 9600)  # Puerto y tasa de baudios
recognizer = sr.Recognizer()  # Instancia de reconocimiento de voz

# Definición de funciones para manejo de comandos específicos
def girar_servomotor(grados):
    """
    Gira el servomotor a los grados especificados.

    :param grados: ángulo en grados para mover el servomotor
    """
    print(f"Girando Servomotor a {grados} grados")
    ser.write(str(grados).encode())  # Envía el ángulo al puerto serial

def cerrar_programa():
    """
    Cierra el programa al cambiar la variable de control 'x'.
    """
    global x
    print("Cerrando programa")
    x = 0  # Cambia la variable de control para salir del bucle principal

# Diccionario que mapea comandos de voz a funciones específicas
comandos = {
    "gira el servomotor a 45 grados": lambda: girar_servomotor(45),
    "gira el servomotor a 45°": lambda: girar_servomotor(45),
    "abre la puerta": lambda: girar_servomotor(90),
    "cierra la puerta": lambda: girar_servomotor(0),
    "cierra el programa": cerrar_programa
}

# Variable de control para el bucle while principal
x = 1

# Función para comparar si dos textos son equivalentes
def comparar_textos(texto1, texto2):
    """
    Compara dos textos, ignorando mayúsculas y espacios extra.

    :param texto1: primer texto para comparar
    :param texto2: segundo texto para comparar
    :return: True si los textos son iguales, False de lo contrario
    """
    return texto1.strip().lower() == texto2.strip().lower()

# Función para detectar el género de la voz basada en el pitch (frecuencia de tono)
def detectar_genero(audio_data, sample_rate=16000):
    """
    Detecta el género de la voz según el pitch promedio.

    :param audio_data: datos de audio de entrada
    :param sample_rate: frecuencia de muestreo
    :return: género detectado como "Hombre", "Mujer" o "Indeterminado"
    """
    pitch = librosa.pyin(audio_data, fmin=75, fmax=300)[0]  # Extrae el pitch
    pitch = pitch[np.logical_not(np.isnan(pitch))]  # Filtra valores no válidos
    avg_pitch = np.mean(pitch) if len(pitch) > 0 else None

    if avg_pitch:
        print(f"Pitch promedio detectado: {avg_pitch} Hz")
        if avg_pitch <= 200:
            return "Hombre"
        elif 200 < avg_pitch:
            return "Mujer"
        else:
            return "Indeterminado (pitch fuera del rango típico)"
    else:
        return "Indeterminado (no se pudo calcular el pitch)"

# Función para calcular MFCCs (Coeficientes Cepstrales en Frecuencia Mel)
def calcular_mfcc(audio_data, sample_rate=16000):
    """
    Calcula los MFCCs para el análisis de audio.

    :param audio_data: datos de audio
    :param sample_rate: frecuencia de muestreo
    :return: promedio de los coeficientes MFCCs
    """
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Función para comparar un audio con muestras guardadas en un directorio
def comparar_con_muestras(audio_generado, sample_rate_generado, directorio_muestras):
    """
    Compara un audio con muestras en un directorio para encontrar la mejor coincidencia.

    :param audio_generado: datos de audio del usuario
    :param sample_rate_generado: frecuencia de muestreo del audio generado
    :param directorio_muestras: directorio donde se almacenan las muestras .wav
    :return: mejor coincidencia y su distancia
    """
    mejor_coincidencia = None
    menor_distancia = float('inf')

    # Itera sobre los archivos en el directorio de muestras
    for archivo_muestra in os.listdir(directorio_muestras):
        if archivo_muestra.endswith(".wav"):
            nombre_persona = archivo_muestra.split("_")[0]  # Extrae nombre de la persona
            ruta_muestra = os.path.join(directorio_muestras, archivo_muestra)

            # Cargar la muestra de audio
            audio_muestra, sample_rate_muestra = librosa.load(ruta_muestra, sr=16000)

            # Calcular MFCCs para ambas muestras
            mfcc_generado = calcular_mfcc(audio_generado, sample_rate_generado)
            mfcc_muestra = calcular_mfcc(audio_muestra, sample_rate_muestra)

            # Calcula la distancia entre los MFCCs
            distancia = np.linalg.norm(mfcc_generado - mfcc_muestra)
            print(f"Comparando con {archivo_muestra}: distancia = {distancia}")

            # Actualiza la mejor coincidencia si la distancia es menor
            if distancia < menor_distancia:
                menor_distancia = distancia
                mejor_coincidencia = nombre_persona

    return mejor_coincidencia, menor_distancia

# Bucle principal que graba y procesa comandos de voz
while x == 1:
    print("Presiona cualquier tecla para continuar...")
    keyboard.read_event()  # Espera a que se presione una tecla
    print("5 segundos para grabar")
    time.sleep(5)
    with sr.Microphone() as source:
        print("Di algo:")
        try:
            # Graba el primer audio
            audio1 = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text1 = recognizer.recognize_google(audio1, language="es-ES")
            print("Texto reconocido: " + text1)

            # Guarda el primer audio
            with open("audio1.wav", "wb") as f:
                f.write(audio1.get_wav_data())

            # Graba el segundo audio para verificar coincidencia
            print("Repite lo que dijiste:")
            audio2 = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text2 = recognizer.recognize_google(audio2, language="es-ES")
            print("Texto reconocido (segunda grabación): " + text2)

            # Guarda el segundo audio
            with open("audio2.wav", "wb") as f:
                f.write(audio2.get_wav_data())

            # Verifica si los textos coinciden
            if comparar_textos(text1, text2):
                print("Las grabaciones son similares.")

                # Carga el audio y detecta el género de la voz
                audio_generado, sample_rate_generado = librosa.load("audio1.wav", sr=16000)
                genero = detectar_genero(audio_generado)
                print(f"Se detectó una voz de {genero}.")

                # Compara el audio con las muestras guardadas
                directorio_muestras = "muestras"
                mejor_coincidencia, distancia = comparar_con_muestras(audio_generado, sample_rate_generado, directorio_muestras)

                if mejor_coincidencia:
                    print(f"La mejor coincidencia es: {mejor_coincidencia} con una distancia de {distancia}.")
                else:
                    print("No se encontró ninguna coincidencia.")
            else:
                print("Las grabaciones no coinciden.")

            # Ejecuta comandos si el texto es un comando reconocido
            comando = text1.lower()
            if comando in comandos:
                comandos[comando]()
            else:
                print("Comando no reconocido. Inténtalo de nuevo.")

        except (TimeoutError, sr.UnknownValueError):
            print("No se reconoció el audio o tiempo de espera agotado, intenta de nuevo.")
            continue
        except sr.RequestError as e:
            print(f"Error en la solicitud: {e}")

# Cierra la conexión serial cuando el programa finaliza
ser.close()
