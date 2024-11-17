# Importación de librerías necesarias
import numpy as np  # Para manejar arrays numéricos y realizar operaciones matemáticas
import matplotlib.pyplot as plt  # Para graficar datos
import pyaudio  # Para grabar y reproducir audio
from scipy.fft import fft, fftfreq  # Para realizar la Transformada Rápida de Fourier (FFT) y calcular las frecuencias
from scipy.io import wavfile  # Para leer archivos de audio en formato WAV

# Función para grabar audio desde el micrófono
def grabar_audio():
    """
    Graba audio desde el micrófono durante un tiempo definido.
    Devuelve los datos de audio normalizados y la frecuencia de muestreo.
    
    Parámetros:
    Ninguno
    
    Retorna:
    - audio_data: np.array - Señal de audio grabada, normalizada.
    - RATE: int - Frecuencia de muestreo del audio grabado.
    """
    # Parámetros de configuración de la grabación
    CHUNK = 1024  # Tamaño del buffer de grabación
    FORMAT = pyaudio.paInt16  # Formato de grabación de 16 bits
    CHANNELS = 1  # Grabación en mono (un canal)
    RATE = 44100  # Frecuencia de muestreo (44.1 kHz)
    RECORD_SECONDS = 10  # Duración de la grabación en segundos

    # Inicializar PyAudio y el flujo de entrada de audio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Grabando...")

    frames = []  # Lista para almacenar los fragmentos de audio grabados

    # Bucle para capturar el audio en fragmentos (buffers)
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)  # Leer el buffer de audio
        frames.append(np.frombuffer(data, dtype=np.int16))  # Convertir el buffer en un array de numpy

    print("Grabación terminada.")
    
    # Cerrar el flujo de audio y terminar la sesión de PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convertir los fragmentos grabados en un solo array y normalizar los datos
    audio_data = np.hstack(frames).astype(np.float32)
    audio_data = audio_data / np.max(np.abs(audio_data))  # Normalizar la señal entre -1 y 1
    
    return audio_data, RATE  # Retornar los datos de audio y la frecuencia de muestreo

# Función para analizar un archivo de audio
def analizar_archivo_audio():
    """
    Pide el nombre de un archivo de audio en formato WAV, lee el archivo y normaliza los datos de la señal.
    Devuelve los datos de audio y la frecuencia de muestreo.
    
    Parámetros:
    Ninguno
    
    Retorna:
    - data: np.array - Señal de audio del archivo, normalizada.
    - sample_rate: int - Frecuencia de muestreo del archivo de audio.
    """
    # Solicitar el nombre del archivo sin la extensión .wav
    archivo = input("Ingrese el nombre del archivo de audio (sin '.wav'): ")
    archivo_wav = archivo + '.wav'  # Agregar la extensión .wav al nombre del archivo
    
    # Leer el archivo de audio WAV
    sample_rate, data = wavfile.read(archivo_wav)
    
    # Si el archivo es estéreo, seleccionar solo un canal
    if len(data.shape) == 2:
        data = data[:, 0]
    
    # Normalizar los datos de la señal de audio
    data = data / np.max(np.abs(data))
    
    return data, sample_rate  # Retornar los datos y la frecuencia de muestreo


# Función para calcular y graficar el oscilograma y el espectro de frecuencia
def graficar_analisis(data, sample_rate):
    """
    Calcula la Transformada Rápida de Fourier (FFT) de la señal de audio y grafica:
    - El oscilograma de la señal (amplitud en función del tiempo)
    - El espectro de magnitud en Volts RMS
    - El espectro de magnitud en voltaje pico.
    
    Parámetros:
    - data: np.array - Señal de audio (normalizada).
    - sample_rate: int - Frecuencia de muestreo del audio.
    
    Retorna:
    Ninguno (grafica los resultados).
    """
    n = len(data)  # Número total de muestras en la señal de audio
    frequencies = fftfreq(n, d=1/sample_rate)  # Calcular las frecuencias correspondientes a la FFT
    fft_magnitude = np.abs(fft(data))  # Calcular la magnitud de la FFT
    
    # Convertir la magnitud de la FFT a Volts RMS y voltaje pico
    fft_magnitude_rms = (fft_magnitude / n) / np.sqrt(2)
    fft_magnitude_peak = fft_magnitude / n

    # Tomar solo las frecuencias positivas (la FFT es simétrica)
    positive_freq_indices = np.where(frequencies >= 0)
    frequencies = frequencies[positive_freq_indices]
    fft_magnitude_rms = fft_magnitude_rms[positive_freq_indices]
    fft_magnitude_peak = fft_magnitude_peak[positive_freq_indices]

    # Crear el vector de tiempo para el oscilograma
    time = np.arange(n) / sample_rate

    # Graficar el oscilograma y el espectro de frecuencia
    plt.figure(figsize=(12, 9))  # Crear una figura de tamaño 12x9 pulgadas

    # Oscilograma de la señal de audio
    plt.subplot(3, 1, 1)
    plt.plot(time, data)
    plt.title('Oscilograma de la Señal de Audio')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (Normalizada)')
    plt.grid()

    # Espectro de magnitud (Volts RMS)
    plt.subplot(3, 1, 2)
    plt.plot(frequencies, fft_magnitude_rms)
    plt.title('Espectro de Magnitud de la Señal de Audio (Volts RMS)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (Volts RMS)')
    plt.grid()

    # Espectro de magnitud (Voltaje Pico)
    plt.subplot(3, 1, 3)
    plt.plot(frequencies, fft_magnitude_peak)
    plt.title('Espectro de Magnitud de la Señal de Audio (Voltaje Pico)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (Voltaje Pico)')
    plt.grid()

    # Ajustar el diseño de los gráficos y mostrar la figura
    plt.tight_layout()
    plt.show()

# Función de menú para elegir entre grabar/analizar o cargar un archivo de audio
def menu():
    """
    Muestra un menú interactivo para que el usuario seleccione:
    1. Analizar un archivo de audio WAV.
    2. Grabar audio desde el micrófono y analizar.
    3. Salir del programa.
    
    Retorna:
    Ninguno.
    """
    while True:
        # Mostrar opciones del menú
        print("Seleccione una opción:")
        print("1. Analizar archivo de audio")
        print("2. Grabar y analizar audio desde el micrófono")
        print("3. Salir")
        
        # Capturar la opción ingresada por el usuario
        opcion = input("Ingrese el número de su elección: ")

        if opcion == '1':
            # Analizar un archivo de audio WAV
            data, sample_rate = analizar_archivo_audio()
            graficar_analisis(data, sample_rate)
            break  # Salir del bucle
        elif opcion == '2':
            # Grabar y analizar audio desde el micrófono
            data, sample_rate = grabar_audio()
            graficar_analisis(data, sample_rate)
            break  # Salir del bucle
        elif opcion == '3':
            # Salir del programa
            print("Saliendo...")
            break
        else:
            # Mostrar mensaje de error si la opción es inválida
            print("Opción no válida, intente de nuevo.")

# Ejecutar el menú
menu()
