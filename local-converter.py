import os
import time
import logging
from queue import Queue
from threading import Thread
from moviepy.editor import VideoFileClip
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import ollama
import whisper

# Parámetros de configuración
PREFERRED_LANGUAGE = "es"  # Cambia a "en" para inglés, "es" para español, etc.
WHISPER_MODEL_SIZE = "medium"  # Tamaños posibles: tiny, base, small, medium, large
OLLAMA_URL = "http://localhost:11434/api/generate"  # URL de la API de Ollama local
# Configuración de Ollama (asume que Ollama está instalado y en ejecución local)
model_name = "llama3"  # Reemplaza con el modelo que uses
ollama = ollama.Client()  # Crea la instancia de Ollama

TARGET_NAME = "Miguel Ángel"  # Nombre a buscar en las transcripciones

# Configuración del registro
log_file = "process_log.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def is_file_fully_copied(file_path, check_interval=1, max_checks=10):
    previous_size = -1
    for _ in range(max_checks):
        try:
            current_size = os.path.getsize(file_path)
        except FileNotFoundError:
            logging.warning(f"Archivo no encontrado: {file_path}. Reintentando...")
            time.sleep(check_interval)
            continue

        if current_size == previous_size:
            return True
        previous_size = current_size
        time.sleep(check_interval)
    return False

def convert_to_mp3(file_path, output_dir):
    try:
        video = VideoFileClip(file_path)
        base_filename = os.path.basename(file_path).rsplit('.', 1)[0]
        mp3_path = os.path.join(output_dir, f"audio_{base_filename}.mp3")
        video.audio.write_audiofile(mp3_path)
        logging.info(f"Audio extraído a: {mp3_path}")
        return mp3_path
    except Exception as e:
        logging.error(f"Error al convertir {file_path} a MP3: {e}")
        return None

def generate_transcript(mp3_path, output_dir):
    try:
        # Detectar si hay una GPU disponible y cargar el modelo en el dispositivo adecuado
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        logging.info(f"Usando dispositivo: {device}")

        # Transcribir usando el modelo cargado
        result = model.transcribe(mp3_path)
        transcript_text = result["text"]  # Obtén el texto de la transcripción

        # Construir la ruta del archivo de transcripción manualmente
        base_filename = os.path.basename(mp3_path).rsplit('.', 1)[0]
        transcript_filename = f"{base_filename}.txt"
        transcript_path = os.path.join(output_dir, transcript_filename)

        # Guardar transcripción en el archivo
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        logging.info(f"Transcripción guardada en: {transcript_path}")
        return transcript_text, transcript_path

    except Exception as e:
        logging.error(f"Error al generar la transcripción para {mp3_path}: {e}")
        return None, None

def summarize_transcript(transcript_text, output_dir, base_filename):
    try:
        response = ollama.generate(
            model=model_name,
            prompt="Eres un asistente de IA diseñado para analizar transcripciones. Proporciona:\n"
                   f"- Un resumen conciso de la transcripción:\n{transcript_text}\n"
                   "- Puntos de acción principales (si los hay).\n"
                   "- Todas las menciones a un nombre específico: {TARGET_NAME}\n",
        )

        summary = response["response"]  # Obtén el texto del resumen
        summary_lines = summary.split('\n')
        formatted_summary = "\n".join([f"- {line}" for line in summary_lines if line.strip()])

        summary_filename_md = f"summary_{base_filename}.md"
        summary_filename_txt = f"summary_{base_filename}.txt"
        summary_path_md = os.path.join(output_dir, summary_filename_md)
        summary_path_txt = os.path.join(output_dir, summary_filename_txt)

        with open(summary_path_md, "w", encoding="utf-8") as f:
            f.write("# Resumen\n\n")
            f.write(formatted_summary)
        with open(summary_path_txt, "w", encoding="utf-8") as f:
            f.write(formatted_summary)

        logging.info(f"Resumen guardado en: {summary_path_md} y {summary_path_txt}")

    except Exception as e:
        logging.error(f"Error al resumir la transcripción: {e}")

def process_file(file_path):
    logging.info(f"Procesando archivo: {file_path}")
    while not is_file_fully_copied(file_path):
        logging.info(f"El archivo {file_path} no se ha copiado completamente, esperando...")
        time.sleep(5)
    base_filename = os.path.basename(file_path).rsplit('.', 1)[0]
    output_dir = os.path.join(os.path.dirname(file_path), f"transcripciones_{base_filename}")
    os.makedirs(output_dir, exist_ok=True)
    
    mp3_path = convert_to_mp3(file_path, output_dir)
    if mp3_path:
        transcript, transcript_path = generate_transcript(mp3_path, output_dir)
        if transcript:
            summarize_transcript(transcript, output_dir, base_filename)

class Watcher:
    DIRECTORY_TO_WATCH = "C:/test"  # Cambia esto por el directorio que quieres monitorear

    def __init__(self):
        self.observer = Observer()
        self.queue = Queue()
        self.worker_thread = Thread(target=self.worker)
        self.worker_thread.start()

    def run(self):
        event_handler = Handler(self.queue)
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
            logging.info("Observer Stopped")

        self.observer.join()
        self.queue.join()

    def worker(self):
        while True:
            file_path = self.queue.get()
            if file_path is None:
                break
            process_file(file_path)
            self.queue.task_done()

class Handler(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if file_path.endswith((".mp4", ".mkv")):
                logging.info(f"Nuevo archivo encolado: {file_path}")
                self.queue.put(file_path)

def test_summarize_with_sample_transcript():
    sample_transcript = "Este es un texto de prueba para generar un resumen y puntos de acción."
    base_filename = "sample_test"
    output_dir = "C:/test/transcripciones_sample_test"
    os.makedirs(output_dir, exist_ok=True)
    summarize_transcript(sample_transcript, output_dir, base_filename)

if __name__ == '__main__':
    os.makedirs("C:/test/transcripciones_sample_test", exist_ok=True)
    # Para pruebas, descomenta la línea siguiente y comenta las dos líneas siguientes
    # test_summarize_with_sample_transcript()
    w = Watcher()
    w.run()