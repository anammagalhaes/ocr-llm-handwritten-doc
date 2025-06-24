import os
import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import cv2
from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("VISION_API_KEY")
ENDPOINT = os.getenv("VISION_ENDPOINT")

client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

def realizar_ocr(arquivo_path):
    if not os.path.exists(arquivo_path):
        raise FileNotFoundError(f"Arquivo {arquivo_path} não encontrado.")

    # Verifica imagem
    imagem = cv2.imread(arquivo_path)
    if imagem is None:
        raise ValueError(f"Arquivo {arquivo_path} está corrompido ou não pôde ser lido.")

    with open(arquivo_path, "rb") as img_file:
        ocr_response = client.read_in_stream(img_file, language="pt", raw=True)

    operation_id = ocr_response.headers["Operation-Location"].split("/")[-1]

    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ["notStarted", "running"]:
            break
        time.sleep(1)

    if result.status == "succeeded":
        texto = "\n".join([line.text for page in result.analyze_result.read_results for line in page.lines])
        return texto
    else:
        raise Exception(f"OCR falhou com status: {result.status}")
