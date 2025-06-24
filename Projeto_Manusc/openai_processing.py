import os
import time
import openai
from dotenv import load_dotenv
import json
from langdetect import detect

load_dotenv()

# Configuração do cliente Azure OpenAI
client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-05-01-preview"
)

def gerar_prompt(texto):
    """
    Gera um prompt multilíngue para correção de OCR,
    mantendo obrigatoriamente o idioma original do texto.
    """
    return [
    {
        "role": "system",
        "content": (
            "You are an expert in OCR text correction and multilingual processing. "
            "Your job is to correct any OCR errors such as spelling or grammar mistakes, "
            "Preserve the original language of the text exactly as it was written."
            "Do not translate or mix languages — all parts of your output must stay in the original language. "
            "If the name and date are mentioned in the text, identify them."
        )
    },
    {
        "role": "user",
        "content": (
            f"Correct the following OCR text "
            f"Do NOT translate, do NOT summarize, do NOT rephrase:\n\n'''{texto}'''"
        )
    }
]


def processar_texto(texto_ocr, bloco_tamanho=200):
    """
    Processa o texto OCR completo. Caso o conteúdo seja extenso ou bloqueado,
    aplica o processamento em blocos menores. Mostra bloqueios de content filter.

    Args:
        texto_ocr (str): Texto bruto extraído por OCR.
        bloco_tamanho (int): Tamanho máximo de caracteres por bloco.

    Returns:
        str: Resultado do processamento (correção).
    """
    
    print(f"Tentando processar texto completo ({len(texto_ocr)} caracteres)...")

    try:
        messages = gerar_prompt(texto_ocr)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
            stream=False
        )

        resposta = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        if resposta and finish_reason != "content_filter":
            print("Texto completo processado com sucesso.")
            print(response.model_dump_json(indent=2))
            return resposta
        elif finish_reason == "content_filter":
            print("⚠️ BLOQUEADO PELO CONTENT FILTER na tentativa de texto completo.")
            print(response.model_dump_json(indent=2))
        else:
            print("⚠️ Texto truncado ou sem conteúdo válido. Iniciando processamento em blocos.")

    except Exception as e:
        print(f" Erro no processamento completo: {e}")
        print("Tentando processamento em blocos menores...")

    # Processamento em blocos
    blocos = [texto_ocr[i:i + bloco_tamanho] for i in range(0, len(texto_ocr), bloco_tamanho)]
    respostas_blocos = []

    for idx, bloco in enumerate(blocos):
        print(f"🔹 Processando bloco {idx + 1}/{len(blocos)}...")

        try:
            messages = gerar_prompt(bloco)
            response_bloco = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                stream=False
            )

            content = response_bloco.choices[0].message.content
            finish = response_bloco.choices[0].finish_reason

            if content and finish != "content_filter":
                respostas_blocos.append(content)
                print(f" Bloco {idx + 1} processado com sucesso.")
            elif finish == "content_filter":
                print(f" Bloco {idx + 1} bloqueado pelo content filter.")
            else:
                print(f" Bloco {idx + 1} não retornou conteúdo válido.")
        except Exception as e:
            print(f" Erro ao processar bloco {idx + 1}: {e}")
            continue

        time.sleep(0.5)  # Evita chamadas consecutivas rápidas

    return "\n".join(respostas_blocos)

def gerar_resumo(texto_corrigido):
    idioma = detect(texto_corrigido)

    idioma_full = {
        "en": "English",
        "pt": "Portuguese",
        "es": "Spanish",
        "fr": "French"
    }.get(idioma, "the original language")

    prompt = [
        {
            "role": "system",
            "content": (
                f"You are a multilingual assistant. The following text is written in {idioma_full}. "
                f"You must summarize it in {idioma_full}, and NEVER change the language. "
                f"Do not translate. Do not switch languages. Do not add any external information. "
                f"Just summarize what is written."
                f"Your summary must be based ONLY on the content of the input text. "
                f"Do NOT infer, imagine, complete or assume any missing information. "
                f"Do NOT use external knowledge or prior understanding of the topic. "
                f"Only use what is explicitly written in the text provided."
            )
        },
        {
            "role": "user",
            "content": f"'''{texto_corrigido}'''"
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            max_tokens=600,
            temperature=0.1,
            stream=False
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erro ao gerar resumo: {e}")
        return "Erro ao gerar resumo."