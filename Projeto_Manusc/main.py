from dotenv import load_dotenv
load_dotenv()

import os
import glob
import openai 
from ocr_azure import realizar_ocr
from preprocess import limpar_texto, cortar_tokens
from openai_processing import processar_texto
from pptx import Presentation
from ppt import inserir_resultado_no_ppt


def encontrar_arquivo(nome):
    padrao = os.path.join("manuscritos", f"{nome}.*")
    arquivos = glob.glob(padrao)
    if arquivos:
        return arquivos[0]
    return None

def iniciar_chat_com_texto(texto_base):
    print("\n Você pode fazer perguntas sobre o texto.")
    print("Digite 'sair' para encerrar o chat.\n")

    contexto = [
        {"role": "system", "content": "Você é um assistente que responde com base neste texto:"},
        {"role": "user", "content": texto_base}
    ]

    while True:
        pergunta = input(" Você: ")
        if pergunta.lower() in ["sair", "exit"]:
            print("Encerrando o chat.")
            break

        contexto.append({"role": "user", "content": pergunta})

        try:
            resposta = openai.AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2024-05-01-preview"
            ).chat.completions.create(
                model="gpt-4o",
                messages=contexto,
                max_tokens=800,
                temperature=0.7,
                stream=False
            )

            msg = resposta.choices[0].message.content
            print(f"Assistente: {msg}\n")
            contexto.append({"role": "assistant", "content": msg})

        except Exception as e:
            print(f"Erro ao gerar resposta: {e}")

def main():
    nome = input("Nome da pessoa (ex: francisco): ").strip().lower()
    manuscrito_path = encontrar_arquivo(nome)

    if not manuscrito_path:
        print("Documento não encontrado.")
        return

    print(f"\n Fazendo OCR em: {manuscrito_path}")
    texto_extraido = realizar_ocr(manuscrito_path)

    print("\n Limpando texto...")
    texto_limpo = limpar_texto(texto_extraido)
    texto_final = cortar_tokens(texto_limpo)

    print("\n Enviando para o GPT (Azure)...")
    resultado = processar_texto(texto_final)

    print("\n Resultado final:\n")
    print(resultado)

    # Iniciar o chat interativo
    iniciar_chat_com_texto(resultado)
    
    # Inserir resultado no PowerPoint
    template_path = r"C:\Users\Ana\Downloads\Audit Report Template - Standard Version_vf.pptx"
    inserir_resultado_no_ppt(resultado=resultado, nome_usuario=nome, template_path=template_path)


if __name__ == "__main__":
    main()
