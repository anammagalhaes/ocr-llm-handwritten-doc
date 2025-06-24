import streamlit as st
import os
from dotenv import load_dotenv
from ocr_azure import realizar_ocr
from preprocess import limpar_texto, cortar_tokens
from openai_processing import processar_texto, gerar_resumo  
import openai
from langdetect import detect

# Carrega variáveis .env
load_dotenv() 

# Configura cliente Azure OpenAI
openai_client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-05-01-preview"
)

st.set_page_config(page_title="Leitor de Manuscritos", layout="centered")

st.title("📜 Leitor de Manuscritos com GPT")
st.write("Selecione um manuscrito da pasta para extrair, corrigir, resumir e interagir com o conteúdo.")

# Lista arquivos da pasta
manuscritos_path = "manuscritos"
extensoes = (".png", ".jpg", ".jpeg", ".jfif")
arquivos_disponiveis = [f for f in os.listdir(manuscritos_path) if f.lower().endswith(extensoes)]

arquivo_escolhido = st.selectbox("🗂️ Selecione um manuscrito:", arquivos_disponiveis)

if arquivo_escolhido:
    caminho_arquivo = os.path.join(manuscritos_path, arquivo_escolhido)
    st.image(caminho_arquivo, caption=arquivo_escolhido, use_container_width=True)

    if st.button("🚀 Processar Manuscrito"):
        try:
            with st.spinner("Executando OCR..."):
                texto_ocr = realizar_ocr(caminho_arquivo)

            st.text_area("📄 Texto extraído (OCR)", value=texto_ocr, height=200)

            texto_limpo = limpar_texto(texto_ocr)
            texto_final = cortar_tokens(texto_limpo)

            with st.spinner("🧠 Enviando para o GPT para correção..."):
                texto_corrigido = processar_texto(texto_final)

            st.text_area("✅ Texto corrigido", value=texto_corrigido, height=200)

            with st.spinner("📝 Gerando resumo do texto..."):
                resumo = gerar_resumo(texto_corrigido)

            st.text_area("🧾 Resumo do texto", value=resumo, height=200)

            st.session_state["texto_corrigido"] = texto_corrigido
            st.session_state["resumo_gerado"] = resumo
            st.success("🎉 Processamento concluído com sucesso!")

        except Exception as e:
            st.error(f"Erro ao processar o manuscrito: {e}")

# Chat interativo
if "texto_corrigido" in st.session_state:
    st.divider()
    st.subheader("💬 Pergunte sobre o texto ou peça um novo resumo")

    pergunta = st.text_input("Digite sua pergunta:")
    if pergunta:
        with st.spinner("Consultando o GPT..."):
            contexto = [
                {"role": "system", "content": "Você é um assistente que responde com base no seguinte texto:"},
                {"role": "user", "content": st.session_state["texto_corrigido"]},
                {"role": "user", "content": pergunta}
            ]

            try:
                resposta = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=contexto,
                    max_tokens=500,
                    temperature=0.2,
                    stream=False
                )

                resposta_gerada = resposta.choices[0].message.content

                if resposta_gerada:
                    st.markdown("**Resposta:**")
                    st.success(resposta_gerada)
                else:
                    st.warning("⚠️ Nenhuma resposta foi gerada. A pergunta pode estar fora do contexto do texto.")
            except Exception as e:
                st.error(f"❌ Erro ao responder: {e}")
