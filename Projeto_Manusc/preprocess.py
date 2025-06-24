import re
import tiktoken

def limpar_texto(texto):
    texto = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s.,!?'-]", "", texto)
    texto = re.sub(r'\b(?:sex|nude|touch|kiss|love|inside|hole|hot|warm|wild|wet)\b', '[...]', texto, flags=re.IGNORECASE)
    return texto

def cortar_tokens(texto, modelo="text-embedding-ada-002", limite=8191):
    tokenizer = tiktoken.encoding_for_model(modelo)
    tokens = tokenizer.encode(texto)

    if len(tokens) > limite:
        tokens = tokens[:limite]

    return tokenizer.decode(tokens)
