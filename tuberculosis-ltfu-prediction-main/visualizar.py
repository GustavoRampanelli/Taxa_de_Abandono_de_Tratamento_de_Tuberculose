import polars as pl

caminho_arquivo = "data/tuberculose_unificado.feather"

print("⏳ Carregando os dados (isso pode levar alguns segundos)...")
try:
    df = pl.read_ipc(caminho_arquivo)
    
    print("\n✅ Arquivo carregado com sucesso!")
    print(f"📊 Total de linhas: {df.height}")
    print(f"📉 Total de colunas: {df.width}")
    
    print("\n👀 Primeiras 5 linhas:")
    print(df.head(5))
    
    print("\n🔍 Informações sobre as colunas:")
    print(df.schema)
    
except Exception as e:
    print(f"❌ Erro ao ler o arquivo: {e}")
    print("O script de download já terminou de rodar? Lembre-se que o arquivo precisa estar completo.")
