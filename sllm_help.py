import configparser
conf = configparser.ConfigParser()
conf.read('config.ini')

OPENAI_API_KEY=conf['openai']['api_key']  
PC_API_KEY=conf['pinecone']['api_key']  
PC_ENV=conf['pinecone']['environment'] 
SA_KEY =conf['serpapi']['serpapi_key']


from langchain.document_loaders import TextLoader
def load_doc(docpath):
    try:
        loader = TextLoader(docpath)
    except Exception as e:
        print(f'Caught {e}')
    return loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
def split_doc(doc):
    # langchain Document -> [langchain Document] 
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ' ', ''],
        chunk_size=400, 
        chunk_overlap=30)
    return text_splitter.split_documents(doc)
