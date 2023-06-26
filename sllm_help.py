
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import HttpResponseError
# https://learn.microsoft.com/en-us/azure/key-vault/secrets/quick-create-python?tabs=azure-cli

def get_azure_secrets():
    keyVaultName = 'kv-artodev'
    kvUrl = f'https://{keyVaultName}.vault.azure.net'
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=kvUrl, credential=credential)
    try:
        OPENAI_API_KEY=client.get_secret('openai-api-key').value
        PC_API_KEY=client.get_secret('pinecone-api-key').value
        PC_ENV=client.get_secret('pinecone-environment').value
        G_KEY=client.get_secret('google-api-key').value
        G_CSE=client.get_secret('google-cse-id').value
    except HttpResponseError as e:
        print('Failed to retrieve.')
        print(e.__str__())
        raise e
    sec ={'openai_api_key': OPENAI_API_KEY,
          'pc_api_key': PC_API_KEY,
          'pc_env': PC_ENV,
          'g_api_key': G_KEY,
          'g_cse': G_CSE
          }
    return sec

import configparser
def get_local_secrets():
    conf = configparser.ConfigParser()
    conf.read('config.ini')

    OPENAI_API_KEY=conf['openai']['api_key']  
    PC_API_KEY=conf['pinecone']['api_key']  
    PC_ENV=conf['pinecone']['environment'] 
    SA_KEY =conf['serpapi']['serpapi_key']
    G_KEY =conf['google']['google_api_key'] 
    G_CSE =conf['google']['google_cse_id'] 
    sec ={'openai_api_key': OPENAI_API_KEY,
          'pc_api_key': PC_API_KEY,
          'pc_env': PC_ENV,
          'g_api_key': G_KEY,
          'g_cse': G_CSE
          }
    return sec

sec = get_azure_secrets()
#sec = get_local_secrets()
index_name = 'langchain-retrieval-agent'

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

import os
def create_dir_if_not_exists(dir):
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)

def confirm_pickle_folder():
    if os.path.exists('./pickles'):
        pass
    else:
        os.makedirs('./pickles')

import json
class MyKnowledgeBase:
    def __init__(self) -> None:
        self.txts = [] 
    def load_documents(self):
        filepath = './data/textfiles'
        txtfiles = os.listdir(filepath)
        chunkdocs =[] 
        for f in txtfiles: 
            doc = load_doc(os.path.join(filepath, f))
            chunkdocs.extend(split_doc(doc))
        self.txts.extend([c.page_content for c in chunkdocs])
        self.txtids =[str(i)for i in range(0,len(self.txts))] 
 

    def load_tweets(self):
        with open('./data/tweetfiles/tweets.txt') as f:
            js = json.loads(f.read())
        self.tweets = [j.get('text') for j in js ]
        self.tweetids = [j.get('id') for j in js ]

from langchain.embeddings import OpenAIEmbeddings

class Embedder:
    def __init__(self) -> None:
        self.embedder = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=sec.get('openai_api_key')
        )

    def create_embeddings(self, txts):    
        return self.embedder.embed_documents(txts)


import pinecone
def create_pinecone_index():
    # Create a Pinecone index if it doesn't exist
    # https://docs.pinecone.io/docs/langchain-retrieval-agent

    pinecone.init(
        api_key=sec.get('pc_api_key'),
        environment=sec.get('pc_env')
    )
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='dotproduct',
            dimension=1536 
            # Supposedly 1536 is linked to the OpenAI model name.
        )

def build_kb():
    kb = MyKnowledgeBase()
    kb.load_documents()
    kb.load_tweets()
    return kb


class PineconeHelper:
    def __init__(self) -> None:
        self.index = pinecone.GRPCIndex(index_name)

    def set_embedder(self, emb):
        self.embedder = emb

    def upsert_to_index(self, txts, ids):            
        # pass data to Pinecone in batches because max size is limited
        from tqdm.auto import tqdm
        batch_size = 30
        for i in tqdm(range(0, len(txts), batch_size)):
            end = min(i+batch_size, len(txts))
            tx_batch = txts[i:end]
            id_batch = ids[i:end]
            # Put the actual text chunks in metadata 
            metadata_batch =[{'text': t} for t in tx_batch]
            ebs_batch = self.emb.create_embeddings(tx_batch)
            # The schema is: id, text embedding, text
            zvect = zip(id_batch, ebs_batch, metadata_batch)
            self.index.upsert(vectors=zvect)

def fill_vectorstore_from_kb(kb: MyKnowledgeBase, emb: Embedder):
    # Execute this only if new documents have been added
    pch = PineconeHelper()
    pch.set_embedder(emb)
    pch.upsert_to_pinecone(kb.txts, kb.txtids)
    pch.upsert_to_pinecone(kb.tweets, kb.tweetids)


import pickle
class Chatdata:
    def __init__(self, uid) -> None:
        self.uid = uid
        self.dir = os.path.join('./pickles', self.uid)

    def save_attribute(self, name, value):
        with open(os.path.join(self.dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(value, f)

    def persist(self, chat):
        create_dir_if_not_exists(self.dir)
        self.save_attribute('prev_prompt', chat.prev_prompt)
        self.save_attribute('temp', chat.temp)
        try:
            self.save_attribute('conv_memory', chat.sllm.conv_memory)
            self.save_attribute('embedder', chat.sllm.emb)
            self.save_attribute('model', chat.sllm.llm)
        except AttributeError:
            pass # sllm not created yet

    def load_attribute(self, name):
        with open(os.path.join(self.dir, f'{name}.pkl'), 'rb') as f:
            return pickle.load(f)  

    def retrieve_chatdata(self):
        self.temp = self.load_attribute('temp')
        self.prev_prompt = self.load_attribute('prev_prompt')
        try:
            self.conv_memory = self.load_attribute('conv_memory')
            self.emb = self.load_attribute('embedder')  
            self.model = self.load_attribute('model')  
        except FileNotFoundError:
            pass # sllm not created yet
        return self      

import time
def cleanup_pickles():
    pickles = os.listdir('./pickles')
    for pk in pickles:
        pkpath = os.path.join('./pickles', pk)
        if time.time() - os.path.getmtime(pkpath) > 3600:
            # delete files older than an hour
            print('deleting files in ' + pkpath)
            for f in os.listdir(pkpath):
                os.remove(os.path.join(pkpath, f))
            os.rmdir(pkpath)

        