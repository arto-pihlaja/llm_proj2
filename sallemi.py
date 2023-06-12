import sllm_help as sh
import json
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import pinecone
from langchain.agents import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA    
import os    


class MyKnowledgeBase:
    def __init__(self) -> None:
        self.txts = [] 
    def load_documents(self):
        filepath = './data/textfiles'
        txtfiles = os.listdir(filepath)
        chunkdocs =[] 
        for f in txtfiles: 
            doc = sh.load_doc(os.path.join(filepath, f))
            chunkdocs.extend(sh.split_doc(doc))
        self.txts.extend([c.page_content for c in chunkdocs])
        self.txtids =[str(i)for i in range(0,len(self.txts))] 
 

    def load_tweets(self):
        with open('./data/tweetfiles/tweets.txt') as f:
            js = json.loads(f.read())
        self.tweets = [j.get('text') for j in js ]
        self.tweetids = [j.get('id') for j in js ]

class Embedder:
    def __init__(self) -> None:
        self.embedder = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=sh.OPENAI_API_KEY
        )

    def create_embeddings(self, txts):    
        return self.embedder.embed_documents(txts)
    
index_name = 'langchain-retrieval-agent'

def create_pinecone_index():
    # Create a Pinecone index if it doesn't exist
    # https://docs.pinecone.io/docs/langchain-retrieval-agent

    pinecone.init(
        api_key=sh.PC_API_KEY,
        environment=sh.PC_ENV
    )
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='dotproduct',
            dimension=1536 
            # Supposedly 1536 is linked to the OpenAI model name.
        )

class Sallemi:    
    def __init__(self, temp) -> None:
        self.temp = temp
        self.kb = MyKnowledgeBase()
        self.kb.load_documents()
        self.kb.load_tweets()
        self.emb = Embedder()
        create_pinecone_index()
        self.create_vectorstore()
        self.create_conversation_memory()
        self.define_model()
        self.define_tools()
        self.agent = None

    def upsert_to_pinecone(self, txts, ids):
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

    def create_vectorstore(self):
        # Add to index. 
        # Here using Pinecone client type of index.
        self.index = pinecone.GRPCIndex(index_name)

        # Commenting this out because I have already loaded the data to the index
        # # Put the actual text chunks to retrieve in metadata 
        # self.upsert_to_pinecone(self.kb.txts, self.kb.txtids)
        # self.upsert_to_pinecone(self.kb.tweets, self.kb.tweetids)
        
        # Specify in which field the actual text chunks are
        text_field='text'
        # Get the index object to use with Langchain
        lc_index=pinecone.Index(index_name)
        self.vectorstore = Pinecone(lc_index, self.emb.embedder.embed_query, text_field)


    def create_conversation_memory(self):
        self.conv_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )

    def define_tools(self):
        # Link the vectorstore to the model and wrap into a tool that the agent can use
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.vectorstore.as_retriever()
        )
        search = SerpAPIWrapper(serpapi_api_key=sh.SA_KEY, search_engine='google')
        self.tools =[
            Tool(
                name='SievoKb',
                func=qa.run,
                description=(
                    'use this tool when answering general knowledge queries to get '
                    'more information about the topic'
                )
            ),
            Tool(
                name='Lmgtfy',
                func=search.run,
                description='useful for when you need to answer questions about current events'
                )
        ] 


    def define_model(self):
    # Specify the chat model 
        self.llm = ChatOpenAI(
            openai_api_key=sh.OPENAI_API_KEY,
            model_name='gpt-3.5-turbo',
            temperature=self.temp
        )

    def start_agent(self):
        # Create the agent that uses the LLM, memory and tool
        from langchain.agents import initialize_agent
        self.agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=self.conv_memory
        )
        


