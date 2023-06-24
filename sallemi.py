import sllm_help as sh
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import pinecone
from langchain.agents import Tool
from langchain import SerpAPIWrapper
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA    
from langchain.utilities import GoogleSearchAPIWrapper
    
index_name = 'langchain-retrieval-agent'

class Sallemi:    
    def __init__(self, temp, cd: sh.Chatdata=None) -> None:
        if cd:
            self.temp = cd.temp
            self.emb = cd.emb
            self.define_vectorstore()
            self.conv_memory = cd.conv_memory
            self.llm = cd.model
            self.define_tools()
            self.agent = None  
        else:         
            self.temp = temp
            self.emb = sh.Embedder()
            self.define_vectorstore()
            self.create_conversation_memory()
            self.define_model()
            self.define_tools()
            self.agent = None

    def restore(self, cd: sh.Chatdata):
        self.temp = cd.temp
        self.emb = cd.emb
        self.define_vectorstore()
        self.conv_memory = cd.conv_memory
        self.define_model()
        self.define_tools()
        self.agent = None

    def define_vectorstore(self):
        # Add to index. 
        # Here using Pinecone client type of index.
        # self.index = pinecone.GRPCIndex(index_name)
        pinecone.init(
            api_key=sh.PC_API_KEY,
            environment=sh.PC_ENV
        )

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
        search = GoogleSearchAPIWrapper(google_api_key=sh.G_KEY, google_cse_id=sh.G_CSE)
#        search = SerpAPIWrapper(serpapi_api_key=sh.SA_KEY)
        self.tools =[
            Tool(
                name='SievoKb',
                func=qa.run,
                description=(
                    'use this tool when answering general knowledge queries to get '
                    'more information about the topic'
                )
            ),
            Tool.from_function(
                name='Search',
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
        


