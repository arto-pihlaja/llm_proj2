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
        try:
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
        except AttributeError: 
            #incorrect cd
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
        pinecone.init(
            api_key=sh.sec.get('pc_api_key'),  
            environment=sh.sec.get('pc_env')
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
        search = GoogleSearchAPIWrapper(google_api_key=sh.sec.get('g_api_key'), google_cse_id=sh.sec.get('g_cse'))
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
            openai_api_key=sh.sec.get('openai_api_key'),
            model_name='gpt-3.5-turbo',
            temperature=self.temp
        )

    def start_agent(self):
        # Create the agent that uses the LLM, memory and tool.
        from langchain.agents import initialize_agent
        from langchain.agents.agent_types import AgentType
        self.agent = initialize_agent(
            # agent='chat-conversational-react-description',
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=self.conv_memory
        )
        


