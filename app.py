from sallemi import Sallemi, sh
from pinecone import PineconeException
from uuid import uuid4
from pickle import PicklingError


def return_ugly_error(msg):
    ugly_error = f"""    
    <h1>Launching failed.</h1>
    <p>Error - {msg} 
    """
    return render_template('launch_error.html', msg=msg)
    #return ugly_error

chat = None

class Chat:
    def __init__(self, uid) -> None:        
        self.prev_prompt = ''
        self.temp = 0    
        self.uid = uid    
    
    # def init_sllm(self):
    #     # Sallemi retains the current agent + tools and conversation history
    #     self.sllm = Sallemi(self.temp)

    def get_response(self, userText):
        prompt = userText.lower().strip()

        if prompt == 'have a drink': 
            if self.temp <= 1: 
                self.temp += .2
                resp = f'Thanks!'  
            else:
                resp = f'Thanks! I\'ve had quite enough.'                        
        else:
            if self.prev_prompt == 'have a drink' or self.prev_prompt == '':
                # Start an agent with selected temperature
                try:
                    self.sllm.temp
                except AttributeError:
                    self.sllm = Sallemi(self.temp)                
                self.sllm.start_agent()              

            resp = self.sllm.agent(userText)
            resp = resp['output'] 
        self.prev_prompt = prompt
        return resp

    def restore(self, cd: sh.Chatdata):        
        self.sllm = Sallemi(cd.temp, cd)
        return self

from flask import Flask, render_template, request, session, redirect, url_for
app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = 'dounfal42'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
launch_error = False

@app.route("/")
def index():    
    uid = session.get("uid")
    # Flask sessions expire when the browser is closed, which is fine https://flask.palletsprojects.com/en/2.3.x/api/#flask.session.permanent
    if not uid:
        ret = render_template("login.html")
    else:
        ret = render_template("index.html")  
    return ret
        # try:
        #     global chat
        #     cd = sh.Chatdata(uid)
        #     if chat == None:
        #         chat = Chat(uid)
        #         chat.init_sllm()                
        #         cd.persist(chat)
        #     elif chat.uid != uid:
        #         try:
        #             chat = Chat(uid)
        #             chat = chat.restore(cd.retrieve_chatdata())
        #         except FileNotFoundError:
        #             # Theres's no previous chat for this uid. Create a new one.
        #             chat = Chat(uid)
        #             chat.init_sllm()                
        #             cd.persist(chat)
        #     ret = render_template("index.html")
        # except PineconeException as e:
        #     print('Sorry, Pinecone is not behaving')        
        #     ret = return_ugly_error(e.__str())
        # except PicklingError:
        #     pass            
        # except FileNotFoundError:

        #     ret = return_ugly_error('Sorry, cannot find files')
    

@app.route('/login', methods=['GET', 'POST'] )
# https://pythonbasics.org/flask-sessions/
def login():
    if request.method == "POST":        
        nm = request.form.get("uname")
        uid = str(uuid4())
        session["uid"] = uid
        if nm:
            session['uname']= nm.lower().replace(' ', '')
            ret = redirect(url_for('index'))  
        else:
            ret = redirect(url_for('login')) 
    else:
        ret = render_template('login.html')
    return ret

@app.route("/get")
# index.html always calls /get
def get_bot_response():
    global chat
    uid = session.get('uid')
    if uid:
        cd = sh.Chatdata(uid)
        try:
            if chat:
                if chat.uid != uid:
                    # The chat currently in memory does not match this session uid. Get the right chat context.
                    chat = chat.restore(cd.retrieve_chatdata())
                else:
                    # all good, proceed to getting response
                    pass
            else:
                chat = Chat(uid)
                # See if there's a persisted chat context.
                chat = chat.restore(cd.retrieve_chatdata())
        except FileNotFoundError:
        # No persisted chat found. Create new chat.
            chat = Chat(uid)
            # chat.init_sllm()                                

        userText = request.args.get('msg')
        res = chat.get_response(userText)            
        cd.persist(chat)
    else:
        res = return_ugly_error("Sorry, couldn't find your chat context.")    
    return res

        
sh.cleanup_pickles()


if __name__ == "__main__":
    app.run(debug=False)