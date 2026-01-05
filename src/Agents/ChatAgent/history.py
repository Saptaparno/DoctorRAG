from langchain_community.chat_message_histories import ChatMessageHistory
_store = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

MAX_TURNS = 6

def trim_history(session_id="default"):
    h = get_history(session_id)
    msgs = h.messages
    if len(msgs) > MAX_TURNS * 2:
        h.messages = msgs[-MAX_TURNS*2:]
        
def clear_session(session_id="default"):
    _store[session_id] = ChatMessageHistory()
