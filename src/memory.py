from langchain_community.chat_message_histories import ChatMessageHistory

chat_history = ChatMessageHistory()

def add_interaction(question, answer):

    chat_history.add_user_message(question)

    chat_history.add_ai_message(answer)

def get_chat_history():

    return chat_history.messages