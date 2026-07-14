
def format_chat(data):
    user_message = data['instruction']

    if data['context']:
        user_message += "\n\nContext:\n" + data['context']

    message = [
        {
            "role" : "user",
            "content" : user_message
        },
        {
            "role" : "assistant",
            "content" : data['response']
        }
    ]

    return {"message": message}