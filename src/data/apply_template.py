def chat_template(data, tokenizer):

    text = tokenizer.apply_chat_template(
        data["message"],
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text" : text}