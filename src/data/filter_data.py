def remove(data):
    train_data = data.remove_columns(
        [
            'instruction',
            'context',
            'response',
            'category',
            'message'
        ]
    )

    return train_data