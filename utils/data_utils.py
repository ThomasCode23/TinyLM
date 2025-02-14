import json

def load_conversation_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        conversations = []

        # Assuming data is a list of conversation dictionaries
        for item in data:
            conversation = item.get('conversation', [])
            # Concatenate the 'text' field from each message in the conversation
            conversation_text = ''
            for turn in conversation:
                conversation_text += turn.get('text', '') + '\n'  # Add newline to separate turns if needed
            conversations.append(conversation_text.strip())  # Remove trailing newline

    return conversations
