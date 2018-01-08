class Text:

    def load_text_file(text_file):
        with open(text_file) as handle:
            return handle.read()
    
    def load_text_files(text_files):
        texts = []
        for text_file in text_files:
            text = Text.load_text_file(text_file)
            texts.append(text)
        return texts
