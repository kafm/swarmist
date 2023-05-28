#create a function that receives a text and returns a list of tokens
def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

#create the simple_preprocess function
def simple_preprocess(text):
    return text.split()

#create that split the test of stopwords where stop
