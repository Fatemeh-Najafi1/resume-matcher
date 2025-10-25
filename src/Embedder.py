import numpy as np
#from config import embedding_model
#didnt compute embedding model from config file yet

class Embedder:
    def __init__(self, model = 'basic', llm_model = None):
         #class embed chooses models to embed text
         #if the client doesnt choose an ai llm model:
         # we have basic model
         self.model = model
         self.llm_model = llm_model
    def get_embeddings(self, text, model):
        #if the llm wasnt chosen or provided: basic
        if self.model == "basic":
            return self.embed_basic(text)
        #if user chooses ollama
        elif self.model == "ollama":
            return self.embed_ollama(text)
        #if user  chooses openai
        elif self.model == "openai":
            return self.embed_openai(text)
        #if system doesnt recognize the model
        else:
            raise ValueError(f"Unknown model type: {self.model}")
    def embed_basic(self, text):
        #turn text into tokens
        words = text.split()
        if not words:
            #avoids division by zero
            return np.zeros(300)
        num_vec = np.array([hash(ch) % 1000 for ch in words])
        #takes list of words turns every word into number using hashing
        #1000 for simplicity
        vector = np.zeros(300)
        #creating the vextor list n size
        vector[:min(len(num_vec), 300)] = num_vec[:min(len(num_vec), 300)]
        #normalizing the vector 
        return vector 

    def embed_ollama(self, text):
        #for development in later phases
        pass 
    def embed_openai(self, text):
         #for development in later phases
         pass 

        