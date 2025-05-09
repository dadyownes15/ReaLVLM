class Detector:
    def __init__(self):
        # The detector is the function that takes a clip (TxHxWxC)and returns a score
        self.detector = None
        self.type = None
    
    def calculate_score(self, embeddings):
        pass

    def detect(self, embeddings):
        pass
    
    def train(self, embeddings):
        pass

    def calculate_scores(self, embeddings_dir):
        pass

    