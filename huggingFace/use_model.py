from transformers import pipeline

from transformers import pipeline

class SentimentClassifier:
    def __init__(self, model_id: str):
        self.pipe = pipeline("sentiment-analysis", model=model_id)
        # Mapeamento de rótulos
        self.label_map = {0: "NEGATIVE", 1: "POSITIVE"}

    def predict(self, text: str):
        result = self.pipe(text)[0]
        # Convertendo LABEL_0/LABEL_1 para rótulos customizados
        label_num = int(result['label'].split('_')[-1])
        label_str = self.label_map.get(label_num, result['label'])
        return f"{label_str} ({result['score']:.2f})"

if __name__ == "__main__":
    clf = SentimentClassifier("RaissaPaula/sentiment-model")
    print(clf.predict("Esse filme é maravilhoso!"))
    print(clf.predict("Eu odiei cada minuto desse filme."))

