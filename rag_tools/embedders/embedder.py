

class AbstractEmbedder:
    def __init__(self, **kwargs):
       pass

class OpenAIEmbedder(AbstractEmbedder):

    def __init__(self):
        super().__init__()

    def embed(self, sentences):
        pass


class SentenceTransformerEmbedder(AbstractEmbedder):
    def __init__(self, model_path, device="cpu"):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path)
        self.model.to(device)

    def embed(self, sentences):
        return self.model.encode(sentences, show_progress_bar=True)


class FashionCLIPEmbedder(AbstractEmbedder):
    def __init__(self):
        super().__init__()
        from fashion_clip.fashion_clip import FashionCLIP

        self.fclip = FashionCLIP('fashion-clip')

    def embed(self, images):
        import numpy as np
        image_embeddings = self.fclip.encode_images(images, batch_size=32)

        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        return image_embeddings




