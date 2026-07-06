from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np


MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegFormerEngine:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label


    def predict(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(**inputs)

        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )

        prediction = logits.argmax(dim=1)[0]

        return prediction.cpu().numpy()


    def statistics(self, mask):

        unique, counts = np.unique(
            mask,
            return_counts=True
        )

        total = counts.sum()

        stats = {}

        for cls, cnt in zip(unique, counts):

            label = self.id2label.get(
                int(cls),
                str(cls)
            )

            stats[label] = round(
                cnt * 100 / total,
                2
            )

        return stats