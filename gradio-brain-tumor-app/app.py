import gradio as gr
from transformers import pipeline
import os

# Modelle laden
vit_classifier = pipeline("image-classification", model="Tharsana/vit-base-brain-tumor")
clip_detector = pipeline("zero-shot-image-classification", model="openai/clip-vit-large-patch14")

# Semantische Labels für CLIP
semantic_labels = [
    "An MRI scan showing a brain tumor",
    "An MRI scan showing a healthy brain"
]

# Mapping zu yes/no
label_map = {
    "An MRI scan showing a brain tumor": "yes",
    "An MRI scan showing a healthy brain": "no"
}

# Klassifikationsfunktion
def classify_brain(image):
    # ViT-Vorhersage
    vit_results = vit_classifier(image)
    vit_output = {res["label"]: res["score"] for res in vit_results}

    # CLIP mit semantischen Labels, gemappt auf yes/no
    clip_results = clip_detector(
        image,
        candidate_labels=semantic_labels,
        hypothesis_template="This MRI scan shows {}."
    )
    clip_output = {
        label_map[res["label"]]: res["score"] for res in clip_results
    }

    # Dateiname extrahieren
    image_name = os.path.basename(image)

    return {
        "Bildname": image_name,
        "ViT Classification": vit_output,
        "CLIP Zero-Shot Classification": clip_output
    }

# Beispielbilder (optional)
example_images = [
    ["example_images/yes_1.jpg"],
    ["example_images/no_1.jpg"],
    ["example_images/yes_2.jpg"],
    ["example_images/no_2.jpg"],
    ["example_images/yes_3.jpg"],
    ["example_images/no_3.jpg"],
]

# Gradio-Interface
iface = gr.Interface(
    fn=classify_brain,
    inputs=gr.Image(type="filepath"),
    outputs=gr.JSON(),
    title="🧠 Brain Tumor Classifier",
    description="Diese Anwendung vergleicht zwei KI-Modelle zur Klassifikation von Hirntumoren:\n\n"
                "**Modell 1:** Ein trainiertes ViT-Modell, das speziell auf MRT-Bilder trainiert wurde.\n\n"
                "**Modell 2:** Ein CLIP Zero-Shot-Modell mit semantischen Labels, aber Ausgabe als 'yes'/'no'.\n\n"
                "🔬 Lade ein MRT-Bild hoch und sieh dir die Ergebnisse beider Modelle an.",
    examples=example_images
)

iface.launch()