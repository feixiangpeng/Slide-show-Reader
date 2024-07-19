import os
from ppt_reader import PPTReader
from text_summarizer import summarize_text  # Ensure correct module name
from image_analyzer import analyze_image
from sentence_transformers import SentenceTransformer, util

def process_presentation(file_path):
    reader = PPTReader(file_path)
    slides = reader.extract_content()

    all_texts = []
    all_images = []

    # Collect all text and images
    for slide in slides:
        if slide['text']:
            all_texts.append(slide['text'])
        all_images.extend(slide['images'])

    # Generate a single summary for all slides
    combined_text = " ".join(all_texts)
    summary = summarize_text(combined_text)
    print("Text Summary: {0}".format(summary))

    # Analyze images and collect their descriptions
    image_descriptions = []
    for image_path in all_images:
        image_description = analyze_image(image_path)
        image_descriptions.append(image_description)

    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the summary and image descriptions
    summary_embedding = model.encode(summary, convert_to_tensor=True)
    image_embeddings = model.encode(image_descriptions, convert_to_tensor=True)

    # Find relevant image descriptions based on semantic similarity
    relevant_image_descriptions = []
    for desc, emb in zip(image_descriptions, image_embeddings):
        similarity = util.pytorch_cos_sim(summary_embedding, emb)
        if similarity.item() > 0.3:  # Adjust threshold as needed
            relevant_image_descriptions.append(desc)

    if relevant_image_descriptions:
        print("\nRelevant Image Descriptions:")
        for i, desc in enumerate(relevant_image_descriptions, 1):
            print("Image {0} Description: {1}".format(i, desc))

    # Clean up temporary image files
    for image_path in all_images:
        try:
            os.remove(image_path)
        except FileNotFoundError:
            pass
    

if __name__ == "__main__":
    ppt_file = "Scout - Pitch Competition Deck.pptx"
    process_presentation(ppt_file)
    
