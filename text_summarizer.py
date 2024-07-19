from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import nltk
from transformers import pipeline

nltk.download('punkt')
nltk.download('stopwords')

# Load the summarization pipeline for refinement
refiner = pipeline("text2text-generation", model="facebook/bart-large-cnn")

def draft_summary(text, num_sentences=2):  # Reduced number of sentences
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and word.isalpha()]

    # Calculate word frequencies
    freq = FreqDist(words)

    # Score sentences based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = freq[word]
                else:
                    sentence_scores[sentence] += freq[word]

    # Get the top N sentences with the highest scores
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    # Join the top sentences to create the summary
    summary = ' '.join(summary_sentences)

    return summary

def summarize_text(text, num_sentences=2):
    # Generate draft summary
    draft = draft_summary(text, num_sentences)
    
    # Refine the draft summary
    refined = refiner(draft, max_length=150, min_length=50, do_sample=False)
    print(refined)  # Print the refined output to check its structure
    return refined[0]['generated_text']  # Update this key based on the print output
