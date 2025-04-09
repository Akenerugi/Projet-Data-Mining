import requests
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re



nltk.download('punkt')
nltk.download('stopwords')


def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        article = soup.find('article')
        if not article:
            article = soup.find('main') 
        if article:
            paragraphs = article.find_all('p')
        else:
            paragraphs = soup.find_all('p')
 
        text = ' '.join([p.get_text() for p in paragraphs]) 
        return text    
    except Exception as e:
        print(f"Erreur lors de l'extraction pour {url}: {e}")
        return ""


def clean_text(text): 
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = text.strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def summarize_text(sentences):
    if len(sentences) < 3:
        print("Pas assez de phrases pour le clustering.")
        return "Résumé non disponible (pas assez de contenu)."

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    n_clusters = min(3, len(sentences))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X) 
    clustered_sentences = {i: [] for i in range(n_clusters)}
    
    for i, label in enumerate(kmeans.labels_):
        clustered_sentences[label].append(sentences[i])
    summary = []
    
    for cluster in clustered_sentences.values():
        summary.append(' '.join(cluster[:2]))
    return '\n'.join(summary)



def wrap_text(text, line_length=80):
    lines = []
    words = text.split()
    current_line = []    
    for word in words:
        if sum(len(w) for w in current_line) + len(word) + len(current_line) > line_length:
            lines.append(' '.join(current_line))
            current_line = [word]
        else:
            current_line.append(word)    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)
 


def process_urls(urls):
    unwanted_phrases = [
        "When you purchase through links on our site, we may earn an affiliate commission",
        "This article contains affiliate links, which means we may earn a small commission",
        "Our reviews are based on independent research and analysis",
        "Affiliate links on Android Authority may earn us a commission. Learn more.",
    ]    
    for i, url in enumerate(urls):
        print(f"Traitement de l'URL {i+1}/{len(urls)}: {url}")
        text = extract_text_from_url(url)
        if text:
            sentences = sent_tokenize(text)
            summary = summarize_text(sentences)
            summary_lines = summary.split('\n')
            filtered_summary = '\n'.join([
                line for line in summary_lines 
                if not any(unwanted_phrase.lower() in line.lower() for unwanted_phrase in unwanted_phrases)
            ])
            filtered_summary = convert_to_third_person(filtered_summary)   
            filename = f"resume_texte_{i+1}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(wrap_text(filtered_summary, line_length=80))
            
            print(f"Résumé enregistré dans {filename}\n")
        else:
            print(f"Pas de texte extrait pour {url}\n")