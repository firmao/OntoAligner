import rdflib
import rdflib.namespace as NS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re

def parse_rdf_file(file_path):
    """
    Parses an RDF file from a URL or local path and returns a list of unique properties.
    """
    g = rdflib.Graph()
    
    if file_path.endswith('.xml'):
        file_format = 'xml'
    elif file_path.endswith('.nt'):
        file_format = 'nt'
    elif file_path.endswith('.trig'):
        file_format = 'trig'
    else:
        file_format = 'xml'

    try:
        g.parse(file_path, format=file_format)
    except Exception as e:
        print(f"Error parsing {file_path} with format {file_format}: {e}")
        return []
    
    properties = set()
    for s, p, o in g:
        properties.add(str(p))
    return list(properties)

def dereference_properties(properties):
    """
    Dereferences a list of property URIs and fetches metadata (labels, comments, etc.).
    Returns a dictionary mapping each URI to a combined text string of its metadata.
    """
    property_metadata = {}
    for uri in properties:
        full_text = []
        
        # Add the cleaned URI part itself to the text
        uri_part = uri.split('/')[-1].split('#')[-1]
        cleaned_uri_part = ' '.join(re.findall(r'[a-zA-Z0-9]+', uri_part))
        full_text.append(cleaned_uri_part)

        # Attempt to dereference the URI
        try:
            # Set a user-agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            response = requests.get(uri, headers=headers, timeout=5)
            response.raise_for_status()

            # Check if the content type is an RDF format before attempting to parse
            content_type = response.headers.get('Content-Type', '').split(';')[0]
            if any(rdf_type in content_type for rdf_type in ['xml', 'rdf', 'nt', 'trig']):
                g = rdflib.Graph()
                g.parse(data=response.text, format=content_type)
                
                # Extract labels, descriptions, comments, and definitions
                for pred in [NS.RDFS.label, NS.RDFS.comment, NS.DC.description, NS.SKOS.definition, NS.SKOS.prefLabel]:
                    for _, _, obj in g.triples((rdflib.URIRef(uri), pred, None)):
                        if isinstance(obj, rdflib.Literal):
                            full_text.append(str(obj))
            else:
                print(f"Warning: URI {uri} returned non-RDF content type: {content_type}. Using URI part only.")
        
        except (requests.exceptions.RequestException, rdflib.exceptions.ParserError) as e:
            print(f"Warning: Could not dereference or parse {uri}: {e}. Using URI part only.")
        
        property_metadata[uri] = ' '.join(full_text)
    return property_metadata

def calculate_semantic_similarity(properties1, properties2):
    """
    Calculates semantic similarity using a TF-IDF vectorizer and cosine similarity,
    based on the dereferenced content of the property URIs.
    """
    # Dereference all properties to get their metadata
    all_properties = list(set(properties1 + properties2))
    property_metadata = dereference_properties(all_properties)
    
    # Create lists of content for vectorization
    content1 = [property_metadata[p] for p in properties1]
    content2 = [property_metadata[p] for p in properties2]

    # Handle case where no content was found
    if not any(content1) and not any(content2):
        print("Warning: No properties found with any content to compare.")
        return pd.DataFrame()
        
    vectorizer = TfidfVectorizer().fit(content1 + content2)
    tfidf1 = vectorizer.transform(content1)
    tfidf2 = vectorizer.transform(content2)
    
    # Calculate cosine similarity between all pairs
    cosine_sim = cosine_similarity(tfidf1, tfidf2)
    
    results = []
    for i, p1 in enumerate(properties1):
        for j, p2 in enumerate(properties2):
            results.append({
                "Property Source": p1,
                "Property Target": p2,
                "Semantic Similarity": cosine_sim[i, j]
            })
    
    return pd.DataFrame(results)

def main():
    """
    Main function to perform property alignment.
    """
    # Updated file URLs
    file1_url = "https://github.com/firmao/OntoAligner/raw/refs/heads/AndreVTestes/assets/ontotry/codelib_cbs.xml"
    file2_url = "https://github.com/firmao/OntoAligner/raw/refs/heads/AndreVTestes/assets/ontotry/code_liss.xml"
    
    try:
        properties1 = parse_rdf_file(file1_url)
        properties2 = parse_rdf_file(file2_url)

        if not properties1 or not properties2:
            print("Could not retrieve or parse one or both RDF files.")
            return
        
        # Calculate semantic similarity
        similarity_df = calculate_semantic_similarity(properties1, properties2)
        
        if similarity_df.empty:
            print("No properties found to compare.")
            return

        # Filter for similar properties (e.g., similarity > 0.0)
        similar_properties = similarity_df[similarity_df['Semantic Similarity'] > 0.0].sort_values(
            by='Semantic Similarity', ascending=False
        )

        print("### Property Alignment Results")
        print("\nHere are the properties with a semantic similarity score greater than 0.0:")
        print(similar_properties.to_markdown(index=False))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
