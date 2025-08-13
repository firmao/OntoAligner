import rdflib
import rdflib.namespace as NS
import pandas as pd
import requests
import re
import json
import time
import random

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

def calculate_llm_similarity(properties1, properties2):
    """
    Calculates semantic similarity between properties using a Large Language Model.
    This function uses a simple API call to an LLM to get a similarity score.
    """
    property_metadata = dereference_properties(properties1 + properties2)
    results = []
    
    # Using a simple nested loop for demonstration
    for p1 in properties1:
        for p2 in properties2:
            metadata1 = property_metadata[p1]
            metadata2 = property_metadata[p2]
            
            # Construct a prompt for the LLM to compare the two properties
            prompt = (
                f"Assess the semantic similarity between Property A and Property B. "
                f"Return a single number from 0 (completely dissimilar) to 1 (perfectly similar). "
                f"Property A URI: {p1}\nProperty A Metadata: {metadata1}\n"
                f"Property B URI: {p2}\nProperty B Metadata: {metadata2}\n"
            )
            
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}]
            }

            apiKey = "AIzaSyAmVmhjmkn7o_4iBeCxxLzXTvup9wfh080"
            apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=" + apiKey
            
            max_retries = 5
            base_delay = 1
            score = 0.0

            for i in range(max_retries):
                try:
                    # Add a random delay to prevent hitting rate limits
                    time.sleep(random.uniform(0.1, 0.5))
                    
                    response = requests.post(
                        apiUrl, 
                        headers={"Content-Type": "application/json"},
                        data=json.dumps(payload)
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # Extract the first number found in the response text
                    match = re.search(r'\d+\.?\d*', response_text)
                    if match:
                        score = float(match.group(0))
                    
                    break # Success, so break the retry loop
                
                except requests.exceptions.RequestException as e:
                    if e.response is not None and e.response.status_code == 429:
                        print(f"Rate limit exceeded for properties {p1} and {p2}. Retrying in {base_delay * (2 ** i)} seconds...")
                        time.sleep(base_delay * (2 ** i) + random.uniform(0, 1))
                    else:
                        print(f"Error during LLM call for properties {p1} and {p2}: {e}. Retry {i + 1}/{max_retries}.")
                        time.sleep(base_delay * (2 ** i))
                
                except (IndexError, KeyError, ValueError) as e:
                    print(f"Error parsing LLM response for properties {p1} and {p2}: {e}. Retrying {i + 1}/{max_retries}.")
                    time.sleep(base_delay * (2 ** i))
                
            results.append({
                "Property Source": p1,
                "Property Target": p2,
                "Semantic Similarity": score
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
        
        # Calculate semantic similarity using the LLM
        similarity_df = calculate_llm_similarity(properties1, properties2)
        
        if similarity_df.empty:
            print("No properties found to compare.")
            return

        # Filter for similar properties (e.g., similarity > 0.0)
        similar_properties = similarity_df[similarity_df['Semantic Similarity'] > 0.0].sort_values(
            by='Semantic Similarity', ascending=False
        )

        print("### Property Alignment Results (LLM-based)")
        print("\nHere are the properties with a semantic similarity score greater than 0.0:")
        print(similar_properties.to_markdown(index=False))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
