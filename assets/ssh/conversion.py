from rdflib import Graph
import urllib.request

# Create a Graph object
g = Graph()

url="https://raw.githubusercontent.com/Mireldar01/Thesis_2025/refs/heads/main/ontology_and_schema_alignment_experiments/clariah-tools.ttl"
# Open the URL and read its content
with urllib.request.urlopen(url) as response:
    turtle_content = response.read().decode('utf-8')

# Parse the Turtle content
g.parse(data=turtle_content, format="turtle")

# Serialize the graph to RDF/XML format
# The 'pretty-xml' format provides a more readable XML output
rdfxml_output = g.serialize(format="pretty-xml")

# Save the RDF/XML output to a file
with open("clariah-tools.xml", "w", encoding="utf-8") as f:
    f.write(rdfxml_output)

print("Conversion complete. The RDF/XML output is saved to clariah-tools.xml")