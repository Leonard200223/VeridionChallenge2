# VeridionChallenge2
The solution I thought about:

I chose to represent companies and their categories as embeddings using a NLP model (all-MiniLM-L6-v2).
I grouped the companies in clusters using K-means clustering (without knowing the categories).
Then, I associated each cluster with the closest name of category, using the cosinus similarity between embeddings.

Python 3.9.0
