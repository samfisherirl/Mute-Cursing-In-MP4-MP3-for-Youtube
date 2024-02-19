import csv
from nltk.corpus import wordnet

# Make sure to download the wordnet corpus if you haven't already
import nltk
nltk.download('wordnet')


def get_near_words(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


# Replace 'your_file.csv' with the path to your CSV file
with open('cleaned_output.csv', mode='r') as infile, open('updated_file.csv', mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        word = row[0]  # Assuming the word is in the first column
        near_words = get_near_words(word)
        for near_word in near_words:
            writer.writerow([near_word])

print("Updated CSV file with near words has been created as 'updated_file.csv'.")
