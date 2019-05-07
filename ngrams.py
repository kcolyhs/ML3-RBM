import re


def generate_ngrams(line, count):
    '''
    Generates a list of ngrams of the string line
    '''
    line = line.lower()
    line = re.sub(r'\W+', ' ', line)

    tokens = [token for token in line.split(" ") if token != ""]

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(count)])
    return [" ".join(ngram) for ngram in ngrams]


if __name__ == "__main__":
    print(generate_ngrams("Hello there, Mr. Finch", 2))
