import spacy_udpipe

nlp = spacy_udpipe.load('en')

text = u"You are using UDPipe inside SpaCy for natural language processing."
doc = nlp(text)

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)
