# anonymiser
import spacy

nlp = spacy.load("en_core_web_sm")


def simple_anonymiser(text):
    doc = nlp(text)

    tokens = []
    for token in doc:
        if len(token.ent_type_):
            tokens.append("X" * len(token.text))
        else:
            tokens.append(token.text)

    return " ".join(tokens)


def complex_anonymizer(text):
    doc = nlp(text)
    entity = ""
    tokens = []
    for token in doc:
        if len(token.ent_type_) and (token.ent_iob_ == 'B' or token.ent_iob_ == 'I'):
            entity = token.ent_type_
        else:
            if len(entity):
                tokens.append(entity)
                entity = ""
            tokens.append(token.text)
    if len(entity):
        tokens.append(entity)
    return " ".join(tokens)


if __name__ == '__main__':
    text = "Policyholder: Jane Smith\n" \
           "Insurance Company: SportsSure Insurance\n" \
           "Policy Number: SSP12345678\n" \
           "Claim Number: SCC98765432\n " \
           "Date: November 14, 2023\n " \
           "On October 15, 2023, I participated in the annual City Triathlon, which took " \
           "place in Sunnyville."
    print(simple_anonymiser(text))
    print(complex_anonymizer(text))
