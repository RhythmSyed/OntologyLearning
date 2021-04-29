import pandas as pd
import csv
import numpy as np
import os
import re
import sys
import spacy
import string
from openie import StanfordOpenIE
import opennre

nlp = spacy.load('en_core_web_lg')
re_model = opennre.get_model('wiki80_bertentity_softmax')
STOP_WORDS = nlp.Defaults.stop_words
PUNC_LIST = nlp.Defaults.prefixes


def format_int(x):
    if x in [float("-inf"),float("inf")]: return float("nan")
    return x


def list_files(startpath):
    taxonomy = {}
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        if level not in taxonomy:
            if not level == 0:
                prev_root = root.split('/')[level+2]
            else:
                prev_root = os.path.basename(root)
            taxonomy[level] = [(prev_root, os.path.basename(root))]
        else:
            if not level == 0:
                prev_root = root.split('/')[level+2]
            else:
                prev_root = os.path.basename(root)
            taxonomy[level].append((prev_root, os.path.basename(root)))

        print('{} {}{}/'.format(level, indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
    return taxonomy


def get_descriptions(taxonomy, dataset):
    descriptions = {}
    members = []
    for key, val in taxonomy.items():
        for i in val:
            members.append(i[0])
            members.append(i[1])
    members = list(set(members))

    for index, row in dataset.iterrows():
        if index % 500 == 0:
            print('processed {} out of {}'.format(index, len(dataset)))
        sentence = dataset.iloc[index][0]
        for member in members:
            found = re.search(member, sentence)
            if found:
                tokens = []
                splitted = sentence.split(' ')
                for token in splitted:
                    if token not in STOP_WORDS and token not in PUNC_LIST:
                        token = token.translate(str.maketrans(dict.fromkeys(string.punctuation.replace('-', ''))))
                        tokens.append(token)

                # doc = nlp(sentence)
                # tokens = []
                # for sent in doc.sents:
                #     for token in sent:
                #         if not token.is_stop and not token.is_punct and not token.like_num:
                #             tokens.append(token.text)

                if member not in descriptions:
                    descriptions[member] = [tokens]
                else:
                    descriptions[member].append(tokens)

    total = 0
    for i in descriptions:
        print(i, len(descriptions[i]))
        total += len(descriptions[i])
    print('\ntotal sentences: {}'.format(total))

    description_corpus = []
    for key in descriptions.keys():
        for sentence in descriptions[key]:
            description_corpus.append(sentence)

    return description_corpus, members


def calculate_PMI(descriptions, taxonomy):
    words = []
    for sentence in descriptions:
        words += sentence
    words = list(set(words))
    word_word_matrix = pd.DataFrame(-10, index=words, columns=words)
    # word_word_matrix = word_word_matrix.drop(index='')
    # word_word_matrix = word_word_matrix.drop(columns='')

    N = len(descriptions)
    cols = word_word_matrix.columns.values

    members = []
    for key, val in taxonomy.items():
        if key in [1]:
            for i in val:
                members.append(i[0])
                members.append(i[1])
    consideration = list(set(members))
    consideration.remove('our-l3-0.25')

    for word in consideration:
        print('working on {}'.format(word))
        for ii, col in enumerate(cols):
            if ii % 500 == 0:
                print('finished {} out of {}'.format(ii, len(cols)))
            p_w = 0
            p_c = 0
            p_w_c = 0
            for i, sentence in enumerate(descriptions):
                if word in sentence:
                    p_w += 1

                if col in sentence:
                    p_c += 1

                if word in sentence and col in sentence:
                    p_w_c += 1

            p_w = p_w / N
            p_c = p_c / N
            p_w_c = p_w_c / N
            # if p_w == 0 or p_c == 0:
            #     continue
            pmi = np.log(p_w_c / (p_w * p_c))
            word_word_matrix.loc[col, word] = format_int(pmi)

    label_replacements = {}
    for word in consideration:
        index = word_word_matrix[word].argmax()
        print('Closest Related Pair: ({}, {}), PMI: {}'.format(word, word_word_matrix.iloc[index].name, word_word_matrix.iloc[index][word]))
        label_replacements[word] = word_word_matrix.iloc[index].name
    return label_replacements


def extract_relations(taxonomy, dataset, labels):
    concept_pairs = taxonomy[2] + taxonomy[3]

    triple_texts = []
    final_triples_both = []

    with StanfordOpenIE() as client:
        for index, row in dataset.iterrows():
            if index % 500 == 0:
                print('processed {} out of {}'.format(index, len(dataset)))
            text = dataset.iloc[index][0]

            for concept1, concept2 in concept_pairs:
                if concept1 == concept2:
                    new_triple = [concept1, 'is', concept2]
                    if new_triple not in final_triples_both:
                        triple_texts.append(text)
                        final_triples_both.append(new_triple)
                    continue

                # if concept1 in labels:
                #     concept1 = labels[concept1]
                # if concept2 in labels:
                #     concept2 = labels[concept2]

                found1 = re.search('(^|\W)' + concept1 + '($|\W)', text)
                found2 = re.search('(^|\W)' + concept2 + '($|\W)', text)

                if found1 is not None and found2 is not None:
                    #         if concept1 in text and concept2 in text:
                    doc = nlp(text)
                    sentences = [sent.string.strip() for sent in doc.sents]
                    triples = []
                    for sentence in sentences:
                        for triple in client.annotate(sentence):
                            triples.append(triple)

                    for t in triples:
                        #                 print(t)
                        if len(t) != 3: continue

                        relation_found = False
                        if concept1 in t['subject'] or concept1 in t['object']:
                            if concept2 in t['subject'] or concept2 in t['object']:
                                if t not in final_triples_both:
                                    triple_texts.append(text)
                                    new_triple = [concept1, t['relation'], concept2]
                                    final_triples_both.append(new_triple)
                                relation_found = True

                        if not relation_found:
                            relation_pred = re_model.infer({'text': text, 'h': {'pos': found2.span()}, 't': {'pos': found1.span()}})
                            if relation_pred[1] >= 0.90:
                                new_triple = [concept1, relation_pred[0], concept2]
                            else:
                                new_triple = [concept1, 'NO-RELATION', concept2]
                            if new_triple not in final_triples_both:
                                triple_texts.append(text)
                                final_triples_both.append(new_triple)

    return final_triples_both, triple_texts


def build_ontology(triples, taxonomy):
    ontology = []
    completed = []

    for key, val in taxonomy.items():
        if key in [2, 3]:
            for pair in val:
                for triple in triples:
                    if pair[0] == triple[0] and pair[1] == triple[2]:
                        if triple[1] != 'NO-RELATION':
                            if pair not in completed:
                                ontology.append(triple)
                                completed.append(pair)

    return ontology


def main(taxonomy_path, dataset_path):
    print('*** Generated Taxonomy ***')
    taxonomy = list_files(taxonomy_path)
    dataset = pd.read_csv(dataset_path, delimiter='\n', header=None, error_bad_lines=False)
    print('*** Generating Descriptions ***')
    descriptions, members = get_descriptions(taxonomy, dataset)
    print('*** Processing PMI for Word Pairs ***')
    labels = calculate_PMI(descriptions, taxonomy)
    print('*** Extracting Relations ***')
    triples, texts = extract_relations(taxonomy, dataset, labels)
    print('*** Building Ontology ***')
    ontology = build_ontology(triples, taxonomy)

    with open('ontology.txt', 'a', newline='') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        for triple in ontology:
            wr.writerow(triple)


if __name__ == '__main__':
    if len(sys.argv) < 1:
        sys.exit()
    taxonomy_path = str(sys.argv[1])
    if "\r" in taxonomy_path:
        taxonomy_path = taxonomy_path.replace("\r", "")
    dataset_path = str(sys.argv[2])

    # taxonomy_path = "./data/starwars/our-l3-0.25"
    # dataset_path = "./star_wars_cleaned.txt"
    main(taxonomy_path, dataset_path)
