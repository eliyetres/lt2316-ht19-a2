import os
import re
import csv
import nltk.data
import nltk.tokenize.punkt
import xml.etree.ElementTree as ET

import config


def process_data_tags(tag, dirname):
    """ Takes a speech tag as string and a directory name. Processes all XML files in the directory """

    words = []
    boundaries = []
    for filename in os.listdir(dirname):
        if not filename.endswith('.xml'):
            continue
    for index, filename in enumerate(os.listdir(dirname)):
        #print("Currently processing: {}".format(filename))
        # if index > 20:
        #     break
        if not filename.endswith('.xml'):
            continue
        fullname = os.path.join(dirname, filename)
        tree = ET.parse(fullname)
        root = tree.getroot()
        speech_tags = root.findall("speech")
        speakers = []
        for item in speech_tags:
            if len(speakers) > 2:
                speakers.pop(0)
            # print(ET.dump(item)) # use this to print roots
            try:  # check if there is no name for speaker
                item.attrib['nospeaker']
                boundary = "[SAME]"
                speakers.append("noname")  # add to speaker list
            except:
                speaker = item.attrib['speakername']
                speakers.append(speaker)
                if len(speakers) < 2:  # first speaker
                    boundary = "[SAME]"
                else:  # thereafter
                    if speakers[0] == speakers[1]:
                        boundary = "[SAME]"
                    else:
                        boundary = "[CHANGE]"

            notags = ET.tostring(item, encoding='unicode', method='text')
            notags = re.sub(r'(\s{2,})', ' ', notags)
            punkt = sent_detector.tokenize(notags)
            for index, s in enumerate(punkt):
                words.append(s.strip())
                boundaries.append("[SAME]")
            boundaries.pop(-1)
            boundaries.append(boundary)

    # remove the last boundary because the very last sentence should not have a boundary after it
    boundaries.pop(-1)
    return words, boundaries


def post_process(words, bound):

    new_words = []
    new_bound = []

    #-1 because the last boundary is removed
    for i in range(0, len(words) - 2):
        # Loops with two sentences at a time to see if they
        # contain the given expressions
        if(check_expression(words[i], words[i + 1])):
            new_w = words[i] + ' ' + words[i + 1]
            new_b = bound[i + 1]

            # Print lines found by an expression
            #print_check(i, words[i], bound[i], i+1, words[i+1], bound[i+1], new_w, new_b)

            # Replace next index in sentence list with the concatenation of current and next
            # Don't need to replace tag since we want the last value and it's already at this index
            words[i + 1] = new_w

        else:
            # Write complete sentences and their boundaries to the lists
            new_words.append(words[i])
            new_bound.append(bound[i])

    # Write last to new list since it's disregarded from the loop
    new_words.append(words[len(words) - 1])

    return new_words, new_bound


def check_expression(sent1, sent2):

    abbr = ['G.P.', 'N.H.S.', 'cent.', 'Ltd.', 'e.g.', 'i.e.', 'etc.']
    comments = ['rose—', 'indicated assent.', 'rose —', 'indicated dissent.']
    number_exps = ['Nos.', 'No.']
    honorary = ['hon.', 'Hon.']
    dashes = ['-', '—', ']']

    sent1_last = sent1.split()[-1]

    # Sentence pair where the words in comments and dashes are left as they are.
    # Second pair entences that start with lowercase but don't fall into the following conditions are also left as they are
    # even though some misspelt sentences like "percent.", "per. cent.", and "Which?" will not be captured

    # Where first sentence ends with 'hon.' or 'Hon''
    if sent1_last in honorary:  # sent2 does not need to start with lowercase
        return True
    # Where first sentence ends with No. or Nos. and second start with a digit
    elif sent1_last in number_exps and len(sent1) > 3 and sent2[:1].isdigit():
        return True
    # Where first sentence ends with an abbreviation in the list and second is lowercase
    elif sent1_last in abbr and sent2[:1].islower():
        return True
    # Where the first sentence ends with a quotation mark and second is lowercase
    elif sent1.endswith('\"') and sent2[:1].islower():
        return True


def print_check(i1, sent1, bound1, i2, sent2, bound2, new_w, new_b):
    print(i1, sent1, bound1)
    print(i2, sent2, bound2)
    print(new_w, new_b)
    print('\n')


def write_sents_to_csv(sentences, filename):
    with open(filename, 'w', encoding="utf-8") as wfile:
        writer = csv.writer(wfile)
        # write header
        writer.writerow(["Sentence 1", "Sentence 2", "Boundary"])
        for index in range(0, len(sentences), 2):
            try:
                sent1 = sentences[index]
                boundary = sentences[index + 1][1:][:-1]
                sent2 = sentences[index + 2]
                writer.writerow([sent1, sent2, boundary])
            except IndexError:
                pass


import pandas as pd


print("Loading data...")
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

print("Processing data tags...")
words, bound = process_data_tags("speech", config.data_dir)

print("Post processing...")
words2, bound2 = post_process(words, bound)

# Prints
print("Lengths of sentence and boundaries lists before post processing:")
print(len(words))
print(len(bound))

print("Lengths of sentence and boundaries lists after post processing:")
print(len(words2))
print(len(bound2))

#df = pd.DataFrame(list(zip(words, bound)), columns =['Sentence', "Boundary"])
df = pd.DataFrame(list(zip(words2, bound2)), columns=['Sentence', "Boundary"])

print("Writing data to csv...")
df.to_csv(config.CSV_FILENAME, index=False, encoding='utf-8')
#write_sents_to_csv(sentences, config.CSV_FILENAME)
print("Done writing data to csv.")
