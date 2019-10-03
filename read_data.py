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
        print("Currently processing: {}".format(filename))
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



sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
words, bound = process_data_tags("speech", config.data_dir)

df = pd.DataFrame(list(zip(words, bound)), columns =['Sentence', "Boundary"]) 
print("Writing data to csv...")
df.to_csv(config.CSV_FILENAME, index=False, encoding='utf-8')
#write_sents_to_csv(sentences, config.CSV_FILENAME)
print("Done writing data to csv.")
