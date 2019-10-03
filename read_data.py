import os
import re
import csv
import nltk.data
import nltk.tokenize.punkt
import xml.etree.ElementTree as ET

import config


def process_data_tags(tag, dirname):
    """ Takes a speech tag as string and a directory name. Processes all XML files in the directory """
    sentences = []
    for filename in os.listdir(dirname):
        if not filename.endswith('.xml'): continue
    for index, filename in enumerate(os.listdir(dirname)):
        if index > 20:
            break
        print("Currently processing: {}".format(filename))
        if not filename.endswith('.xml'):
            continue
        fullname = os.path.join(dirname,filename)
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
                speakers.append("noname")  #add to speaker list 
            except:
                speaker = item.attrib['speakername']
                speakers.append(speaker)
                if len(speakers) < 2:  # first speaker                               
                    boundary = "[SAME]"
                else: # thereafter
                    if speakers[0] == speakers[1]:
                        boundary = "[SAME]"
                    else:                      
                        boundary = "[CHANGE]"

            notags = ET.tostring(item, encoding='unicode', method='text')
            notags = re.sub(r'(\s{2,})', ' ', notags)
            punkt = sent_detector.tokenize(notags)
            for s in punkt:
                sentences.append(s.strip())
                sentences.append("[SAME]")
            sentences.pop(-1) # idk how to do this nicer, tried for s in punkt[:-1]
            sentences.append(boundary)
            
    #print(sentences)
    sentences.pop(-1)
    return sentences


def write_sents_to_csv(sentences, filename):
    with open(filename, 'w') as wfile:
        writer = csv.writer(wfile)
        # write header
        writer.writerow(["Sentence 1", "Sentence 2", "Boundary"])
        for index in range(len(sentences)):
            try:
                sent1 = sentences[index]
                sent2 = sentences[index + 1]
                rg_boundary = r'\[(SAME|CHANGE)\]'
                # get the boundary from the end of the first sentence. We get group(1) so that we don't get the 
                # surrounding square brackets
                boundary = re.search(
                    rg_boundary, sent1[-10:]).group(1)
                # remove the [SAME] or [CHANGE] from the end of the sentence
                sent1 = re.sub(rg_boundary, '', sent1).strip()
                sent2 = re.sub(rg_boundary, '', sent2).strip()
                writer.writerow([sent1, sent2, boundary])
            except IndexError:
                pass


print(config.data_dir)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = process_data_tags("speech", config.data_dir)
#write_sents_to_csv(sentences, config.CSV_FILENAME)
