import os
import re
import nltk.data
import nltk.tokenize.punkt 
import xml.etree.ElementTree as ET
from config import data_dir

print(data_dir)

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def process_data_tags(tag, dirname):
    """ Takes a speech tag as string and a directory name. Processes all XML files in the directory """
    sentences = []
    for filename in os.listdir(dirname):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(dirname, filename)
        tree = ET.parse(fullname)
        root = tree.getroot()
        speech_tags = root.findall("speech")
        speakers = []
        for item in speech_tags:
            #print(ET.dump(item)) # use this to print roots           
            try:  # check if there is no name for speaker
                item.attrib['nospeaker']
                boundary = "[SAME]"
            except:                
                while len(speakers) < 2: # check if two speakers are the same    
                    speaker = item.attrib['speakername']
                    speakers.append(speaker)
                if speakers[0] == speakers[1]:
                    boundary = "[SAME]"
                else:
                    boundary = "[CHANGE]"        
            speakers = [] # reset check            
            notags = ET.tostring(item, encoding='unicode', method='text') 
            notags = re.sub(r'(\s{2,})', ' ', notags)
            punkt = sent_detector.tokenize(notags)           
            for s in punkt:
                sentences.append(s+ " "+boundary) # add boundaries to every sentence 
    print(sentences)
    return sentences

process_data_tags("speech", data_dir)