import os
import re
import nltk.tokenize.punkt
import xml.etree.ElementTree as ET
from config import data_dir

print(data_dir)

def process_data_tags(tag, dirname):
    text = ""
    for filename in os.listdir(dirname):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(dirname, filename)
        tree = ET.parse(fullname)
        root = tree.getroot()
        speech_tags = root.findall("speech")
        speakers = []
        for item in speech_tags:           
            try:  # check if there is no name for speaker
                item.attrib['nospeaker']
                boundary = "[CHANGE]" # should there be a different tag if we don't know if they change?
            except:                
                while len(speakers) < 2:    
                    speaker = item.attrib['speakername']
                    speakers.append(speaker)
                    #print(speaker)
                #print(speakers)
                if speakers[0] == speakers[1]:
                    boundary = "[SAME]"
                else:
                    boundary = "[CHANGE]"
        
            speakers = []
            notags = ET.tostring(item, encoding='unicode', method='text')  
            notags = re.sub(r'(\s{2,})', '', notags)
            text = text + notags + " "+boundary+" " 
            #print(ET.dump(item))
            
    print(text)
    return text

process_data_tags("speech", data_dir)