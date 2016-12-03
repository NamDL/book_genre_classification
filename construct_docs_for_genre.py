__author__ = 'supriyaanand'
# Script to format topic word_probability distribution into a single file composed of al words for the topics
# generated per genre
# arg1 is the file containing the topics word distribution output from the topic modeler script with the timestamps
# stripped
# arg2 is the name of the file to be created as a document most representative of a genre in terms of words
# neglecting word ordering

import sys

with open(sys.argv[1]) as fh:
    contents = fh.readlines()

words = []
for line in contents:
    line = line.strip()
    prob_word = line.split(" + ")
    words.extend([x.split("*")[1].replace('"', '') for x in prob_word])

with open(sys.argv[2], 'w') as fh:
    fh.write("%s\n" % " ".join(words))
