# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:43:48 2017

@author: BALASUBRAMANIAM
"""

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
'''
nlp = StanfordCoreNLP(configdict={
    'annotators': "tokenize,ssplit,pos,lemma,parse",
    'parse.model': 'edu/stanford/nlp/models/srparser/englishSR.ser.gz'},  
    corenlp_jars=["E:/software/stanford-corenlp-full-2017-06-09/to/stanford-corenlp-full-2015-04-20/*", "E:/software/to/stanford-srparser-2014-10-23-models.jar"])
'''

res = nlp.annotate("""I love you.  I hate him. get lost. You are nice. He is dumb.
                   The best way to hope for any chance of enjoying 
                   this film is by lowering your expectations.The show starts out as competent 
                   but unremarkable and gradually grows in to a considerable power. will kill you. don't come. fool""",
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json'
                   })
#print(res)
#The average sentiment of tweets is between Neutral (2) and Negative (1), the range is from 
#VeryNegative (0) to VeryPositive (4) which appear to be quite rare.
for s in res["sentences"]:
    print ("%d: '%s': %s %s" % (
        s["index"],
        " ".join([t["word"] for t in s["tokens"]]),
        s["sentimentValue"], s["sentiment"]))
    
text = (
        'Pusheen and Smitha walked along the beach. Pusheen wanted to surf,'
        'but fell off the surfboard.')
output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse',
        'outputFormat': 'json'
    })
print(output['sentences'][0]['parse'])
output = nlp.tokensregex(text, pattern='/Pusheen|Sumitha/', filter=False)
print(output)
output = nlp.semgrex(text, pattern='{tag: VBD}', filter=False)
print(output)