from nltk.corpus import stopwords
import re

dataset = 'ohsumed'
stop = set(stopwords.words('english'))

with open(dataset+'.txt', 'r', encoding='latin1') as f:
  x = f.readlines()

temp = []
for s in x:
  s = s.lower()
  s = re.sub('[-+]?\d*\.\d+|\d+', 'number', s)
  s = re.sub('[?|!|\'|"|#|:|;|`|=]', ' ', s)
  s = re.sub('[%|$|&|*|^|+|.|,|\|/|\[|\]]', ' ', s)
  s = re.sub('-', ' ', s)
  s = re.sub('\(', ' ( ', s)
  s = re.sub('\)', ' ) ', s)
  words = [w for w in s.split() if w not in stop]
  temp.append(words)

with open(dataset+'.clean.txt', 'w', encoding='utf-8') as f:
  for row in temp:
    seq = ''
    for word in row:
      seq = seq + ' ' + word
    f.writelines(seq + '\n')
