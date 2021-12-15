# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
import string
from wordcloud import WordCloud

!pip install sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from IPython.core.display import HTML

df = pd.read_csv('/content/sts.csv')

df = df.replace(to_replace = ['MUITO_INSATISFEITO','INSATISFEITO', 'INDIFERENTE', 'SATISFEITO', 'MUITO_SATISFEITO'], value = [1,2,3,4,5])

df.describe()

sns.heatmap(df.isnull(), cbar=False)

df = df.dropna(how='any').reset_index(drop = True)

df['LENGTH'] = df['OBSERVACAO'].apply(len)

df['LENGTH'].plot(bins=100, kind='hist')

df[df['LENGTH'] < 25]

df.drop(df.loc[df['LENGTH'] < 25].index, inplace=True)

df = df.reset_index(drop = True)

df = df.drop(columns='LENGTH')

df.describe()

df['VACINACAO'] = df['OBSERVACAO'].apply(lambda x: 1 if 'vacin' in x else 0)

vcn = df[df['VACINACAO'] == 1].reset_index(drop = True)

vcn = vcn.drop(columns='VACINACAO')

vcp = vcn[vcn['GRAU_SATISFACAO'] >= 4].reset_index(drop = True)
vcn = vcn[vcn['GRAU_SATISFACAO'] <= 3].reset_index(drop = True)

vcn.describe()

vcp.describe()

vcn.describe()

vcn.hist(bins = 30, figsize=(10,7), color = '#33adffff')

vcp.hist(bins = 30, figsize=(10,7), color = '#33adffff')

vcn.hist(bins = 30, figsize=(10,7), color = '#33adffff')

plt.figure(figsize = (10,7))
sns.barplot(x = 'VERSAO', y = 'GRAU_SATISFACAO', data = vcn, palette=sns.light_palette("#0099ffff", n_colors=15, reverse=True))

plt.figure(figsize = (10,7))
sns.barplot(x = 'VERSAO', y = 'GRAU_SATISFACAO', data = vcp, palette=sns.light_palette("#0099ffff", n_colors=15, reverse=True))

plt.figure(figsize = (10,7))
sns.barplot(x = 'VERSAO', y = 'GRAU_SATISFACAO', data = vcn, palette=sns.light_palette("#0099ffff", n_colors=15, reverse=True))

sts = vcn['OBSERVACAO'].tolist()
stp = vcp['OBSERVACAO'].tolist()
stn = vcn['OBSERVACAO'].tolist()

def non(sents):
  _s = []
  for s in sents:
    s = s.replace('\n', '')
    s = s.replace('  ', ' ')
    _s.append(s)
  return _s

sts = non(sts)
stp = non(stp)
stn = non(stn)

s = ' '.join(sts)
p = ' '.join(stp)
n = ' '.join(stn)

nltk.download('punkt')
nltk.download('stopwords')

parser = PlaintextParser.from_string(s, Tokenizer('portuguese'))

sumLuhn = LuhnSummarizer()

reLuhn = sumLuhn(parser.document, 10)

hgh = ''

for s in reLuhn:
  hgh += str(s)

_t = ''

display(HTML(f'<h1>Vacinação</h1>'))
for s in sts:
  if s in hgh:
    _t += str(s).replace(s, f'<mark>{s} </mark>')
  else:
    _t += s + ' '
display(HTML(f'{_t}'))

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(hgh))

stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords.extend(['aí', 'ser', 'sistema', 'nao', 'ter', 'toda', 'vezes', 'pois', 'muita', 'pra', 'poderia', 'poder', 'fica', 'ficou', 'ficar', 'muitos', 'muitas', 'dia', 'versão', 'sempre', 'demais', 'exemplo', 'tudo', 'ainda', 'colocar', 'programa', 'paciente', 'atendimento', 'deveria', 'consigo', 'aparece'])

def format(input):
  _l = input.lower()
  _tks = []
  for t in nltk.word_tokenize(_l):
    _tks.append(t)

  _tks = [p for p in _tks if p not in stopwords and p not in string.punctuation]

  _f = ' '.join([str(e) for e in _tks])

  return _f

hgh = format(hgh)

plt.figure(figsize=(20,10))
plt.imshow(WordCloud().generate(hgh))

parser = PlaintextParser.from_string(stp, Tokenizer('portuguese'))

reLuhn = sumLuhn(parser.document, 10)

hgh = ''

for s in reLuhn:
  hgh += str(s)

_t = ''

display(HTML(f'<h1>Vacinação - Positivos</h1>'))
for s in stp:
  if s in hgh:
    _t += str(s).replace(s, f'<mark>{s} </mark>')
  else:
    _t += s + ' '
display(HTML(f'{_t}'))

hgh = format(hgh)

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(hgh))

parser = PlaintextParser.from_string(stn, Tokenizer('portuguese'))

reLuhn = sumLuhn(parser.document, 10)

hgh = ''

for s in reLuhn:
  hgh += str(s)

_t = ''

display(HTML(f'<h1>Vacinação - Negativos</h1>'))
for s in stn:
  if s in hgh:
    _t += str(s).replace(s, f'<mark>{s} </mark>')
  else:
    _t += s + ' '
display(HTML(f'{_t}'))

hgh = format(hgh)

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(hgh))
