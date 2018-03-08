import numpy as np
import sys
import pdb
import json
import logging
import logging.config
from os import path, remove
import json
from pathlib import Path
import yaml
import re
from sklearn.model_selection import train_test_split
with open('config.yml', 'r') as configfile:
	cfg = yaml.load(configfile)
import pprint
import operator
import pickle
def get_logger(file_name):
	if path.isfile(file_name):
    		remove(file_name)

	config_dict = json.load(open('log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = file_name.replace('/', '-')
	logging.config.dictConfig(config_dict)

	logger = logging.getLogger(file_name)
	logger.addHandler(logging.StreamHandler(sys.stdout))
	# logger.info('Completed configuring logger!')
	return logger

class Vocab:
	def __init__(self, vec_filename, rel_filename):
		self.word2id = {}
		self.id2word = []
		self.word2vec = []
		self.load_vecs(vec_filename)
		self.tmpw2v = []
		self.rel2id = {}
		self.id2rel = []
		self.load_rel(rel_filename)

	def load_rel(self, rel_filename):
		with open(rel_filename, 'r') as f:
			lines = f.read().split('\n')
		for r in lines:
			r = r.split(' ')
			if len(r) <= 1:
				continue
			self.rel2id[r[0]] = int(r[1])
			self.id2rel.append(r[0])

	def load_vecs(self, vec_filename):
		with open(vec_filename, 'r') as f:
			line1 = f.readline().split(' ')
			lines = f.read().split('\n')
		i = 1
		vocab_sz = int(line1[0])
		vec_dim = int(line1[1].replace('\n', ''))
		self.word2vec = np.ndarray((vocab_sz+1, vec_dim))
		self.id2word.append('UNK')
		self.word2id['UNK'] = 0
		self.word2vec[0] = np.zeros(vec_dim)
		for l in lines:
			l = l.split(' ')
			if len(l) <= 1:
				continue
			self.word2id[l[0]] = i
			self.word2vec[i] = np.array(l[1:], dtype = np.float)
			self.id2word.append(l[0])
			i += 1
	def add_word(self, word):
		if word not in self.word2id:
			self.id2word.append(word)
			self.word2id[word] = len(self.id2word) - 1
			# np.append(self.word2vec, np.random.uniform(-0.25, 0.25, (1, len(self.word2vec[0]))), axis=0)
			self.tmpw2v.append(np.random.uniform(-0.25, 0.25, len(self.word2vec[0])))

def make_tab_sep(test_file_name, new_file_name):
	'''
	input: sharmishtha's riedel test data file
	output: each line's sentence and end marker is now tab separated
	'''
	new_f = open(new_file_name, 'w')
	with open(test_file_name, 'r') as f:
		for line in f:
			m1, m2, e1, e2, r, sent_end = line.split('\t')
			end = sent_end.split(' ')[-1]
			sent = ' '.join(sent_end.split(' ')[:-1])
			new_line = '\t'.join([m1, m2, e1, e2, r, sent, end])
			new_f.write(new_line)
	new_f.close()
def add_underscores(filename, new_file_name):
	new_f = open(new_file_name, 'w')
	with open(filename, 'r') as f:
		for line in f:
			m1, m2, e1, e2, r, sent, end = line.split('\t')
			if r == 'NA':
				continue
			e1_ = e1.replace('_', ' ')
			e2_ = e2.replace('_', ' ')
			if len(e1) > len(e2):
				sent_new = sent.replace('_', ' ')
				sent_new = sent_new.replace(e1_, e1)
				sent_new = sent_new.replace(e2_, e2)
			else:
				sent_new = sent.replace('_', ' ')
				sent_new = sent_new.replace(e2_, e2)
				sent_new = sent_new.replace(e1_, e1)
			new_f.write('\t'.join([m1, m2, e1, e2, r, sent_new, end]))
	new_f.close()
def get_statistics(filename, logger):
	'''
	input: filename of riedel dataset: train or test
	output: stats - types and counts of relations, number of sentences, vocab_sz, avg sent_len, types of entities, their freq
	'''
	rel2freq = {}
	n_sents = 0
	vocab = Vocab()
	sent_lens = []
	ent2freq = {}
	ent2rel = {}
	with open(filename, 'r') as f:
		for line in f:
			m1, m2, e1, e2, r, sent, end = line.split('\t')
			sent_lens.append(len(sent.split(' ')))
			if r in rel2freq:
				rel2freq[r] += 1
			else:
				rel2freq[r] = 1
			for word in sent.split(' '):
				vocab.add_word(word)
			if e1 in ent2freq:
				ent2freq[e1] += 1
			else:
				ent2freq[e1] = 1
			if e2 in ent2freq:
				ent2freq[e2] += 1
			else:
				ent2freq[e2] = 1
			if e1+e2 in ent2rel:
				ent2rel[e1+e2].add(r)
			else:
				ent2rel[e1+e2] = set()
				ent2rel[e1+e2].add(r)
	n_multiple = 0
	n_total = len(ent2rel)
	for e in sorted(ent2rel.items(), key=operator.itemgetter(1)):
		if len(e[1]) > 1:
			n_multiple += 1
	logger.info(f'{filename}\n entities = {len(ent2freq)} vocab_sz = {len(vocab.id2word)} n_sent = {len(sent_lens)} avg_sent_len = {np.mean(np.array(sent_lens))}')
	for rel in sorted(rel2freq.items(), key=operator.itemgetter(1)):
		logger.info(rel)
	logger.info(f'total e pairs: {n_total} and multiples: {n_multiple}')
	for e in sorted(ent2freq.items(), key=operator.itemgetter(1), reverse=True)[:20]:
		logger.info(e)

class Corpus():
	def __init__(self, train_file_name, test_file_name, vec_filename, rel_filename):
		# self.sents = sents
		self.vocab = Vocab(vec_filename, rel_filename)
		self.ent2rel = self.get_entity_to_rel([train_file_name, test_file_name])
		self.ent2label = self.get_entity_to_label(self.ent2rel)
		self.train_bags_x = {}
		self.train_bags_y = {}
		self.test_bags_x = {}
		self.test_bags_y = {}
		self.make_bags(train_file_name, test_file_name)
		self.tr_x, self.dev_x, self.tr_y, self.dev_y = train_test_split(list(self.train_bags_x.values()), list(self.train_bags_y.values()), test_size=cfg['test_size'], random_state = cfg['random_state'])
		self.te_x, self.te_y = list(self.test_bags_x.values()), list(self.test_bags_y.values())
		self.normalise_distances(self.tr_x)
		self.normalise_distances(self.dev_x)
		self.normalise_distances(self.te_x)
		self.vocab.word2vec = np.append(self.vocab.word2vec, self.vocab.tmpw2v, axis=0)

	def get_entity_to_rel(self, filenames):
		ent2rel = {}
		for fname in filenames:
			with open(fname, 'r') as f:
				for line in f:
					m1, m2, e1, e2, r, sent, end = line.split('\t')
					if r == 'NA':
						continue
					ent2rel[e1+e2] = r
					# TODO: handle case when two entities occur in more than two relations
		return ent2rel
	def get_entity_to_label(self, ent2rel):
		ent2label = {}
		for label, e in enumerate(ent2rel):
			ent2label[e] = label
		return ent2label
	def make_bags(self, train_file_name, test_file_name):
		'''
		input: file name from riedel. each element is tab separated. 
		output: a list of bags, each bag has a list of sentences, each sentence is a list of token ids
		'''
		with open(train_file_name, 'r') as f:
			for line in f:
				m1, m2, e1, e2, r, sent, end = line.split('\t')
				if r == 'NA' or len(sent.split()) > 75 or r not in self.vocab.rel2id:
					continue
				bag_label = self.ent2label[e1+e2]
				if bag_label in self.train_bags_x:
					self.train_bags_x[bag_label].append((e1, e2, sent, self.get_ids(sent), self.get_dist(sent, e1), self.get_dist(sent, e2)))
				else:
					self.train_bags_x[bag_label] = [(e1, e2, sent, self.get_ids(sent), self.get_dist(sent, e1), self.get_dist(sent, e2))]
					self.train_bags_y[bag_label] = self.vocab.rel2id[r]

		with open(test_file_name, 'r') as f:
			for line in f:
				m1, m2, e1, e2, r, sent, end = line.split('\t')
				if r == 'NA' or len(sent.split()) > 75 or r not in self.vocab.rel2id:
					continue
				bag_label = self.ent2label[e1+e2]
				if bag_label in self.test_bags_x:
					self.test_bags_x[bag_label].append((e1, e2, sent, self.get_ids(sent), self.get_dist(sent, e1), self.get_dist(sent, e2)))
				else:
					self.test_bags_x[bag_label] = [(e1, e2, sent, self.get_ids(sent), self.get_dist(sent, e1), self.get_dist(sent, e2))]
					self.test_bags_y[bag_label] = self.vocab.rel2id[r]
					
	def normalise_distances(self, bags):
		max_d = 0
		min_d = 500
		min_sent = ""
		max_sent_len = 0
		for bag in bags:
			for (e1, e2, sent, sent_, d1, d2) in bag:
				if len(sent.split()) > max_sent_len:
					max_sent_len = len(sent.split())
				if np.max(d1) > max_d:
					max_d = np.max(d1)
				if np.max(d2) > max_d:
					max_d = np.max(d2)
				if np.min(d1) < min_d:
					min_d = np.min(d1)
					min_sent = sent
				if np.min(d2) < min_d:
					min_d = np.min(d2)
					min_sent = sent
		for bag in bags:
			for i, (e1, e2, sent, sent_, d1, d2) in enumerate(bag):
				d1 = list(np.add(d1, -1*min_d))
				d2 = list(np.add(d2, -1*min_d))
				bag[i] = (e1, e2, sent, sent_, d1, d2)
		print(min_d, max_d, max_sent_len, min_sent)
	def get_ids(self, sent):
		ids = []
		for w in sent.split(' '):
			self.vocab.add_word(w)
			ids.append(self.vocab.word2id[w])
		return ids
	def get_dist(self, sent, e):
		try:
			dist = []
			epos = sent.split(' ').index(e)
			for i, w in enumerate(sent.split(' ')):
				dist.append(i-epos)
		except:
			print(e)
			return [0]*len(sent.split(' '))
		return dist


def check(filename):
	with open(filename, 'r') as f:
		for line in f:
			m1, m2, e1, e2, r, sent, end = line.split('\t')
			if e1 not in sent or e2 not in sent:
				print(e1, e2, sent)
if __name__ == '__main__':
	train_file_name = cfg['data_path'] + '/tr_tab_un.txt'
	test_file_name = cfg['data_path'] + '/te_tab_un.txt'
	vec_filename = cfg['data_path'] + '/vec.txt'
	rel_filename = cfg['data_path'] + '/relation2id.txt'
	# logger = get_logger('data_log')
	# get_statistics(cfg['data_path'] + '/train_tab_sep.txt', logger)
	# get_statistics(cfg['data_path'] + '/test.txt')
	# add_underscores(test_file_name, cfg['data_path'] + '/te_tab_un.txt')
	corpus = Corpus(train_file_name, test_file_name, vec_filename, rel_filename)
	print(f'vocab size is: {len(corpus.vocab.id2word)} word dimensions: {len(corpus.vocab.word2vec[0])}')
	print(len(corpus.tr_x), len(corpus.te_x), len(corpus.dev_x))
	print(len(corpus.ent2rel))
	with open(cfg['data_path'] + '/corpus.p', 'wb') as f:
		pickle.dump(corpus, f)
	# print(f'')

	# print(f'total bags: {len(corpus.bags_x)}')
	# corpus = Corpus(cfg['data_path'] + '/test.txt')
	# max_bag_len = 0
	# for i, bag in enumerate(corpus.tr_x):
	# 	if len(bag) > max_bag_len :
	# 		max_bag_len = len(bag)
	# print(max_bag_len)
	# print(corpus.tr_x[0])
	# pprint.pprint(corpus.te_x[0])
	# pprint.pprint(corpus.tr_x[0])

	# print(len(corpus.vocab.id2word))
	# check(cfg['data_path'] + '/tr_tab_un.txt')