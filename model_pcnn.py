#input to the pcnn will be a batch of sentences. loss has to be propagated after processing each bag
#optimal param: window = 3, filters = 230, vec_dim = 50, pos_dim = 5, bs = 50, ada_p = 0.95, ada_e = 1e-6, dropout = 0.5
import torch
import torch.nn as nn
from torch.autograd import Variable as Variable
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import glob
import sys
import pdb
import pprint
import json
import logging
import logging.config
from os import path, remove
import json
from pathlib import Path
import yaml
import re
import time
import argparse
with open('config.yml', 'r') as configfile:
	cfg = yaml.load(configfile)

class PCNN(nn.Module):
	def __init__(self, vocab_sz, vec_dim, n_filters, window, dropout, n_classes, word_vecs, max_sent_len, pos_dim):
		super(PCNN, self).__init__()
		self.word_embed = nn.Embedding(vocab_sz, vec_dim, padding_idx = 0)
		self.left_embed = nn.Embedding(2*max_sent_len + 5, pos_dim)
		self.right_embed = nn.Embedding(2*max_sent_len + 5, pos_dim)
		self.conv = nn.Conv2d(1, n_filters, kernel_size=[window, vec_dim+2*pos_dim], padding=(window-1, 0))
		self.drop = nn.Dropout(dropout)
		self.linear = nn.Linear(n_filters*3, n_classes)
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.parameters())
	def init_weights(self):
		self.word_embed.data.copy_(torch.from_numpy(word_vecs))

	def forward(self, inp, ldist, rdist, epos):
		#inp is batchsize X max_sent_len_in_the_batch
		inp_embed = self.word_embed(inp)
		ldist_embed = self.left_embed(ldist)
		rdist_embed = self.right_embed(rdist)
		inp = torch.cat([inp_embed, ldist_embed, rdist_embed], inp_embed.dim()-1)
		#inp is batchsize X max_sent_len X 60
		inp = torch.unsqueeze(inp, 1)
		#inp is batchsize X 1 X max_sent_len X 60
		out = self.conv(inp)
		#out is batchsize X n_filters X max_sent_len+window-1 X 1
		pool = self.get_pool(out, epos)
		#pool is batchsize X n_filters*3
		pool = pool.view(pool.size(0), -1)
		# print(pool.size())
		out = self.linear(self.drop(pool))
		#out is batchsize X n_classes
		return out

	def get_pool(self, out, epos):
		concat_ls = []
		for idx, e in enumerate(epos):
			elem = out.narrow(0, idx, 1)
			size1 = int(e[0])
			size2 = int(e[1] - e[0])
			size3 = int(out.size(2) - e[1])
			# print(size1, size2, size3, int(e[0]), int(e[1]))
			if size1 == 0:
				size1 = 1
			if size1 == size2:
				size2 = size1 + 1
			pool1 = F.max_pool2d(elem.narrow(2, 0, size1), (size1, 1))
			pool2 = F.max_pool2d(elem.narrow(2, int(e[0]), size2), (size2, 1))
			pool3 = F.max_pool2d(elem.narrow(2, int(e[1]), size3),(size3, 1))
			concat = torch.cat((pool1, pool2, pool3), out.dim()-1)
			concat_ls.append(concat)
		final_out = torch.cat(concat_ls, 0)
		return final_out

def get_batch(bag, bsz, augment=True):
        '''
        this function creates batches of a bag. It repeats the elements from
        the beginning of the bag if bag size is less than batch size
        '''
	batches = []
	ldists = []
	rdists = []
	eposs = []
	tbatch = []
	tl = []
	tr = []
	te = []
	cnt = 0
	# print(f'bag len {len(bag)}')
	for e1, e2, sent, id_sent, ldist, rdist in bag:
		try:
			e = [sent.split(' ').index(e1), sent.split(' ').index(e2)]
			e.sort()
			# print(e)
			te.append(e)
		except:
			# print(f'exception: {e1} | {e2} | {sent}')
			continue
		tbatch.append(id_sent)
		tl.append(ldist)
		tr.append(rdist)
		cnt += 1
		if cnt == bsz:
			batches.append(tbatch)
			ldists.append(tl)
			rdists.append(tr)
			eposs.append(te)
			tbatch = []
			tl = []
			tr = []
			te = []
			cnt = 0
	if cnt != 0 and augment:
		i = 0
		while i < (len(bag)):
			e1, e2, sent, id_sent, ldist, rdist = bag[i]
			try:
				e = [sent.split(' ').index(e1), sent.split(' ').index(e2)]
				e.sort()
				# print(e)
				te.append(e)
			except:
				# print(f'exception: {e1} | {e2} | {sent}')
				continue
			tbatch.append(id_sent)
			tl.append(ldist)
			tr.append(rdist)
			cnt += 1
			i = (i+1)%len(bag)
			if cnt == bsz:
				batches.append(tbatch)
				ldists.append(tl)
				rdists.append(tr)
				eposs.append(te)
				tbatch = []
				tl = []
				tr = []
				te = []
				break
	# print("batches: ", len(batches))
	for i, batch in enumerate(batches):
		max_sent_len = 0
		for sent in batches[i]:
			if len(sent) > max_sent_len:
				max_sent_len = len(sent)
		for j, sent in enumerate(batches[i]):
			batches[i][j] = sent + [0]*(max_sent_len - len(sent))
			# if len(bag) == 13: print(len(sent), "sent len", sent)
		for j, ldist in enumerate(ldists[i]):
			ldists[i][j] = ldist + [0]*(max_sent_len - len(ldist))
		for j, rdist in enumerate(rdists[i]):
			rdists[i][j] = rdist + [0]*(max_sent_len - len(rdist))
		
	# if len(bag) == 13:
	# 	for sent in batches[0]:
	# 		print(len(sent), "before returning")
	return (batches, ldists, rdists, eposs)

def train(bags, target, model, epochs, bsz):
	total_loss = 0.0
	model.train()
	for epoch in range(epochs):
		for i, bag in enumerate(bags):
			outs = []
			sents, ldists, rdists, eposs = get_batch(bag, bsz)
			if len(sents) < 1:
				continue
			max_sent = 0
			max_batch = 0
			max_val = 0
			for bnum in range(len(sents)):
				inp = Variable(torch.from_numpy(np.array(sents[bnum])).cuda())
				ldist = Variable(torch.from_numpy(np.array(ldists[bnum])).cuda())
				rdist = Variable(torch.from_numpy(np.array(rdists[bnum])).cuda())
				epos = Variable(torch.from_numpy(np.array(eposs[bnum])).cuda())
				target_tensor = Variable(torch.from_numpy(np.array([target[i]])).cuda())
				out = model(inp, ldist, rdist, epos)
				out = out.data.cpu().numpy()
				# print(out[:,target[i]])
				if np.max(out, 0)[target[i]] > max_val:
					max_val = np.max(out, 0)[target[i]]
					max_sent = np.argmax(out, 0)[target[i]]
					max_batch = bnum
					# print(max_sent, max_val)
			try:
				inp = Variable(torch.from_numpy(np.array(sents[max_batch][max_sent])).cuda())
			except:
				print(max_batch, max_sent)
				sys.exit()
			ldist = Variable(torch.from_numpy(np.array(ldists[max_batch][max_sent])).cuda())
			rdist = Variable(torch.from_numpy(np.array(rdists[max_batch][max_sent])).cuda())
			epos = Variable(torch.from_numpy(np.array(eposs[max_batch][max_sent])).cuda())
			inp = torch.unsqueeze(inp, 0)
			ldist = torch.unsqueeze(ldist, 0)
			rdist = torch.unsqueeze(rdist, 0)
			epos = torch.unsqueeze(epos, 0)
			# print(inp)

			target_tensor = Variable(torch.from_numpy(np.array([target[i]])).cuda())
			out = model(inp, ldist, rdist, epos)
			# print(out.data, target_tensor.data)
			loss = model.criterion(out, target_tensor)
			total_loss += loss.data[0]
			loss.backward()
			model.optimizer.step()
			print(f'processed bag {i+1}/{len(bags)} total_loss = {total_loss}', end = '\r')
		print()
		eval(corpus.dev_x, corpus.dev_y, model, bsz)
		print("--------------------------")
		total_loss = 0

def eval(bags, target, model, bsz):
	model.eval()
	corrects = 0
	for i, bag in enumerate(bags):
		outs = []
		sents, ldists, rdists, eposs = get_batch(bag, bsz)
			#enusre to ignore sentences with all distances = 0
		if len(sents) < 1:
			continue
		for bnum in range(len(sents)):
			inp = Variable(torch.from_numpy(np.array(sents[bnum])).cuda())
			ldist = Variable(torch.from_numpy(np.array(ldists[bnum])).cuda())
			rdist = Variable(torch.from_numpy(np.array(rdists[bnum])).cuda())
			epos = Variable(torch.from_numpy(np.array(eposs[bnum])).cuda())
			target_tensor = Variable(torch.from_numpy(np.array([target[i]])).cuda())
			out = model(inp, ldist, rdist, epos)
			out = out.data.cpu().numpy()
			preds = np.argmax(out, 1)
			if target[i] in preds:
				corrects += 1
				break
	print(f'dev accuracy = : {corrects/len(bags)}')


if __name__ == '__main__':
	from data_processing import *
	with open(cfg['data_path'] + '/corpus.p', 'rb') as f:
		corpus = pickle.load(f)
	n_tr = len(corpus.tr_x)
	n_te = len(corpus.te_x)
	max_bag_len = 0
	min_bag_len = 10000
	for bag in corpus.tr_x:
		if len(bag) < min_bag_len:
			min_bag_len = len(bag)
		if len(bag) > max_bag_len:
			max_bag_len = len(bag)
	print('training bags: {}, testing bags: {}, max_bag_len: {}, min_bag_len:{}'.format(n_tr,n_te, max_bag_len, min_bag_len))
	# print(corpus.tr_x[0])
	# for bag in corpus.tr_x[:3]:
	# 	sents, ldists, rdists, eposs = get_batch(bag, 3)
	# 	print(sents, ldists, rdists, eposs)
		# sys.exit()
	vocab_sz = len(corpus.vocab.id2word)
	vec_dim = 50
	n_filters = 230
	window = 3
	dropout = 0.5
	n_classes = 53
	word_vecs = None
	epochs = 5
	max_sent_len = 75
	pos_dim = 5
	bsz = 5
	model = PCNN(vocab_sz, vec_dim, n_filters, window, dropout, n_classes, word_vecs, max_sent_len, pos_dim)
	model.cuda()
	train(corpus.tr_x, corpus.tr_y, model, epochs, bsz)
	# eval(corpus.dev_x, corpus.dev_y, model, bsz)
