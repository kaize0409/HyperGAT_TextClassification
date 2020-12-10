import argparse
import pickle
import time
import numpy as np
from utils import split_validation, Data
from preprocess import *
from model import *
from sklearn.utils import class_weight
import random
import warnings

warnings.filterwarnings('ignore') 


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='R52', help='dataset name')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--initialFeatureSize', type=int, default=300, help='initial size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate') 
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-6, help='l2 penalty')  
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--rand', type=int, default=1234, help='rand_seed')
parser.add_argument('--use_LDA', action='store_true', help='use LDA to construct semantic hyperedge')



args = parser.parse_args()
print(args)


SEED = args.rand
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



def main():
	doc_content_list, doc_train_list, doc_test_list, vocab_dic, labels_dic, max_num_sentence, keywords_dic, class_weights = read_file(args.dataset, args.use_LDA)
	
	train_data, valid_data = split_validation(doc_train_list, args.valid_portion, SEED)
	test_data = split_validation(doc_test_list, 0.0, SEED)

	num_categories = len(labels_dic)
	
	train_data = Data(train_data, max_num_sentence, keywords_dic, num_categories, args.use_LDA)
	valid_data = Data(valid_data, max_num_sentence, keywords_dic, num_categories, args.use_LDA)
	test_data = Data(test_data, max_num_sentence,  keywords_dic, num_categories, args.use_LDA)
	

	model = trans_to_cuda(SessionGraph(args, class_weights, len(vocab_dic)+1, len(labels_dic)))
	
	for epoch in range(args.epoch):
		print('-------------------------------------------------------')
		print('epoch: ', epoch)
		
		train_model(model, train_data, args)

		valid_detail, valid_acc = test_model(model, valid_data, args, False)
		detail, acc = test_model(model, test_data, args, False)
		print('Validation Accuracy:\t%.4f, Test Accuracy:\t%.4f'% (valid_acc,acc))
		
		

if __name__ == '__main__':

	main()