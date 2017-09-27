import codecs
import sys

def get_train_data(list_folders_name):
	X = []
	Y = []
	task_labels = [] # keeps track of where instances come from "task1" or "task2"..
	
        # word 2 indices and tag 2 indices
	w2i = {} # word to index
	c2i = {} # char to index
	task2tag2idx = {} # id of the task -> tag2idx
	
	w2i["_UNK"] = 0  # unk word / OOV
	
	for i, folder_name in enumerate( list_folders_name ):
		num_sentences=0
		num_tokens=0
		task_id = 'task'+str(i)
		if task_id not in task2tag2idx:
			task2tag2idx[task_id] = {}
		for instance_idx, (words, tags) in enumerate(read_conll_file(folder_name)):
			num_sentences += 1
			instance_word_indices = [] #sequence of word indices
			instance_char_indices = [] #sequence of char indices 
			instance_tags_indices = [] #sequence of tag indices
			
		for i, (word, tag) in enumerate(zip(words, tags)):
			num_tokens += 1
			
                    # map words and tags to indices
			if word not in w2i:
				w2i[word] = len(w2i)
			instance_word_indices.append(w2i[word])
		
			if tag not in task2tag2idx[task_id]:
				task2tag2idx[task_id][tag]=len(task2tag2idx[task_id])

			instance_tags_indices.append(task2tag2idx[task_id].get(tag))

			X.append(instance_word_indices)
			Y.append(instance_tags_indices)
			task_labels.append(task_id)

	if num_sentences == 0 or num_tokens == 0:
		sys.exit( "No data read from: "+folder_name )
		
	print("TASK "+task_id+" "+folder_name, file=sys.stderr )
	print("%s sentences %s tokens" % (num_sentences, num_tokens), file=sys.stderr)
	print("%s w features" % len(w2i), file=sys.stderr)

	assert(len(X)==len(Y))
	return X, Y, task_labels, w2i, task2tag2idx  #sequence of features, sequence of labels, necessary mappings



def read_conll_file(file_name, raw=False):
	current_words = []
	current_tags = []
	
	for line in codecs.open(file_name, encoding='utf-8'):
        #line = line.strip()
		line = line[:-1]

		if line:
			if raw:
				current_words = line.split() ## simple splitting by space
				current_tags = ['DUMMY' for _ in current_words]
				yield (current_words, current_tags)
			else:
				if len(line.split("\t")) != 2:
					if len(line.split("\t")) == 1: # emtpy words in gimpel
						raise IOError("Issue with input file - doesn't have a tag or token?")
					else:
						print("erroneous line: {} (line number: {}) ".format(line), file=sys.stderr)
						exit()
				else:
					word, tag = line.split('\t')
					current_words.append(word)
					current_tags.append(tag)

		else:
			if current_words and not raw: #skip emtpy lines
				yield (current_words, current_tags)
			current_words = []
			current_tags = []


	if current_tags != [] and not raw:
		yield (current_words, current_tags)

def get_data_as_instances(folder_name, task, w2i, task2tagidx, raw=False):
	X, Y = [],[]
	org_X, org_Y = [], []
	task_labels = []
	word_indices = []
	for (words, tags) in read_conll_file(folder_name, raw=raw):
		for word in words:
			if word in w2i:
				word_indices.append(w2i[word])
			else:
				word_indices.append(w2i["_UNK"])
		tag_indices = [task2tagidx[task].get(tag) for tag in tags]
		X.append(word_indices)
		Y.append(tag_indices)
		org_X.append(words)
		org_Y.append(tags)
		task_labels.append( task )
	return X, Y, org_X, org_Y, task_labels
