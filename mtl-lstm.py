#import subprocess
#subprocess.call(["export","DYLD_LIBRARY_PATH=/Users/soegaard/anaconda/lib/:$DYLD_LIBRARY_PATH"])
from lib.mio import read_conll_file, get_train_data, get_data_as_instances
from lib.mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor

import argparse,random,sys
import dynet as dy
import numpy as np

parser = argparse.ArgumentParser(description="""Multi-task learning with bi-LSTMs""")
parser.add_argument("--train", nargs='*', help="train folder for each task") # allow multiple train files, each asociated with a task = position in the list
parser.add_argument("--iters", help="training iterations [default: 30]", required=False,type=int,default=30)
parser.add_argument("--layers", help="layers [default: 2]", required=False,type=int,default=2)
parser.add_argument("--in_dim", help="input dimensions [default: 30]", required=False,type=int,default=30)
parser.add_argument("--h_dim", help="hidden layer dimensions [default: 30]", required=False,type=int,default=30)
parser.add_argument("--dev", help="dev file(s)", required=False) 
args = parser.parse_args()

# MODEL ARCHITECTURE

model = dy.ParameterCollection()

train_X, train_Y, task_labels, w2i, task2t2i = get_train_data(args.train)

wembeds = model.add_lookup_parameters((len(w2i), args.in_dim))

layers = []

for layer_num in range(0,args.layers):
    if layer_num == 0:
        f_builder = dy.LSTMBuilder(1, args.in_dim, args.h_dim, model)
        b_builder = dy.LSTMBuilder(1, args.in_dim, args.h_dim, model)
    else:
        f_builder = dy.LSTMBuilder(1, args.h_dim, args.h_dim, model)
        b_builder = dy.LSTMBuilder(1, args.h_dim, args.h_dim, model)
    layers.append(BiRNNSequencePredictor(f_builder, b_builder))

predictors={}
predictors["inner"]=layers
predictors["outer"]=FFSequencePredictor(Layer(model, args.h_dim*2, len(task_labels), dy.softmax))

def predict(word_indices, task_id, train=False):
        """
        predict tags for a sentence represented as char+word embeddings
        """
        
        # word embeddings
        features = [wembeds[w] for w in word_indices]
        
        # go through layers
        prev = features
        prev_rev = features

        for i in range(0,args.layers):
            predictor = predictors["inner"][i]
            forward_sequence, backward_sequence = predictor.predict_sequence(prev, prev_rev)        
            if i > 0:
                forward_sequence = [dy.tanh(s) for s in forward_sequence]
                backward_sequence = [dy.tanh(s) for s in backward_sequence]

            if i == 1:
                output_predictor = predictors["outer"]
                concat_layer = [dy.concatenate([f, b]) for f, b in zip(forward_sequence,reversed(backward_sequence))]

                output = output_predictor.predict_sequence(concat_layer)
                return output

            prev = forward_sequence
            prev_rev = backward_sequence 

        return None

def pick_neg_log(pred, gold):
        return -dy.log(dy.pick(pred, gold))

def evaluate(test_X, test_Y, org_X, org_Y, w2i, task2tag2idx, task_labels, output_predictions=None, verbose=True, raw=False):
        correct = 0
        total = 0.0

        if output_predictions != None:
            i2w = {w2i[w] : w for w in self.w2i.keys()}
            task_id = task_labels[0] # get first
            i2t = {task2tag2idx[task_id][t] : t for t in task2tag2idx[task_id].keys()}

        for i, (word_indices, gold_tag_indices, task_of_instance) in enumerate(zip(test_X, test_Y, task_labels)):
            if verbose:
                if i%100==0:
                    sys.stderr.write('%s'%i)
                elif i%10==0:
                    sys.stderr.write('.')

            output = predict(word_indices, task_of_instance)
            predicted_tag_indices = [np.argmax(o.value()) for o in output]  # logprobs to indices
            if output_predictions:
                prediction = [i2t[idx] for idx in predicted_tag_indices]

                words = org_X[i]
                gold = org_Y[i]

                for w,g,p in zip(words,gold,prediction):
                    if raw:
                        print(u"{}\t{}".format(w, p)) # do not print DUMMY tag when --raw is on
                    else:
                        print(u"{}\t{}\t{}".format(w, g, p))
                print("")

            correct += sum([1 for (predicted, gold) in zip(predicted_tag_indices, gold_tag_indices) if predicted == gold])
            total += len(gold_tag_indices)

        return correct, total

# TRAINING

train_data=list(zip(train_X,train_Y,task_labels))
for iter in range(args.iters):
    total_loss=0.0
    total_tagged=0.0
    random.shuffle(train_data)
    for (word_indices, y, task_of_instance) in train_data:
         dy.renew_cg() # new graph per item
         output = predict(word_indices, task_of_instance, train=True) 
         total_tagged += len(word_indices)
         loss1 = dy.esum([pick_neg_log(pred,gold) for pred, gold in zip(output, y)]) 
         lv = loss1.value()
         total_loss += lv
         loss1.backward()
         trainer=dy.SimpleSGDTrainer(model)
         trainer.update()
         if args.dev:
             
             dev_X,dev_Y,org_X,org_Y,dev_task_labels=get_data_as_instances(args.dev,"task0",w2i,task2t2i) # check this
             # evaluate after every epoch
             correct, total = evaluate(dev_X, dev_Y, org_X, org_Y, w2i, task2t2i, dev_task_labels) # implement this
             val_accuracy = correct/total
             print("\ndev accuracy: %.4f" % (val_accuracy), file=sys.stderr)
             
