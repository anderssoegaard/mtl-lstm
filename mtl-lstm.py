from lib.mioc import read_conll_file, get_train_data, get_data_as_instances, load_embeddings_file
from lib.mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor

import argparse,random,sys
import dynet as dy
import numpy as np

def predict(word_indices, task_id, wembeds, predictors):
    dy.renew_cg()
    features = [wembeds[w] for w in word_indices]
    prev = features
    prev_rev = features
    for i in range(0,args.layers):
        predictor = predictors["inner"][i]
        forward_sequence, backward_sequence = predictor.predict_sequence(prev, prev_rev)        
        forward_sequence = [dy.tanh(s) for s in forward_sequence]
        backward_sequence = [dy.tanh(s) for s in backward_sequence]

        if i == args.layers-1:
            output_predictor = predictors["outer"][task_id]
            concat_layer = [dy.concatenate([f, b]) for f, b in zip(forward_sequence,reversed(backward_sequence))]
            output = output_predictor.predict_sequence(concat_layer)
            return output
        prev = forward_sequence
        prev_rev = backward_sequence 
    return None

def pick_neg_log(pred, gold):
    return -dy.log(dy.pick(pred, gold))

def evaluate(test_X, test_Y, org_X, org_Y, w2i, task2tag2idx, task_labels, wembeds, predictors, output_predictions=None, verbose=True, raw=False):
    correct = 0
    total = 0.0
    if output_predictions != None:
        i2w = {w2i[w] : w for w in w2i.keys()}
        task_id = task_labels[0] 
        i2t = {task2tag2idx[task_id][t] : t for t in task2tag2idx[task_id].keys()}
    for i, (word_indices, gold_tag_indices, task_of_instance) in enumerate(zip(test_X, test_Y, task_labels)):
        output = predict(word_indices, task_of_instance, wembeds, predictors)
        predicted_tag_indices = [np.argmax(o.value()) for o in output] 
        if output_predictions:
            prediction = [i2t[idx] for idx in predicted_tag_indices]            
            words = org_X[i]
            gold = org_Y[i]
            
            for w,g,p in zip(words,gold,prediction):
                if raw:
                    print(u"{}\t{}".format(w, p)) 
                else:
                    print(u"{}\t{}\t{}".format(w, g, p))
            print("")

        correct += sum([1 for (predicted, gold) in zip(predicted_tag_indices, gold_tag_indices) if predicted == gold])
        total += len(gold_tag_indices)
    return correct, total


parser = argparse.ArgumentParser(description="""Multi-task learning with bi-LSTMs""")
parser.add_argument("--train", nargs='*', help="train folder for each task") 
parser.add_argument("--test", nargs='*', help="test file", required=False) 
parser.add_argument("--iters", help="training iterations [default: 30]", required=False,type=int,default=30)
parser.add_argument("--layers", help="layers [default: 1]", required=False,type=int,default=1)
parser.add_argument("--in_dim", help="input dimensions [default: 30]", required=False,type=int,default=30)
parser.add_argument("--h_dim", help="hidden layer dimensions [default: 30]", required=False,type=int,default=30)
parser.add_argument("--dev", help="dev file(s)", required=False) 
parser.add_argument("--embeds_file", help="embeddings file", required=False)
args = parser.parse_args()

model = dy.ParameterCollection() # l218
train_X, train_Y, task_labels, w2i, task2t2i, tasks_ids  = get_train_data(args.train) # l254
trainer=dy.SimpleSGDTrainer(model) #l287

# Build computation graph

dy.renew_cg()
if args.embeds_file:
    print("loading embeddings", file=sys.stderr)
    embeddings, emb_dim = load_embeddings_file(args.embeds_file,args.in_dim)
    assert(emb_dim==args.in_dim)
    num_words=len(set(embeddings.keys()).union(set(w2i.keys()))) 
    wembeds = model.add_lookup_parameters((num_words, args.in_dim))
    init=0
    l = len(embeddings.keys())
    for word in embeddings.keys():
        if word not in w2i:
            w2i[word]=len(w2i.keys()) # add new word
        wembeds.init_row(w2i[word], embeddings[word])
        init+=1
    print("initialized: {}".format(init), file=sys.stderr)
else:
    wembeds = model.add_lookup_parameters((len(w2i), args.in_dim)) #l376

layers = []

for layer_num in range(args.layers): #l411
    if layer_num == 0:
        f_builder = dy.LSTMBuilder(1, args.in_dim, args.h_dim, model)
        b_builder = dy.LSTMBuilder(1, args.in_dim, args.h_dim, model)
        layers.append(BiRNNSequencePredictor(f_builder, b_builder))
    else:
        f_builder = dy.LSTMBuilder(1, args.h_dim, args.h_dim, model)
        b_builder = dy.LSTMBuilder(1, args.h_dim, args.h_dim, model)
        layers.append(BiRNNSequencePredictor(f_builder, b_builder))

predictors={}
predictors["inner"]=layers
predictors["outer"]={}
for task_id in tasks_ids:
    task_num_labels= len(task2t2i[task_id])
    predictors["outer"][task_id] = FFSequencePredictor(Layer(model, args.h_dim*2, len(task_labels), dy.softmax))
    
# TRAINING

train_data=list(zip(train_X,train_Y,task_labels))
print("%d training instances..." % len(train_data))
for iter in range(args.iters):
    print("Iteration:"+str(iter))
    total_loss=0.0
    total_tagged=0.0
    random.shuffle(train_data)
    j=0
    for (word_indices, y, task_of_instance) in train_data:
        j+=1
        if j%200==0:
            print('\t'+str(j)+" instances...")
        output = predict(word_indices, task_of_instance, wembeds, predictors) 
        total_tagged += len(word_indices)
        loss1 = dy.esum([pick_neg_log(pred,gold) for pred, gold in zip(output, y)]) 
        lv = loss1.value()
        total_loss += lv
        loss1.backward()
        trainer.update()

    print("iter {2} {0:>12}: {1:.2f}".format("total loss",total_loss/total_tagged,iter),file=sys.stderr)
    if args.dev:
        dev_X,dev_Y,org_X,org_Y,dev_task_labels=get_data_as_instances(args.dev,"task0",w2i,task2t2i)
        correct, total = evaluate(dev_X, dev_Y, org_X, org_Y, w2i, task2t2i, dev_task_labels, wembeds, predictors)
        val_accuracy = correct/total
        print("\ndev accuracy: %.4f" % (val_accuracy), file=sys.stderr)
             
# One file per test ...
for i, test in enumerate( args.test ):
    test_X, test_Y, org_X, org_Y, task_labels = tagger.get_data_as_indices(test, "task0",w2i,task2t2i)
    correct, total = tagger.evaluate(test_X, test_Y, org_X, org_Y, w2i,task2t2ti,task_labels,wembeds, predictors)
    print("\nTask%s test accuracy on %s items: %.4f" % (i, i+1, correct/total), file=sys.stderr)
            
