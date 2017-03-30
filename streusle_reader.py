"""
Created Sept 2016
@author M Regan
Reading Streusle verb supersense data; clustering using sklearn; compare results of algorithms; visualize
Chaotic implementation
"""

import csv
import json
import re

import pandas as pd
pd.set_option('display.width', 1000)

import numpy as np
from collections import defaultdict 
from collections import Counter

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Processing Streusle (Amalgram) preposition and verb supersenses; 
# Using dependency parse features (NLP4J); adding Glove word vectors
# Clustering to see what comes up (kmeans); which semantic spaces could be linked to force dynamic patterns
# Lots of preprocessing necessary (dependency parses are messy)

path='/path/to/streusle.tags.sst'

dependency_path='/path/to/streusle_dependency.txt.nlp'


streusle_dict={}

prep_hier_count_dict=defaultdict(int)
ind_prep_count_dict=defaultdict(int)
verb_ss_count_dict=defaultdict(int)
vsst_psst_collocation_dict=defaultdict(int)
initial_psst_with_first_vsst_dict=defaultdict(int)  #this is the non-hierarchical form

hierarchical_initial_psst_with_first_vsst_dict=defaultdict(int)

hierarchical_prep_dict={}

feature_dict={}  #this would be better called preliminary_feature or preprocessing_features

dependency_dict={}

final_feature_dict={}

features_ready_for_preprocessing_dict={}  

reduced_final_feature_dict={}

dictionary_vectors_averaged={} 
dictionary_vectors_concatenated={}


one_hot_all_dict={}  #one_hot_all_dict is used for the dependency-based, one-hot encoding clustering


def create_streusle_dict():
    
    with open(path) as f:
    	reader = csv.reader(f, delimiter="\t")
    	d=list(reader)
    
    for item in d:
        try:
            key=item[0]
            sentence=item[1]
            tag_set=json.loads(item[2])
            tag_set['sentence']=sentence
            streusle_dict[key]=tag_set
        except IndexError:
            print("Error with: ", item[0])
            continue


def display_streusle_dict(): 
    i=0   
    for k, v in streusle_dict.items():
        print(k) 
        print('SENT:', v['sentence'])
        print('TAGS:', v['tags'])
        print('LEMMAS:', v['lemmas'])
        print('LABELS:', v['labels'])
        print('WORDS:', v['words'])
        print('\n')
        i+=1
        if i>5:
            break
            
            

def basic_analytics():
    
    count_sentences=0
    count_words=0
    
    for v in streusle_dict.values():
        count_sentences+=1
        count_words+=len(v['tags'])
        
    print("Total number of sentences: ", count_sentences)
    print("Total number of words: ", count_words)
    print("Average length of sentence: ", count_words/count_sentences)
    

def find_nsst_types():
    
    nsst_count_dict=defaultdict(int)
    
    
    for v in streusle_dict.values():
        labels=v['labels']   # 'labels' is a nested dictionary
        for item in labels.values():
            if item[1].isupper():
                nsst_count_dict[item[1]]+=1
                
    df=pd.DataFrame(sorted(list(nsst_count_dict.items())), columns=['nsst', 'count'])
    #df['prob']=pd.Series(df['count']/total_count_v_ss, index=df.index)
    #print(df)
    
    print(df.to_latex())
            
    
def deprecated_output_sentences_for_dependency_parsing():
    
    _path='/Users/Michael/Desktop/streusle_dependency.txt'
    file_path='/Users/Michael/Desktop/streusle_IDs.txt'
    
    with open(_path, 'w') as f, open(file_path, 'w') as g:
        for k, v in streusle_dict.items():
            sentence=v['sentence']
            
            sentence=sentence.split()   #getting rid of those pesky ellipses ...
            sentence_with_less_ellipsis=[]
            for i in sentence:
                if i in ['..', '...', '....', '.....', '......'] and sentence.index(i)!=-1:
                     sentence_with_less_ellipsis.append(',')
                else:
                    sentence_with_less_ellipsis.append(i)
            
            sentence=sentence_with_less_ellipsis
            
            sentence_with_punct_inside_quotes=[]
            for i in enumerate(sentence):
                this_index=i[0]
                this_token=i[1]
                if this_token in ['!', '?']:
                    try:
                        if sentence[this_index+1] in ["\"", "\'"] and this_index+1==len(sentence):
                            sentence_with_punct_inside_quotes.append(',')
                        else:
                            sentence_with_punct_inside_quotes.append('.')
                            
                    except IndexError:
                        sentence_with_punct_inside_quotes.append(this_token)
                else:
                    if this_token in ["\"", "\'"] and this_index==len(sentence)-1:    #getting rid of final quote so it doesn't link to following sentence
                        pass
                    else:
                        sentence_with_punct_inside_quotes.append(this_token)
                    
            sentence=sentence_with_punct_inside_quotes
            
            
            sentence_with_no_dot_websites=[]  #because these dots also mess up the sentence breaks in nlp4j
            for i in sentence:
                m = re.search(r'.com', i)
                if m is not None: 
                    new_string='website'
                    sentence_with_no_dot_websites.append(new_string)
                else:
                    sentence_with_no_dot_websites.append(i)
                    
            sentence=sentence_with_no_dot_websites
            
            sentence_with_converted_LST_punct=[]  #because LST punct can also mess up parsing
            for i in enumerate(sentence):
                this_index=i[0]
                this_token=i[1]
                if this_token in ['.'] and this_token!=sentence[-1] and sentence[this_index-1].isdigit()==True:
                    new_string=')'
                    sentence_with_converted_LST_punct.append(new_string)
                else:
                    sentence_with_converted_LST_punct.append(this_token)
            
            sentence=sentence_with_converted_LST_punct   
                    
            sentence=" ".join(sentence)
            
            
            if v['words'][-1][0] not in [".", "\"", "\'", "!", "?", ")", "]", "-"]:
                sentence=sentence+" ."  #adding punctuation for split between sentences; 

            f.write(sentence)
            f.write('\n')
            g.write(k)
            g.write('\n')
            
            
"""
using NLP4J
command line: 
bin/nlpdecode -c config-decode-en.xml -i /path/to/streusle_dependency.txt 
"""
     
     
     
def cleaner_output_sentences_for_dependency_features():
    
    _path='/path/to/streusle_dependency.txt'
    #file_path='/path/to/streusle_IDs.txt'

    with open(_path, 'w') as f:#, open(file_path, 'w') as g:
        for k, v in streusle_dict.items():
            #lemmas=v['lemmas']
            lemmas=v['sentence'].split()  #trying to preserve MWEs, which are not preserved in actual lemma list
            tags=v['tags']  
            words=v['words']
            
            lemmas.pop()
            stop_words=[x for x in lemmas if x in ['.', '..', '...', '....', '.....', '......']]
            if len(stop_words)>0:
                pass
            else: 
                zip_list=zip(tags, words)
            
                sentence_list=[]
                for i in zip_list: 
                    tag=i[0]
                    if '-' in tag:
                        tag=tag.split("-", 1)[1]
                    else: 
                        tag='O'  #default
                    
                    word=i[1][0]
                    if '.' in word: word=word.replace('.', '')
                    if '\'' in word: word=word.replace('\'', '')
                    
                    pos=i[1][1]

                    feature=(tag, word, pos)
                    if pos not in ['.', "''", '``', "'", '-LRB-', '-RRB-', 'LS']:
                        sentence_list.append(feature)
                sentence_list.append(('O', '.'))
                if len(sentence_list)>3:
                
                    feature_dict[k]=sentence_list
                    
                    """commented out as this file has already been created"""
                    
                    # new_sentence=[]
                    # for i in sentence_list:
                    #     new_sentence.append(i[1])
                    # sentence=sentence=" ".join(new_sentence)
                    # f.write(k + '.')
                    # f.write('\n')
                    # f.write(sentence)
                    # f.write('\n')


                
def read_dependency_data():
    
    with open(dependency_path, 'r') as f:
    	reader = csv.reader(f, delimiter="\t")
    	d=list(reader)
      
    key=''   
    for i in d:
        if i==[]: pass

        elif i[0]=='2' and i[1]=='.': pass
        elif 'ewtb' in i[1]: 
            key=i[1]
            dependency_dict[key]=[]
        else:
            dependency_dict[key].append(i)
            
    for k, v in dependency_dict.items():
        temp_dep=v
        temp_feature=feature_dict[k]
        new_features=[]
        zip_list=zip(temp_dep, temp_feature)
        for i in zip_list:
            dep=i[0]
            feat=i[1][0]
            dep.append(feat)
            new_features.append(dep)
        dependency_dict[k]=new_features
            
    # for k, v in dependency_dict.items():
    #     print(k)
    #     for i in v:
    #         print(i)
    #     print('\n')

    print("length", len(dependency_dict)) #3384

            
def create_final_features():
    
    # writing to final_feature_dict; 
    # key will be tuple (fileID, token, vsst); value will be list of features
    
    nsubjnsubjpass_dict=defaultdict(int)
    nsubjnsubjpassPOS_dict=defaultdict(int)
    dobj_dict=defaultdict(int)
    dobjPOS_dict=defaultdict(int)
    prep_dict=defaultdict(int)
    pobj_dict=defaultdict(int)
    pobjPOS_dict=defaultdict(int)
    # prep2_dict=defaultdict(int)
    # pobj2_dict=defaultdict(int)
    
    

    for k, v in dependency_dict.items():
        for i in v:
            pos=i[3]
            sst=i[9]
            
            if pos in ['VB', 'VBP', 'VBZ', 'VBN', 'VBD', 'VBG'] and sst.islower()==True and sst not in ['O', '`a', '`', '`j', '`d']:
                
                root_index=i[0]
                token=i[1]
                vsst=sst
                feature_representation=[] 
                myKey=(k, token, vsst)
                for j in v:
                    if j[5]==root_index and j[3] in ['IN'] and len(j[9])>2 and j[9][0].isupper()==True:  #adding prepositions (obliques)
                        if j[9][1].islower()==True:
                            if j not in feature_representation:
                                feature_representation.append(j)
                            prepos_index=j[0]
                            for m in v:
                                #if m[0]==prepos_index and len(m[9])>1 and m[9].isupper()==True:  #adding objects of prepositions with nssts
                                if m[5]==prepos_index:
                                    feature_representation.append(m)
                    #elif j[5]==root_index and len(j[9])>1 and j[9].isupper()==True:  #adding arguments that have nssts
                    elif j[5]==root_index:
                        if j[3] not in ['RB', 'WRB', 'CC', 'MD', 'RP', 'TO', 'WP', 'WDT'] and j[6] not in ['aux', 'punct', 'auxpass'] and j[9] not in ['`a', '`', '`j', '`d']:
                            feature_representation.append(j)
                        elif j[3] in ['RB'] and j[9] not in ['O', '`r', '`d']:
                            feature_representation.append(j)
                    elif abs(int(j[5])-int(root_index))<3 and len(j[9])>2 and j not in feature_representation: # add ssts that are near root
                         if j[6]!='root' and j[2]!=token:
                             feature_representation.append(j)
                             
                
                #creating list of all nssts; these are not used in the end
                for i in feature_representation:
                    dep=i[6]
                    pos=i[3]
                    sst=i[9]
                    if dep in ['nsubj', 'nsubjpass']:
                        nsubjnsubjpass_dict[sst]+=1
                        nsubjnsubjpassPOS_dict[pos]+=1
                    elif dep == 'dobj':
                        dobj_dict[sst]+=1
                        dobjPOS_dict[pos]+=1
                    elif dep == 'prep':
                        prep_dict[sst]+=1
                    elif dep == 'pobj':
                        pobj_dict[sst]+=1 
                        pobjPOS_dict[pos]+=1                    
                    
                        
                thisArray=np.array(feature_representation)
                final_feature_dict[myKey]=thisArray

                
    # removing all statives
    for k, v in final_feature_dict.items():
        if k[2]=='stative': pass
        elif k[2]=='`i': pass
        else:
            reduced_final_feature_dict[k]=v
            
    cnt_mean=0       
    for k, v in reduced_final_feature_dict.items():
        cnt_mean+=len(v)
        print(k, v)
        
    # print()
    # print("Average number of dependency features per verb supersense: ", cnt_mean/len(reduced_final_feature_dict))
    #
    # pause=input("enter")

    for k, v in reduced_final_feature_dict.items():

        if len(v)>2:
            myFeat=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] 
            #nsubjsst, nsubjpos, dobjsst, dobjpos, prep1sst, pobj1sst, pobj1pos, prep2sst, pobj2sst, pobj2pos, grabbag1sst, grabbag2sst, verb lemma

            for i in v:
                pos=i[3]
                dep=i[6]
                sst=i[9]
                if dep in ['nsubj', 'nsubjpass']:
                    myFeat[0]=sst
                    myFeat[1]=pos
                elif dep == 'dobj':
                    myFeat[2]=sst
                    myFeat[3]=pos
                elif dep == 'prep':
                    if myFeat[4]==-1:
                       myFeat[4]=sst
                    else:
                        myFeat[7]=sst 
                elif dep == 'pobj':
                    if myFeat[5]==-1:
                       myFeat[5]=sst
                       myFeat[6]=pos
                    else:
                        myFeat[8]=sst
                        myFeat[9]=pos
                else:
                    if myFeat[10]==-1 and len(sst)>2:
                        myFeat[10]=sst
                    elif len(sst)>2:
                        myFeat[11]=sst
                        
                myFeat[12]=k[1]  #adding the lemma of the verb predicated itself
                
            features_ready_for_preprocessing_dict[k]=myFeat
           
    count_vsst_dict=defaultdict(int)
    for k in features_ready_for_preprocessing_dict.keys():
        count_vsst_dict[k[2]]+=1
    
    print()
    
    df_cnt=pd.DataFrame(list(sorted(count_vsst_dict.items())), columns=['vsst', 'count'])
    print(df_cnt.to_latex())
    pause=input("enter")
    # a few items that can be sorted out: ??, XX

    
def kmeans_dependency_features():

    complete_features_list=[]
    for k, v in features_ready_for_preprocessing_dict.items():
        for i in v:
            if i not in complete_features_list:
                complete_features_list.append(i)
                
    le=LabelEncoder()
    le.fit(complete_features_list)
    #print(list(le.classes_))
    
    label_encoded_dict={}
    all_labels_encoded_list=[]
    for k, v in features_ready_for_preprocessing_dict.items():
        trans_v=le.transform(v)
        label_encoded_dict[k]=trans_v
        all_labels_encoded_list.append(trans_v)
         
    enc = OneHotEncoder()
    enc.fit(all_labels_encoded_list)
    #print(enc.n_values_)  
    
    

    for k, v in label_encoded_dict.items():
        size = 1
        thisA=enc.transform([v]).toarray()
        #for dim in np.shape(thisA): size *= dim
        thisA=thisA.tolist()
        for i in thisA:
            one_hot_all_dict[k]=i
 
 
    #feature selection  
    
    list_all_one_hot=[]
    labels=[]
    for k, v in one_hot_all_dict.items():
        labels.append(k[2])
        list_all_one_hot.append(v)
        
    labels=np.array(labels)   #labels are the gold standard values for the clustering
    
    sel = VarianceThreshold(threshold=(.995 * (1 - .995)))
    X=sel.fit_transform(list_all_one_hot)

    X=np.array(list_all_one_hot)
    
    
    #clustering
    
    np.random.seed(42)
    
    data = scale(X)
    #not scaling the data gives only slightly better results
    data=X
    
    n_samples, n_features = data.shape
    #print(n_samples, n_features)
    n_vssts = len(np.unique(labels))  #14
    
    sample_size = 300
    
    print("n_Vssts: %d, \t n_samples %d, \t n_features %d"
          % (n_vssts, n_samples, n_features))
          
    print(79 * '_')
    print('% 9s' % 'init'
          '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')
     
            
    def bench_k_means(estimator, name, data):
        t0 = time.time()
        estimator.fit(data)
        print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
              % (name, (time.time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(labels, estimator.labels_),
                 metrics.completeness_score(labels, estimator.labels_),
                 metrics.v_measure_score(labels, estimator.labels_),
                 metrics.adjusted_rand_score(labels, estimator.labels_),
                 metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=sample_size))) 
          
                                      
    a=bench_k_means(KMeans(init='k-means++', n_clusters=n_vssts, n_init=10),
                  name="k-means++", data=data)
                  
    b=bench_k_means(KMeans(init='random', n_clusters=n_vssts, n_init=10),
                  name="random", data=data)
                  
    pca = PCA(n_components=n_vssts).fit(data)
    c=bench_k_means(KMeans(init=pca.components_, n_clusters=n_vssts, n_init=1),
                  name="PCA-based",
                  data=data)
    print(79 * '_')
    
    bench_kmeans_dict={}
    bench_kmeans_dict['k-means++']=a
    bench_kmeans_dict['random']=b
    bench_kmeans_dict['PCA-based']=c
    
    
    
    df_bench_kmeans=pd.DataFrame(list(bench_kmeans_dict.items()))#, columns=['time', 'inertia', 'homo', 'compl', 'v-meas', 'ARI', 'AMI'])
    print(df_bench_kmeans.to_latex())
    pause=input("enter")
    
    
    
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_vssts, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .01     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    with PdfPages('basicdependency.pdf') as pdf:
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means with basic dependency features\n'
                  '(PCA-reduced); Centroids=White cross', fontsize=14)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        pdf.savefig() #uncomment this for a pdf to be saved
        plt.show()
    
    output_content_clusters(kmeans.labels_, labels)



def output_content_clusters(pred_model_labels, actual_labels):    
    
    zip_gold_predicted=zip(pred_model_labels, actual_labels)
    
    evaluation_dict={}  #keys are the predicted clusters; value is a list of all gold standard (not very readable)
    
    for i in zip_gold_predicted:
        pred=i[0]
        gold=i[1]
        if pred not in evaluation_dict.keys():
            evaluation_dict[pred]=[gold]
        else:
            evaluation_dict[pred].append(gold)
            
    
    readable_eval_dict={}     
    for k, v in evaluation_dict.items():
        cnt = Counter()
        for vsst in v:
            cnt[vsst] += 1
        readable_eval_dict[k]=cnt
      
    print()    
    print("Cluster contents")
    for k, v in readable_eval_dict.items():
        print(k, v)
        print()
        



def create_word_vector_features():
    
    """this relies mainly on the info in reduced_final_feature_dict"""
    """first run will NOT use the verb itself, just six tokens in its vicinity, as defined by dependency relations"""
    
    lemmas_for_Glove_vectors_dict={}
    
    print("Length of Final feature dictionary", len(reduced_final_feature_dict))
    print()
    
    for k, v in reduced_final_feature_dict.items():
        if len(v)>2:
            myFeat=[]
            #nsubj, dobj, prep1, pobj1, prep2, pobj2, prep3, pobj3, verb

            for i in v:
                lemma=i[2]
                dep=i[6]
                if dep in ['nsubj', 'nsubjpass']:
                    myFeat.append(lemma)
                elif dep == 'dobj':
                    myFeat.append(lemma)
                elif dep == 'prep':
                    myFeat.append(lemma)
                elif dep == 'pobj':
                    myFeat.append(lemma)
                
            myFeat.append(k[1])
            
            lemmas_for_Glove_vectors_dict[k]=myFeat
     
     
    cnt=0
    count_length_dict=defaultdict(int)
                
    for k, v in lemmas_for_Glove_vectors_dict.items():
        count_length_dict[len(v)]+=1
        print(k, v)
    
    # print()
    # print("distributions of lengths in word vector dict")
    # for k, v in count_length_dict.items():
    #     print(k, v)
        
    # print()
    # print("Ready to read in Glove vector values")
    # pause=input("Enter to continue")
    glove_path='/path/to/glove.6B.300d.txt'        
      
    glove_dict={}


    with open(glove_path) as f:
        for line in f:
            array=line.split()
            word=array[0]
            vector=array[1:]
            glove_dict[word]=vector
            
    for k, v in lemmas_for_Glove_vectors_dict.items():
        #this should give about 2400 data points
        if len(v)>1:
            raw_vector_list=[]
            for i in v:
                if i not in glove_dict:
                    pass
                else:
                    token=i.lower()
                    vector=glove_dict[token]
                    raw_vector_list.append(vector)
            if len(raw_vector_list)>0:
                sum_of_vectors_averaged=sum(np.asarray(raw_vector_list, dtype=float)) / float(len(raw_vector_list))
                dictionary_vectors_averaged[k]=sum_of_vectors_averaged
     
    print()        
    #print(len(dictionary_vectors_averaged)) (with vectors composed of at least two elements)
    print()
    
    
    for k, v in lemmas_for_Glove_vectors_dict.items():
        #this will be a test, vectors will be of different lengths
        if len(v)>3:
            raw_vector_list=[]
            v=v[-1:] + v[:-1]   #putting the vsst first
            for i in v:
                token=i.lower()
                if token not in glove_dict:
                    pass
                else:
                    vector=glove_dict[token]
                    raw_vector_list.append(vector)
            if len(raw_vector_list)>3:
                concatenated_vectors=[]
                for i in raw_vector_list:
                    concatenated_vectors+=i
                    x=np.array(concatenated_vectors)  #converting list of strings to array of real numbers
                    y=x.astype(np.float)
                
                y=y[:1200]
                dictionary_vectors_concatenated[k]=y

    #print(dictionary_vectors_concatenated[('ewtb.r.333243.6', 'purchased', 'possession')])
    print()
    #print("length one example concatenated vector", len(dictionary_vectors_concatenated[('ewtb.r.333243.6', 'purchased', 'possession')]))
    print()
    print("concatenated dictionary length", len(dictionary_vectors_concatenated))
    print()
    #pause=input("Enter")
    
    
    
     
def kmeans_wordvectors():
    
    """will use dictionary_vectors_averaged={} and dictionary_vectors_concatenated={}"""
    
    for thisDict in [dictionary_vectors_concatenated, dictionary_vectors_averaged]:
        
        if thisDict==dictionary_vectors_averaged: 
            thisLabel='summed and averaged word vectors'
        else:
            thisLabel='concatenated word vectors'
    
        #feature selection 
        training_features=[] 
        training_labels=[]
        for k, v in thisDict.items():
            training_labels.append(k[2])
            training_features.append(v)
    
        labels=np.array(training_labels)   #labels are the gold standard values for the clustering

        # sel = VarianceThreshold(threshold=(.995 * (1 - .995)))
        # X=sel.fit_transform(training_features)

        X=np.array(training_features)
    
        #clustering
    
        np.random.seed(42)
    
        #data = scale(X)
        #not scaling the data gives slightly (only slightly) better results
        data=X
    
        n_samples, n_features = data.shape
        n_vssts = len(np.unique(labels))  #14
    
        sample_size = 300
    
        print()
        print("Kmeans with sum and averaged word vectors")
        print()
    
        print("n_Vssts: %d, \t n_samples %d, \t n_features %d"
              % (n_vssts, n_samples, n_features))
          
        print(79 * '_')
        print('% 9s' % 'init'
              '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')
     
            
        def bench_k_means(estimator, name, data):
            t0 = time.time()
            estimator.fit(data)
            print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
                  % (name, (time.time() - t0), estimator.inertia_,
                     metrics.homogeneity_score(labels, estimator.labels_),
                     metrics.completeness_score(labels, estimator.labels_),
                     metrics.v_measure_score(labels, estimator.labels_),
                     metrics.adjusted_rand_score(labels, estimator.labels_),
                     metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
                     metrics.silhouette_score(data, estimator.labels_,
                                              metric='euclidean',
                                              sample_size=sample_size))) 
          
                                      
        bench_k_means(KMeans(init='k-means++', n_clusters=n_vssts, n_init=10),
                      name="k-means++", data=data)
                  
        bench_k_means(KMeans(init='random', n_clusters=n_vssts, n_init=10),
                      name="random", data=data)
                  
        pca = PCA(n_components=n_vssts).fit(data)
        bench_k_means(KMeans(init=pca.components_, n_clusters=n_vssts, n_init=1),
                      name="PCA-based",
                      data=data)
        print(79 * '_')
    
    
        reduced_data = PCA(n_components=2).fit_transform(data)
        kmeans = KMeans(init='k-means++', n_clusters=n_vssts, n_init=10)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        
        with PdfPages(thisLabel+'.pdf') as pdf:
            plt.figure(1)
            plt.clf()
            plt.imshow(Z, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       cmap=plt.cm.Paired,
                       aspect='auto', origin='lower')

            plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
            # Plot the centroids as a white X
            centroids = kmeans.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker='x', s=169, linewidths=3,
                        color='w', zorder=10)
            plt.title('K-means clustering using ' +thisLabel+ '\n'
                      '(PCA-reduced); Centroid=White cross', fontsize=14)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            pdf.savefig()
            plt.show()
     
        
        print("content clusters: ", thisLabel)
        print()
        output_content_clusters(kmeans.labels_, labels)
       



def multiple_clustering():
    
    from sklearn import cluster
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    
    #one-hot, dependency based
    list_all_one_hot=[]
    labels=[]
    for k, v in one_hot_all_dict.items():
        labels.append(k[2])
        list_all_one_hot.append(v)
        
    onehot=(np.array(list_all_one_hot), np.array(labels))
    print(onehot[0].shape, onehot[1].shape)
    
    #concatenated word vectors
    conc_training_features=[] 
    conc_training_labels=[]
    for k, v in dictionary_vectors_concatenated.items():
        conc_training_labels.append(k[2])
        conc_training_features.append(v)
    
    concatenated=(np.array(conc_training_features), np.array(conc_training_labels))
    print(concatenated[0].shape, concatenated[1].shape)
        
    #summed and averaged word vectors
    
    summed_training_features=[] 
    summed_training_labels=[]
    for k, v in dictionary_vectors_averaged.items():
        summed_training_labels.append(k[2])
        summed_training_features.append(v)

    summed=(np.array(summed_training_features), np.array(summed_training_labels))
    print(summed[0].shape, summed[1].shape)
    
    pause=input("enter: shapes of features above")
    
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    
    clustering_names = ['Ward']#['Ward', 'AgglomerativeClustering', 'DBSCAN']
    
    plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1

    datasets = [onehot, concatenated, summed]
    for i_dataset, dataset in enumerate(datasets):
        X, y = dataset
            
        # normalize dataset for easier parameter selection
        #X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # create clustering estimators

        ward = cluster.AgglomerativeClustering(n_clusters=13, linkage='ward',
                                               connectivity=connectivity)
        # dbscan = cluster.DBSCAN(eps=.2)
        #
        # average_linkage = cluster.AgglomerativeClustering(
        #     linkage="average", affinity="cityblock", n_clusters=13,
        #     connectivity=connectivity)

        clustering_algorithms = [ward]#[ward, average_linkage, dbscan]

        for name, algorithm in zip(clustering_names, clustering_algorithms):
            # predict cluster memberships
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            # plot
            plt.subplot(4, len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)
            plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

            if hasattr(algorithm, 'cluster_centers_'):
                centers = algorithm.cluster_centers_
                center_colors = colors[:len(centers)]
                plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

    plt.show()
    
    
    
def ward_clustering():
    
    from sklearn.cluster import AgglomerativeClustering
    #from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    import mpl_toolkits.mplot3d.axes3d as p3
    
    #one-hot, dependency based
    list_all_one_hot=[]
    labels=[]
    for k, v in one_hot_all_dict.items():
        labels.append(k[2])
        list_all_one_hot.append(v)
        
    onehot=(np.array(list_all_one_hot), np.array(labels))
    
    #concatenated word vectors
    conc_training_features=[] 
    conc_training_labels=[]
    for k, v in dictionary_vectors_concatenated.items():
        conc_training_labels.append(k[2])
        conc_training_features.append(v)
    
    concatenated=(np.array(conc_training_features), np.array(conc_training_labels))
        
    #summed and averaged word vectors
    
    summed_training_features=[] 
    summed_training_labels=[]
    for k, v in dictionary_vectors_averaged.items():
        summed_training_labels.append(k[2])
        summed_training_features.append(v)

    summed=(np.array(summed_training_features), np.array(summed_training_labels))

    
    clustering_names = ['Ward']#['Ward', 'AgglomerativeClustering', 'DBSCAN']

    datasets = [onehot, concatenated, summed]
    for i_dataset, dataset in enumerate(datasets):
        X, _ = dataset
        
        if dataset==onehot:
            thisLabel='basic dependencies'
        elif dataset==concatenated:
            thisLabel='concatenated word vectors'
        else:
            thisLabel='summed and averaged word vectors'
        
            
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        #bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        # connectivity matrix for structured Ward
        #connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # make connectivity symmetric
        #connectivity = 0.5 * (connectivity + connectivity.T)

        # create clustering estimators
        print("Compute unstructured hierarchical clustering...")
        st = time.time()
        ward = AgglomerativeClustering(n_clusters=13, linkage='ward').fit(X)
        elapsed_time = time.time() - st
        label = ward.labels_
        print("Elapsed time: %.2fs" % elapsed_time)
        print("Number of points: %i" % label.size)
        
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(7, -80)
        for l in np.unique(label):
            ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
                      'o', color=plt.cm.jet(np.float(l) / np.max(label + 1)))
        plt.title('Ward clustering of ' + thisLabel + '\n Without connectivity constraints (time %.2fs)' % elapsed_time)
        plt.show()
      

def ward_clustering_take_two():
    
    from sklearn import manifold
    
    vssts=['body', 'change',  'cognition', 'communication', 'competition', 'consumption', 
            'contact', 'creation', 'emotion', 'motion', 'perception', 'possession', 'social']
    
    print("Key for vssts as seen in graphs for Ward hierarchical clustering")
    for i in enumerate(vssts):
        print(i[0], i[1])
    
    #one-hot, dependency based
    list_all_one_hot=[]
    labels=[]
    for k, v in one_hot_all_dict.items():
        index_vsst=vssts.index(k[2])
        labels.append(index_vsst)
        list_all_one_hot.append(v)
        
    onehot=(np.array(list_all_one_hot), np.array(labels))
    
    #concatenated word vectors
    conc_training_features=[] 
    conc_training_labels=[]
    for k, v in dictionary_vectors_concatenated.items():
        index_vsst=vssts.index(k[2])
        conc_training_labels.append(index_vsst)
        conc_training_features.append(v)
    
    concatenated=(np.array(conc_training_features), np.array(conc_training_labels))
        
    #summed and averaged word vectors
    
    summed_training_features=[] 
    summed_training_labels=[]


    for k, v in dictionary_vectors_averaged.items():
        index_vsst=vssts.index(k[2])
        summed_training_labels.append(index_vsst)  #now labels are indices
        summed_training_features.append(v)

    summed=(np.array(summed_training_features), np.array(summed_training_labels))
    
    
    for ddata in [onehot, concatenated, summed]:
        
        if ddata==onehot:
            thisLabel='dependencies with one-hot'
        elif ddata==concatenated:
            thisLabel='concatenated word vectors'
        else:
            thisLabel='summed and averaged word vectors'    
        
        X, y = ddata
        n_samples, n_features = X.shape
    
        np.random.seed(0)
    
        def plot_clustering(X_red, X, labels, title=None):
            x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
            X_red = (X_red - x_min) / (x_max - x_min)

            plt.figure(figsize=(6, 4))
            for i in range(X_red.shape[0]):
                plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                         color=plt.cm.spectral(labels[i] / 10.),
                         fontdict={'weight': 'normal', 'size': 4})

            plt.xticks([])
            plt.yticks([])
            if title is not None:
                plt.title(title, size=13)
            plt.axis('off')
            plt.tight_layout()

        #----------------------------------------------------------------------
        # 2D embedding of the dataset
        print("Computing embedding")
        X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
        print("Done.")

        from sklearn.cluster import AgglomerativeClustering
    
        labels=y
        data=X_red

        for linkage in ('ward', 'average', 'complete'):
        
            print('Clustering with Ward: ', linkage, 'with ', thisLabel)
            print(79 * '_')
        
            def bench_ward(estimator, name, data):
                t0 = time.time()
                estimator.fit(data)
                print('% 9s   %.2fs    %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
                      % (name, (time.time() - t0),
                         metrics.homogeneity_score(labels, estimator.labels_),
                         metrics.completeness_score(labels, estimator.labels_),
                         metrics.v_measure_score(labels, estimator.labels_),
                         metrics.adjusted_rand_score(labels, estimator.labels_),
                         metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
                         metrics.silhouette_score(data, estimator.labels_,
                                                  metric='euclidean')))#, sample_size=sample_size))) 
                                  
            bench_ward(AgglomerativeClustering(linkage=linkage, n_clusters=13),
                          name=linkage, data=data)

            print(79 * '_')
        
            clustering = AgglomerativeClustering(linkage=linkage, n_clusters=13)
            t0 = time.time()
            clustering.fit(X_red)
            print("%s : %.2fs" % (linkage, time.time() - t0))

        
            with PdfPages('Ward_'+thisLabel+'_'+linkage+'.pdf') as pdf:
                plot_clustering(X_red, X, clustering.labels_, "Clustering "+thisLabel+" %s linkage" % linkage)
                pdf.savefig()
                plt.show()  
    
    

      
      
                  
            
            

def count_verb_supersenses():
    
    """results stored in verb_ss_count_dict"""
    
    vsst='body change cognition communication competition consumption contact creation emotion motion perception possession social stative weather'.split()
    
    for v in streusle_dict.values():
        labels=v['labels']   # 'labels' is a nested dictionary
        for item in labels.values():
            if item[1] in vsst:
                verb_ss_count_dict[item[1]]+=1
                
    print("Count of verb supersenses")
    print('\n')
    
    total_count_v_ss=0
    for k, v in verb_ss_count_dict.items():
        print(k, v)
        total_count_v_ss+=v

    print("Total number of verb supersenses: ", total_count_v_ss)
    print('\n')
    
    df=pd.DataFrame(list(verb_ss_count_dict.items()), columns=['vsst', 'count'])
    df['prob']=pd.Series(df['count']/total_count_v_ss, index=df.index)
    #print(df)
    
    print(df.to_latex())
    
    
    

def count_prep_supersenses():
    
    """Hierarchical categories based on Nathan's list, with some changes, particularly to Attribue and Place; adding Transaction and Approximator"""
    """results stored in hierarchical_prep_dict"""

    Experiencer='Experiencer'.split()
    Stimulus='Stimulus'.split()
    Affector='Agent Causer Creator'.split()
    Coparticipant='Co-Agent Co-Participant Co-Theme'.split()
    Undergoer='Accompanier Instrument Means Activity Beneficiary Patient Theme Topic ProfessionalAspect'.split()
    Manner='Course Transit Via'.split()
    Place='Contour Destination Direction Extent Goal InitialLocation Location Locus Path Source Traversed 1DTrajectory 2DArea 3DMedium'.split()
    Transaction='Donor/Speaker Recipient'.split()
    Approximator='Approximator'.split()
    Attribute='Material Scalar/Rank EndState StartState Value ValueComparison Age Function'.split()
    Temporal='ClockTimeCxn DeicticTime Duration EndTime Frequency RelativeTime StartTime Time'.split()
    Explanation='Purpose Reciprocation'.split()
    Configuration='Elements Instance Possessor Quantity Species Superset Whole'.split()
    Comparison='Comparison/Contrast'.split()
    Infinitive='`i'

    psst=[Experiencer, Stimulus, Affector, Coparticipant, Undergoer, Manner, Place, Transaction, Approximator, 
        Attribute, Temporal, Explanation, Configuration, Comparison, Infinitive]
  
    psst_l=['Experiencer', 'Stimulus', 'Affector', 'Coparticipant', 'Undergoer', 'Manner', 'Place', 'Transaction', 'Approximator', 
            'Attribute', 'Temporal', 'Explanation', 'Configuration', 'Comparison', 'Infinitive']
            
    zip_psst=zip(psst_l, psst)  
    for i in zip_psst:
        myKey=i[0]
        myValue=i[1]
        """ this dictionary will be used later in counting the sequential vssts and pssts"""
        hierarchical_prep_dict[myKey]=myValue
            
    grand_set=[]    #for individual count of prepositions
    for i in psst:
        for j in i:
            grand_set.append(j)
  

    for v in streusle_dict.values():
        labels=v['labels']
        for item in labels.values():
            zip_p=zip(psst, psst_l)
            for tagset in zip_p:
                group=tagset[0]
                lab=tagset[1]
                if item[1] in group:
                    prep_hier_count_dict[lab]+=1
            if item[1] in grand_set:
                ind_prep_count_dict[item[1]]+=1
                
    print('\n')   
    print("Total number of prep supersenses, sorted hierarchically") 
    print('\n')   
    total_count=0    
    for k, v in sorted(prep_hier_count_dict.items()):
        print(k, v)
        total_count+=v
                
    print("Total number of pssts: ", total_count)
    print('\n')
    print("Pssts hierarchical to latex")
    print('\n')
    
    df=pd.DataFrame(list(sorted(prep_hier_count_dict.items())), columns=['psst', 'count'])
    print(df.to_latex())
    
    
    print('\n')
    print('Individual prep counts')
    print('\n')
    
    """output individual preposition count in descending order"""
    total_prep_count=0
    for k, v in [(k, ind_prep_count_dict[k]) for k in sorted(ind_prep_count_dict, key=ind_prep_count_dict.get, reverse=True)]:
    #for k, v in ind_prep_count_dict.items():
        print(k, v)
        total_prep_count+=v

    print('\n')
    #print("Total number of ind preps: ", total_prep_count)
   
    
    
def count_prep_supersenses_per_sentence():
    
    total_p_count_dict=defaultdict(int)

    for v in streusle_dict.values():
        labels=v['labels']   # 'labels' is a nested dictionary
        p_count_per_sent=0
        for item in labels.values():
            if item[1] in ind_prep_count_dict:
                p_count_per_sent+=1
        total_p_count_dict[p_count_per_sent]+=1
        
    print("Number of sentences with given number of PREP supersenses")
    print('\n')
    check_total_num_sents=0
    check_num_preps=0
    for k, v in total_p_count_dict.items():
        print(k, v)
        check_total_num_sents+=v
        check_num_preps+=(k*v)
        
    
    # print('\n')
    # print("Check total # preps: ", check_num_preps)
    # print('\n')
    
    
      
def count_verb_supersenses_per_sentence():
    
    total_V_count_dict=defaultdict(int)

    for v in streusle_dict.values():
        labels=v['labels']   # 'labels' is a nested dictionary
        V_count_per_sent=0
        for item in labels.values():
            if item[1] in verb_ss_count_dict:
                V_count_per_sent+=1
        total_V_count_dict[V_count_per_sent]+=1
        
    print('\n')    
    print("Number of sentences with given number of VERB supersenses")
    print('\n')
    check_num_verb_ss=0
    for k, v in total_V_count_dict.items():
        print(k, v)
        check_num_verb_ss+=(k*v)
        
    
    print('\n')
    print("Check total # verb supersenses: ", check_num_verb_ss)  
    
    

def determine_vsst_psst_collocations():
    
    """will create a sequential list of vsst-psst orderings, for now only those pssts that follow a vsst are counted"""
    
    """Result: vsst_psst_collocation_dict=defaultdict(int) which includes the hierarchical pssts"""
    
    for v in streusle_dict.values():
        labels=v['labels']   # 'labels' is a nested dictionary
        
        have_vsst=False
        vsst_psst_list=[]
        
        
        labelList=[]
        i=0
        for i in sorted(labels):
            """converting to list to be able to determine last item in sentence; 
            items in labels dict are now added sequentially according to their key, which is their index in the sentence"""
            labelList.append(labels[i])  
        
        initial_psst_list=[]  #initializing list to store prepositions at beginning of sentence before first vsst, includes that vsst at last index
        
        for item in labelList:
            
            if have_vsst==False:
                if item[1] in verb_ss_count_dict:
                    vsst_psst_list=[item[1]]
                    initial_psst_list.append(item[1])  #if list is empty, only the first vsst is written
                    initial_psst_tuple=tuple(initial_psst_list)
                    initial_psst_with_first_vsst_dict[initial_psst_tuple]+=1
                    have_vsst=True
                    
                elif item[1] in ind_prep_count_dict:
                    initial_psst_list.append(item[1])
                    
            if have_vsst==True:
                
                """the second set of commands should count the hierarchical prep categories, 
                    while the below counts individual prep instances"""            
                # if item[1] in ind_prep_count_dict and item[1] not in verb_ss_count_dict:
                #     vsst_psst_list.append(item[1])
                
                """the 'not in' statement defines the window to include the prepositions for a specific vsst as those prepositions that follow it before 
                the next vsst in the same sentence; makes the assumption that preceding prepositions are less frequent, therefore less predictive of the vsst
                they precede"""
        
                if item[1] in ind_prep_count_dict and item[1] not in verb_ss_count_dict:
                    for k, v in hierarchical_prep_dict.items():
                        if item[1] in v:
                            vsst_psst_list.append(k)

                elif item[1] in verb_ss_count_dict:
                    vsst_psst_tuple=tuple(vsst_psst_list)
                    vsst_psst_collocation_dict[vsst_psst_tuple]+=1
                    vsst_psst_list=[item[1]]
                    
                elif labelList.index(item)==-1:
                    vsst_psst_tuple=tuple(vsst_psst_list)
                    vsst_psst_collocation_dict[vsst_psst_tuple]+=1
                    
            if have_vsst==False and labelList.index(item)==-1:
                vsst_psst_tuple=tuple(["No vssts"])
                vsst_psst_collocation_dict[vsst_psst_tuple]+=1
    
    cnt=0
    print('\n')
    print("Vsst and psst collocations: ")
    print('\n')
    for k, v in sorted(vsst_psst_collocation_dict.items()):
        print(k, v)
        cnt+=len(k)*v
        
    
    print('\n')
    print("Length of vsst-psst collocation dict: ", len(vsst_psst_collocation_dict.keys()))
    
    print('\n')
    print("sentence-initial pssts with first vsst")
    print('\n')
    for k, v in sorted(initial_psst_with_first_vsst_dict.items()):
        print(k, v)
        cnt+=len(k)*v
        
    print('\n')
    print('\n')
    print('Combined length of all items in the two collocation dictionaries ', cnt)  # a bit more than combined totals of vssts and pssts
    print('\n')
    print('\n')
    
    
    """Making the hierarchical dictionary of sentence initial pssts with the first vsst"""

    print("Converting sentence initial psst dictionary to hierarchical form")
    for k, v in initial_psst_with_first_vsst_dict.items():
        tuple_initial_pssts_w_final_vsst=k
        count=v
        hier_tuple_list=[]
        for item in tuple_initial_pssts_w_final_vsst:
            if item==tuple_initial_pssts_w_final_vsst[-1]:
                hier_tuple_list.append(item)
            else:
                #converting psst to hierarchical psst
                for key, value in hierarchical_prep_dict.items():
                    if item in value:
                        hier_tuple_list.append(key)
            
        hier_tuple=tuple(hier_tuple_list)
        hierarchical_initial_psst_with_first_vsst_dict[hier_tuple]=count
    
    
                
                
def contingency_table_vsst_psst_window_of_4():
    
    """will create contingency table with row values the vssts and columns the hierarchical pssts"""
    """exclude temporal and approximator (for now)"""
    
    all_vsst_with_following_psst_count={}   #dictionary of dictionaries
        
    contact={}
    stative={}
    emotion={}
    motion={}
    creation={}
    possession={}
    communication={}
    competition={}
    perception={}
    cognition={}
    change={}
    social={}
    consumption={}
    body={}
    
    all_dicts=[contact, stative, emotion, motion, creation, possession, communication, competition, perception, cognition, change, social, consumption, body]

    all_dicts_names=['contact', 'stative', 'emotion', 'motion', 'creation', 'possession', 'communication', 'competition', 'perception', 'cognition', 'change', 'social', 'consumption', 'body']

    psst_list=[]
    for k in hierarchical_prep_dict.keys():
        psst_list.append(k[0:8])   # abbreviating each preposition here and below so that the table fits onto one page

    # print(psst_list)
       
    for psst in psst_list:
        for dic in all_dicts:
            dic[psst[0:8]]=0
            
    
    for k, v in vsst_psst_collocation_dict.items():
        _vsst=k[0]
        index_vsst=all_dicts_names.index(_vsst)
        vsst_dict=all_dicts[index_vsst]
        for psst in k[1:]:
            vsst_dict[psst[0:8]]+=v
            
    #now including the prepositions at the beginning of sentence before first vsst; in these tuples, the vsst is the last element        
    for k, v in hierarchical_initial_psst_with_first_vsst_dict.items():
        _vsst=k[-1]
        index_vsst=all_dicts_names.index(_vsst)
        vsst_dict=all_dicts[index_vsst]
        for psst in k[:-1]:
            vsst_dict[psst[0:8]]+=v
            
            

 
    # abbrev_psst_list=[]
    # for ps in psst_list:
    #     abbrev_psst_list.append(ps[0:4])
 
    df = pd.DataFrame(columns=psst_list, index=all_dicts_names)  
    
    # for i in all_dicts:
    #     print(i)
    
    zip_a=zip(all_dicts, all_dicts_names)    
    for item in zip_a:
        this_dict=item[0]
        name=item[1] 
        #print(this_dict)           
        df.loc[name] = pd.Series(this_dict)
        
    df.loc['Total']= df.sum()  #adding Row with sum of each column

    df['RowTotal']=df.sum(axis=1)  #adding Column with sum of each row
    

    # print(df)

    print('\n')

    print(df.to_latex())

    print('\n')

    totalTable=df['RowTotal']['Total']

    #print("Total of the table: ", totalTable)

    print(round(df/totalTable, 3).to_latex())    #probabilities of all values in table
    
    csv_path='/Users/Michael/Desktop/psst_freq.csv'
    
    # df.to_csv(csv_path, sep='\t')
    
    # print('\n')
    # print("Row sum: ")
    # print('\n')
    # print(df.sum(axis=1))
    # print('\n')
    #
    # print("Column sum: ")
    # print('\n')
    # print(df.sum(axis=0))
    # print('\n')
    
    
            
    # zip_a=zip(all_dicts, all_dicts_names)
    # for item in zip_a:
    #     this_dict=item[0]
    #     name=item[1]
    #     print('\n')
    #     print(name)
    #     print('\n')
    #     for k, v in sorted(this_dict.items()):
    #         print(k, v)
            

def what_actual_verbs_come_up_with_distributions():
    
    verb_lemma_count_dict=defaultdict(int)
    
    for k, v in streusle_dict.items():
        for i in zip(v['lemmas'], v['tags']):
            lemma=i[0]
            tag=i[1]
            if len(tag)>1:
                if tag.split("-")[1] in verb_ss_count_dict:
                    verb_lemma_count_dict[lemma]+=1
                    if lemma in ['you', 'i', 'ca', 'well', 'who', 'where']:
                        print(lemma, tag)
            
                    
    for k, v in sorted(verb_lemma_count_dict.items()):
        print(k, v)
        
    print('\n')
    print("Length", len(verb_lemma_count_dict))


  
       

    
    
  
def main():
    
      create_streusle_dict()
      
      basic_analytics()
      
      #deprecated_output_sentences_for_dependency_parsing()
      
      cleaner_output_sentences_for_dependency_features() #this worked well
      
      read_dependency_data()  #this worked well
      
      create_final_features()
      
      kmeans_dependency_features()
      
      create_word_vector_features()
      
      kmeans_wordvectors()
      
      #multiple_clustering()
      
      #ward_clustering()
      
      #ward_clustering_take_two()
      
      
      #count_verb_supersenses()
      

  
            
        
        
if __name__== "__main__":
    main()  


