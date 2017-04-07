"""
Created Oct 2016
@author M Regan
"""

import csv
import json
import pandas as pd
pd.set_option('display.width', 1000)

import numpy as np
from collections import defaultdict 


"""Creates the feature vector from a dependency parse for clustering experiments using verb and preposition supersenses (Amalgram)"""

path='path/to/streusle.tags.sst'

dependency_path='path/to/streusle_dependency.txt.nlp'

ID_path='path/to/streusle_IDs.txt'

streusle_dict={}

dependency_dict={} 

no_match_dependency_dict={} 


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
     
     
def read_dependency_data():
    
    with open(dependency_path, 'r') as f:
    	reader = csv.reader(f, delimiter="\t")
    	d=list(reader)  
    
    with open(ID_path, 'r') as g:
    	reader = csv.reader(g)
    	e=list(reader)
        
    temp_list=[]
    
    temp_list_of_lists=[]
    
    for item in d:

        if item!=[]:
            temp_list.append(item)
        else:
            if len(temp_list)==1 and temp_list[0][1]=='.':  #getting rid of individual periods (e.g., double punctuation)
                pass
            else:
                temp_list_of_lists.append(temp_list)
                temp_list=[]
            
    
    zip_list=zip(e, temp_list_of_lists)
    
    for element in zip_list:
        key=element[0][0]
        value=element[1]
        dependency_dict[key]=value
        
        

    for k, v in dependency_dict.items():
        print(k, v)
    
    print()
    print("length streusle", len(streusle_dict))
    print("length dependency", len(dependency_dict))
        

    
          
def main():
    
      create_streusle_dict()
      
      #display_streusle_dict()
      
      read_dependency_data()
      

        
        
if __name__== "__main__":
    main()  
