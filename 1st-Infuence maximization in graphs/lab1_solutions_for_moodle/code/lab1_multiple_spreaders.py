import networkx as nx
import os, os.path
import numpy as np
import operator
import re
from prettytable import *
import matplotlib.pyplot as plt

base_dir="../MultipleSpreaders/Greedy"
settings=['MC_100','MC_10000']
results_task1a='../MultipleSpreaders/task1a.pdf'
results_task1b='../MultipleSpreaders/task1b.txt'
undirected_network_file="../undirected_gcc/undirected_Wiki-Vote.txt"
results_task2='../MultipleSpreaders/task2.txt'

## get directory needed ##
def get_directory_needed(base_dir, setting):

    dirs=[x[0] for x in os.walk(base_dir)]
    for directory in dirs:
        if (setting in directory):
            return directory

## extract results needed from files ##
def get_results(folder,file,column_needed,type):

    results=[]
    with open(base_dir+'/' +folder+'/'+file, "r") as ins:

        for line in ins:
            line.rstrip()
            temp=line.split(' ')
            if (type=='float'):
                results.append(float(temp[column_needed]))
            elif (type=='str'):
                results.append(str(temp[column_needed]))


    return results

def get_last_line_results(dir,file,column_needed):

    with open(dir+'/'+file, "r") as f:

        last = None

        for last in (line for line in f if line.rstrip('\n')):
            last.rstrip()
            temp=last.split(' ')
            result=float(temp[column_needed])
            pass

    return result

## plot results ##
def plot_results(results,results_file,l):

    t = np.arange(0., l, 1.0)

    plt.plot(t, results['MC_100'], 'ro--', label='MC_100')
    plt.plot(t, results['MC_10000'], 'bs--', label='MC_10000')
    plt.legend(loc='lower right')
    plt.xlabel('# Nodes')
    plt.ylabel('Influence')

    plt.savefig(results_file)

    plt.close()

## plot table of results ##
def plot_table_of_results(results,results_file):

    headers='Total time'
    headers= ['settings'] + [headers]

    t=PrettyTable(headers)

    for key,value in results.items():

        key=str(key)
        row= [key] + [value]
        t.add_row(row)


    f=open(results_file,'w')
    sys.stdout = f
    print (t)
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")
    f.close()


### task 1a ###
l=30
results={}

for setting in settings:

    results[setting]=get_results(setting,'LT_Greedy.txt',1,'float')

plot_results(results,results_task1a,l)

### task 1b ###

results={}

for setting in settings:

    dir=get_directory_needed(base_dir,setting)

    results[setting]=get_last_line_results(dir,'LT_Greedy.txt',5)

plot_table_of_results(results,results_task1b)


### task 2 ###

G=nx.read_edgelist(undirected_network_file)

kcore_node_dictionary=nx.core_number(G)

node_ids=get_results(setting,'LT_Greedy.txt',0,'str')

degree_node_dictionary=G.degree()

f=open(results_task2,'w')
f.write('Max_Degree Max_Core'+'\n')

max_degree=max(degree_node_dictionary.items(), key=operator.itemgetter(1))[0]

max_core=max(kcore_node_dictionary.items(), key=operator.itemgetter(1))[0]

f.write(str(degree_node_dictionary[max_degree])+ ' ' + str(kcore_node_dictionary[max_core])+'\n')

f.write('NodeIDs Degree Core_Number'+'\n')

for node_id in node_ids:

    f.write(node_id+ ' ' + str(degree_node_dictionary[node_id]) + ' ' +str(kcore_node_dictionary[node_id]) + '\n')

f.close()
