import networkx as nx
import os, os.path
import numpy as np
import operator
import re
from prettytable import *
import matplotlib.pyplot as plt

base_dir="../lab1_data/SingleSpreaders/sir"
groups=['groupA','groupB','groupC']
scenarios=['scenario1','scenario2','scenario3']
results_task1a='../result/task1a.txt'
results_task1b='../result/task1b.txt'
results_task2='../result/task2.pdf'

## get last line of a file ##
def get_last_line(node_file):

    with open(node_file) as f:

        last = None

        for last in (line for line in f if line.rstrip('\n')):
            pass
    return last

## get directory needed ##
def get_directory_needed(base_dir, scenario, group):
    # dirs = [root for root, dirs, files in os.wals(base_dir)]
    dirs=[x[0] for x in os.walk(base_dir)]
    for directory in dirs:
        if ((scenario in directory) and (group in directory)):
            return directory

## process all files in directory and get the max timestep achieved##
def get_max_timestep_and_length_of_group(dir):

    count=0
    last_line=0

    for file in os.listdir(dir):

        ## get only .txt files ##
        if file.endswith(".txt"):

            ## get only last line of files and extract first colummn##
            temp=get_last_line(dir+'/'+file)
            temp.rstrip()
            last_line_array=temp.split(' ')
            timestep=int(last_line_array[0])
            ## retain the maximum number of timesteps achieved in the group ##
            if int(timestep)> int(last_line):

                last_line=timestep

        count+=1

    return last_line, count

## process each txt file, store the column of results needed for each node and get the average behavior of the group ##
def get_average_behavior_of_group(dir,column_needed):
    max_timestep,group_length=get_max_timestep_and_length_of_group(dir)

    m=int(group_length)
    n=max_timestep+1
    nodes_results=np.asfarray([[0 for x in range(n)] for y in range(m)])

    m=0
    for file in os.listdir(dir):

        ## get only .txt files ##
        if file.endswith(".txt"):

            with open(dir+'/'+file, "r") as ins:
                n=0
                for line in ins:
                    line.rstrip()
                    temp=line.split(' ')
                    nodes_results[m,n]=temp[int(column_needed)]
                    n+=1
        m+=1

    # average every column of the array #
    average_behavior_of_nodes=nodes_results.mean(axis=0)
    return average_behavior_of_nodes

## process each txt file, store the cumulative number of nodes being influenced at the end of the process for each node of the group ##
## (which occurs at a different timestep for every node) and get the average cumulative influence of the group ##
def get_average_cumulative_influence_of_group(dir,column_needed):

    max_timestep, group_length=get_max_timestep_and_length_of_group(dir)

    node_cumulative_influence=[]

    for file in os.listdir(dir):

        ## get only .txt files ##
        if file.endswith(".txt"):

            ## Hint: use the get_last_line function to get the last line the file
            ## and get the value needed from the correct column.

            ###################
            #                 #
            # YOUR CODE HERE  #
            #                 #
            ###################

            line = get_last_line(dir+'/'+file)
            line.rsplit()
            temp = line.split(' ')
            total_inf = temp[column_needed]

            node_cumulative_influence.append(float(total_inf))

    return ( sum(node_cumulative_influence) / float(len(node_cumulative_influence)) )

## find min length of process duration for plotting purposes ##
def reformulate_results(results,min_length):
    for key,value in results.items():

        if len(value)<min_length:

            min_length=len(value)

    for key,value in results.items():

        value=value[:min_length]
        results[key]=value

    return results

## plot table of results ##
def plot_table_of_results(results,results_length,results_file,option):

    if (option=='all'):
        headers=list(range(0,results_length))
        headers= ['groups'] + headers
    elif (option=='total'):

        headers='Total Influence'
        headers= ['groups'] + [headers]

    t=PrettyTable(headers)

    if (option=='all'):
        row=[]
        for key,value in results.items():

            key=str(key)
            row= [key] + list(value)
            t.add_row(row)
    elif (option=='total'):
        row=[]
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

## plot results ##
def plot_results(results,results_length,results_file):

    t = np.arange(0., results_length, 1.0)

    plt.plot(t, results['scenario1'], 'ro--', label='scenario1')
    plt.plot(t, results['scenario2'], 'bs--', label='scenario2')
    plt.plot(t, results['scenario3'], 'g^--', label='scenario3')
    plt.legend(loc='upper right')
    plt.xlabel('Timesteps')
    plt.ylabel('Influence')

    plt.savefig(results_file)

    plt.close()

### task 1a ###
results={}

for group in groups:


    ## Hint: use the get_directory_needed function to get the directory needed to access the txt files
    ## in every group and then use the get_average_behavior_of_group to get the array needed.
    ## Store the array as a value in the dictionary named results.

    ###################
    #                 #
    # YOUR CODE HERE  #
    #                 #
    ###################

    dir = get_directory_needed(base_dir, 'scenario1', group)
    average_behavoir_of_group = get_average_behavior_of_group(dir, 1)
    results[group] = average_behavoir_of_group

    min_length = len(average_behavoir_of_group)


results=reformulate_results(results,min_length)

plot_table_of_results(results,min_length,results_task1a,'all')

### task 1b ###
results={}

for group in groups:

    ## Hint: use the get_directory_needed function to get the directory needed to access the txt files
    ## in every group and then use the get_average_cumulative_influence_of_group to get the array needed.
    ## Store the array as a value in the dictionary named results.


    ###################
    #                 #
    # YOUR CODE HERE  #
    #                 #
    ###################
    dir = get_directory_needed(base_dir, 'scenario1', group)
    average_behavior_of_group = get_average_cumulative_influence_of_group(dir, 2)
    results[group] = average_behavior_of_group


plot_table_of_results(results,1,results_task1b,'total')


### task 2 ###

results={}
for scenario in scenarios:

    ## Hint: use the get_directory_needed function to get the directory needed to access the txt files
    ## in every scenario and then use the get_average_behavior_of_group to get the array needed.
    ## Store the array as a value in the dictionary named results.

    ###################
    #                 #
    # YOUR CODE HERE  #
    #                 #
    ###################
    dir = get_directory_needed(base_dir, scenario, 'groupC')
    average_behavior_of_group = get_average_behavior_of_group(dir, 1)
    results[scenario] = average_behavoir_of_group

    min_length=len(average_behavior_of_group)

results=reformulate_results(results,min_length)

plot_results(results,min_length,results_task2)
