from networkx import read_graphml
import os 

path = '/home/mcdansl1/Data/hsbm'

list_of_graphml_files = os.listdir(path) # returns list

aGraph = read_graphml(path + '/' + list_of_graphml_files[0])

print(aGraph)

for k in aGraph.edges():
    print(k)

for k in aGraph.nodes():
    print(k)