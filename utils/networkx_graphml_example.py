import networkx as nx
import os 
import numpy as np
import matplotlib.pyplot as plt

path = '/home/mcdansl1/Data/hsbm'

list_of_graphml_files = os.listdir(path) # returns list

G = nx.read_graphml(path + '/' + list_of_graphml_files[0])

'''
print(G)

for k in G.edges():
    print(k)

for k in G.nodes():
    print(k)
'''

# converts A's networkx to adj matrix to numpy numpy array
A = nx.to_numpy_matrix(G)

np.savetxt('test.out', A, delimiter=',', fmt='%i') # specity int format for write out

layout = nx.spring_layout(G)
nx.draw(G, layout)
nx.draw_networkx_labels(G,pos=layout)
plt.savefig('netx_grapml.png')