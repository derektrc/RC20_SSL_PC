import pickle
import os
import random
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean,cdist
from collections import OrderedDict
import sys
from scipy import sparse
from heapq import nsmallest 

def ball_vec(tree,nodes,pts):
    ball = np.zeros((len(tree.keys()),pts.shape[1]))
    for i in range(len(tree.keys())):
        key = nodes[i]
        
        items = tree[key]
        centre = np.empty([0,pts.shape[1]])
        for j in range(1,len(tree[key])):
            if type(items[j])==str:
                idx = nodes.index(items[j])
                centre = np.concatenate((centre, np.array([ball[int(idx),:]])),axis=0)
            elif items[j]>=0 and items[j]<pts.shape[0]:
                centre = np.concatenate((centre, np.array([pts[int(items[j]),:]])), axis=0)
        if centre.shape[0]>0:
            ball[i,:] = np.mean(centre,axis=0)
    return ball

def find_quad(P,C):
    quadrants = np.zeros((P.shape[0],),dtype=int)
    for i in range(P.shape[0]):
        x = C[i,0]
        y = C[i,1]
        z = C[i,2]
        cx = P[i,0]
        cy = P[i,1]
        cz = P[i,2]

        if (x >= cx and y >= cy and z >= cz ) or (x < cx and y >= cy and z >= cz) :
            quadrants[i] = 1
        elif (x < cx and y < cy and z >= cz) or (x >= cx and y < cy and z >= cz):
            quadrants[i] = 2
        elif (x >= cx and y >= cy and z < cz) or (x < cx and y >= cy and z < cz):
            quadrants[i] = 3
        elif (x < cx and y < cy and z < cz) or (x >= cx and y < cy and z < cz):
            quadrants[i] = 4

    return quadrants

def rev_dict(tree,nodes):
    rev_tree={}
    for i in range(len(tree.keys())):
        rev_tree[nodes[i]]=[]
    for i in range(len(tree.keys())-1,0,-1):
        ball = nodes[i]
        children = tree[ball]
        for j in range(1, len(children)):
            if children[j] in tree.keys():
                rev_tree[children[j]].append(ball)

    rev_tree = OrderedDict(sorted(rev_tree.items(), key=lambda t: t[0]))
    return rev_tree
def get_pairs(rev_tree,tree,nodes,pts,path1):
    diff_pair_quad = np.empty([0,2],dtype=int)
    same_pair_dist = np.empty([0,2],dtype=int)
    same_pair_label = np.empty([0,],dtype=int)
    for i in range(len(tree.keys())):
        ball = nodes[i]
        children = tree[ball]
        if len(children)>1:
            for j in range(1, len(children)):
                if children[j] in tree.keys() and abs(children[0]-tree[children[j]][0])==1:
                    p = nodes.index(ball)#i
                    c = nodes.index(children[j])
                    diff_pair_quad = np.concatenate((diff_pair_quad,np.array([[p,c]])),axis=0)
    for i in range(len(tree.keys())):
        ball = nodes[i]
        children = tree[ball]
        count=i+1
        while count<len(tree.keys()) and children[0]-tree[nodes[count]][0]==0:
            p = nodes.index(ball)#i
            c = nodes.index(nodes[count])#count
            same_pair_dist = np.concatenate((same_pair_dist,np.array([[p,c]])),axis=0)
            if bool(set(rev_tree[ball]) & set(rev_tree[nodes[count]])):
                same_pair_label = np.concatenate((same_pair_label,np.array([1])),axis=None)
            else:
                same_pair_label = np.concatenate((same_pair_label,np.array([0])),axis=None)
            count=count+1
    balls = ball_vec(tree,nodes,pts)
    distance_same_pair = np.sqrt(((balls[same_pair_dist[:,0],:]-balls[same_pair_dist[:,1],:])**2).sum(axis=1))
    diff_pair_label = find_quad(balls[diff_pair_quad[:,0],:],balls[diff_pair_quad[:,1],:])
    if not os.path.exists(path1):
                os.mkdir(path1)
    with open(path1+'/ball_pair_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
       pickle.dump([diff_pair_quad, diff_pair_label, same_pair_dist, distance_same_pair, same_pair_label, len(nodes)], f)
  
def get_ball_indexes(tree,nodes,adj):
    tree_ids=tree.copy()

    for i in range(len(tree.keys())):
        key = nodes[i]
        items = tree[key]
        ids=[items[0]]
        if len(items)>1:
            for j in range(1, len(items)):
                if type(items[j])==str:
                    ids.extend(tree_ids[items[j]][1:])
                else:
                    ids.append(items[j])
            tree_ids[nodes[i]] = ids
    print(tree)
    print(tree_ids)
    exit()
    
def get_ball_indexes(tree,nodes):
    tree_ids=tree.copy()

    for i in range(len(tree.keys())):
        key = nodes[i]
        items = tree[key]
        ids=[items[0]]
        if len(items)>1:
            for j in range(1, len(items)):
                if type(items[j])==str:
                    ids.extend(tree_ids[items[j]][1:])
                else:
                    ids.append(items[j])
            tree_ids[nodes[i]] = ids
    return tree_ids

def make_dict(pts,balls,path,path1):
    levels = np.array([int(ball[:-4]) for ball in balls])
    levels = np.sort(levels)
    parents=[]
    children=[]
    adjacency=[]
    level=[]
    count=0
    for i in range(levels[0],levels[-1]+1):
        f = open(path+str(i)+'.pkl', 'rb')
        [parent,child] = pickle.load(f)
        f.close()
        p = np.unique(parent,axis=0)
        q = np.unique(child,axis=0)
        adj = np.zeros((p.shape[0],q.shape[0]))
        for j in range(parent.shape[0]):
            p1 = parent[j]
            q1 = child[j]
            u = np.where(np.all(p==p1,axis=1))[0]
            v = np.where(np.all(q==q1,axis=1))[0]
            adj[u,v]=1
        parents.append(p)
        children.append(q)
        adjacency.append(adj)
        level.append(chr(65+count))
        count=count+1
    for i in range(len(parents)):
        p = np.array(parents[i])
        q = np.array(children[i])
        adj = np.array(adjacency[i])
        for j in range(p.shape[0]):
            pt= p[j]
            if np.sum(adj[j,:])>0:
                idxs = np.nonzero(adj[j,:])[0]
                centre = np.mean(q[idxs,:],axis=0)
                if pt in np.array(parents[i]):
                    idx = np.where(np.all(np.array(parents[i])==pt,axis=1))[0]
                    for l in range(len(idx)):
                        parents[i][idx[l],:]=centre
                for k in range(i+1,len(children)):
                    if pt in np.array(children[k]):
                        idx = np.where(np.all(np.array(children[k])==pt,axis=1))[0]
                        for l in range(len(idx)):
                            children[k][idx[l],:]=centre
                    if pt in np.array(parents[k]):
                        idx = np.where(np.all(np.array(parents[k])==pt,axis=1))[0]
                        for l in range(len(idx)):
                            parents[k][idx[l],:]=centre
    tree = {}
    nodes=[]
    pt_nodes=[]
    for i in range(len(parents)):
        p = np.array(parents[i])
        q = np.array(children[i])
        adj = np.array(adjacency[i])
        for j in range(p.shape[0]):
            pt= p[j]
            if np.sum(adj[j,:])>0:
                if level[i]+str(j) not in tree.keys():
                    tree[level[i]+str(j)] = [i]
                    nodes.append(level[i]+str(j))
                idxs = np.nonzero(adj[j,:])[0]
                for k in idxs:
                    if np.sum(np.all(pts==q[k],axis=1))>0:
                        u = np.where(np.all(pts==q[k],axis=1))[0][0]
                        pt_nodes.append(u)
                        tree[level[i]+str(j)].append(u)
                    elif i>0 and np.sum(np.all(np.array(parents[i-1])==q[k],axis=1))>0:
                        u = np.where(np.all(np.array(parents[i-1])==q[k],axis=1))[0][0]
                        if level[i-1]+str(u) in tree.keys() and len(tree[level[i-1]+str(u)])>1:
                            tree[level[i]+str(j)].append(level[i-1]+str(u))
    cp_tree=tree.copy()
    modi_key=[]
    modi_val=[]
    for i in range(len(cp_tree.keys())):
        if len(cp_tree[nodes[i]])<3:
            count=0
            for j in range(len(cp_tree[nodes[i]])):
                if type(cp_tree[nodes[i]][j])!=str:
                    count=count+1
            if count>1 and len(cp_tree[nodes[i]]) == count:
                modi_key.append(nodes[i])
                modi_val.append(cp_tree[nodes[i]][1])
                del tree[nodes[i]]
            if count==1 and len(cp_tree[nodes[i]]) == count:
                del tree[nodes[i]]
    tree = OrderedDict(sorted(tree.items(), key=lambda t: t[0]))
    new_nodes=[]
    for node in nodes:
        if node in tree.keys():
            new_nodes.append(node)

    nodes=new_nodes
    for i in range(len(tree.keys())):
        for j in range(1,len(tree[nodes[i]])):
            if tree[nodes[i]][j] in modi_key:
                idx = modi_key.index(tree[nodes[i]][j])
                tree[nodes[i]][j]=modi_val[idx]
    new_pt_nodes = []
    for k,v in tree.items():
        if len(v)==1:
            print('true')
        new_pt_nodes.extend(v[1:])
    new_pt_nodes = list(set(new_pt_nodes))
    new_ptnodes=[]
    for node in pt_nodes:
        if node in new_pt_nodes:
            new_ptnodes.append(node)
    
    pt_nodes = new_ptnodes
    nodes.extend(list(set(pt_nodes)))
    tree_ids = get_ball_indexes(tree,nodes)
    if not os.path.exists(path1):
                os.mkdir(path1)
    with open(path1+'/dict_adj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
       pickle.dump([tree, nodes, tree_ids], f)

def filter_ball(dataset,dic,name,pc,tree,nodes,tree_ids,diff_pair_quad,diff_pair_label,same_pair_dist,distance_same_pair,same_pair_label,dir_path):
    diff=2
    queue=[]
    new_nodes=[]
    top_level = int(tree[nodes[len(tree.keys())-1]][0])
    idx_key=np.empty([0,],dtype=int)
    for i in range(len(tree.keys())-1,-1,-1):
        key = nodes[i]
        level = int(tree[key][0])
        if top_level-level<=diff:
            new_nodes.append(key)
            idx_key = np.concatenate((idx_key,i),axis=None)
        elif top_level-level==diff+1 and len(new_nodes)<20:
            diff=diff+1
            new_nodes.append(key)
            idx_key = np.concatenate((idx_key,i),axis=None)
    idx_key = np.sort(idx_key)
    new_nodes = new_nodes[::-1]
    new_tree = tree.copy()
    new_tree_ids = tree_ids.copy()
    for i in range(0,len(tree.keys())):
        key = nodes[i]
        if key not in new_nodes:
            del new_tree[key]
            del new_tree_ids[key]
    diff_level_ids = np.empty([0,2],dtype=int)
    diff_level_label=np.empty([0,],dtype=int)
    same_level_label=np.empty([0,],dtype=int)
    same_level_dist=np.empty([0,])
    same_level_ids = np.empty([0,2],dtype=int)
    for i in range(diff_pair_quad.shape[0]):
        if diff_pair_quad[i,0] in idx_key and diff_pair_quad[i,1] in idx_key:
            id1 = diff_pair_quad[i,0]
            id2 = diff_pair_quad[i,1]
            key1 = nodes[id1]
            key2 = nodes[id2]
            idx1=new_nodes.index(key1)
            idx2=new_nodes.index(key2)
            diff_level_ids = np.concatenate((diff_level_ids,np.array([[idx1,idx2]])),axis=0)
            diff_level_label = np.concatenate((diff_level_label,diff_pair_label[i]),axis=None)
    for i in range(same_pair_dist.shape[0]):
        if same_pair_dist[i,0] in idx_key and same_pair_dist[i,1] in idx_key:
            id1 = same_pair_dist[i,0]
            id2 = same_pair_dist[i,1]
            key1 = nodes[id1]
            key2 = nodes[id2]
            idx1=new_nodes.index(key1)
            idx2=new_nodes.index(key2)
            same_level_ids = np.concatenate((same_level_ids,np.array([[idx1,idx2]])),axis=0)
            same_level_label = np.concatenate((same_level_label,same_pair_label[i]),axis=None)
            same_level_dist = np.concatenate((same_level_dist,distance_same_pair[i]),axis=None)
    ball_count = len(new_nodes)
    if not os.path.exists(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]):
        os.mkdir(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4])
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/tree.npy", new_tree)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/nodes.npy", new_nodes)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/tree_ids.npy", new_tree_ids)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/diff_pair_quad.npy", diff_level_ids)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/diff_pair_label.npy", diff_level_label)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/same_pair_dist.npy", same_level_ids)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/distance_same_pair.npy", same_level_dist)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/same_pair_label.npy", same_level_label)
    np.save(dir_path+dataset+"/"+name+"/"+dic+"/"+pc[:-4]+"/ball_count.npy", ball_count)

def knn(k, pt, pts, dist_f):
    '''Return the k-nearest points in pts to pt using a naive
    algorithm. dist_f is a function that computes the distance between
    2 points'''
    return nsmallest(k, pts, lambda x: dist_f(x, pt))

def nn(pt, pts, dist_f):
    '''Like knn but return the nearest point in pts'''
    return knn(1, pt, pts, dist_f)[0]