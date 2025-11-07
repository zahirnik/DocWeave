import os, json, numpy as np
from typing import List, Dict, Any
class SimpleVectorStore:
    def __init__(self, root='./data/outputs', collection='default'):
        self.root=root; self.collection=collection; os.makedirs(root, exist_ok=True)
        self.vec_path=os.path.join(root,f'{collection}.npy'); self.meta_path=os.path.join(root,f'{collection}.meta.json')
        self.vectors=np.load(self.vec_path) if os.path.exists(self.vec_path) else np.empty((0,0),dtype=np.float32)
        if os.path.exists(self.meta_path): self.meta=json.load(open(self.meta_path))
        else: self.meta=[]
    def upsert(self, X:np.ndarray, metadata:List[Dict[str,Any]]):
        if self.vectors.size==0: self.vectors=X.astype(np.float32)
        else:
            if self.vectors.shape[1]!=X.shape[1]: self.vectors=X.astype(np.float32); self.meta=[]
            else: self.vectors=np.vstack([self.vectors, X.astype(np.float32)])
        self.meta.extend(metadata); np.save(self.vec_path, self.vectors); json.dump(self.meta, open(self.meta_path,'w'))
    def search(self, qvec:np.ndarray, top_k:int=4):
        if self.vectors.size==0: return []
        def l2n(A):
            n=np.linalg.norm(A,axis=1,keepdims=True); n[n==0]=1.0; return A/n
        V=l2n(self.vectors); q=l2n(qvec.reshape(1,-1)); sims=(V@q.T).ravel(); idx=np.argsort(-sims)[:top_k]
        return [{ 'score': float(sims[i]), 'metadata': self.meta[i] if i < len(self.meta) else {} } for i in idx]
