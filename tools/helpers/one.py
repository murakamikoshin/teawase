import sys, importlib.util, numpy as np, cv2
sys.path.insert(0,'/home/claude/tools'); sys.argv=[sys.argv[0]]+sys.argv[1:]
cid=sys.argv[1]; ts=[float(x) for x in sys.argv[2].split(',')]
sp=importlib.util.spec_from_file_location("m","/home/claude/tools/hs_%s.py"%cid); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
cls=[v for v in vars(m).values() if isinstance(v,type) and v.__module__=="m"][-1]; r=cls()
fr=[(np.clip(r.frame(t),0,1)*255).astype(np.uint8) for t in ts]
cv2.imwrite(sys.argv[3], np.concatenate(fr,1)[:,:,::-1])
