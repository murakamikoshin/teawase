import sys; sys.path.insert(0,'/home/claude/tools')
import hissatsu as hz, numpy as np, cv2
i=int(sys.argv[1]); ch=hz.load_char(i,760); h,w=ch.shape[:2]
bg=np.full((h,w,3),(80,60,60),np.float32)/255; a=ch[:,:,3:4]; im=(bg*(1-a)+ch[:,:,2::-1]*a)
im=(im*255).astype(np.uint8)
for x in range(0,w,100): cv2.line(im,(x,0),(x,h),(0,255,255),1); cv2.putText(im,str(x),(x+2,12),0,0.4,(0,255,255),1)
for y in range(0,h,100): cv2.line(im,(0,y),(w,y),(0,255,255),1); cv2.putText(im,str(y),(2,y+12),0,0.4,(0,255,255),1)
cv2.imwrite('/tmp/meas.png',im); print(w,h)
