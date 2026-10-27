"""Verified sprite references, independent of residual frame motion."""
from pathlib import Path

import cv2
import numpy as np

from .enemy_vision import RexDetector
from .types import iou


class LargeProjectileDetector:
    def __init__(self):
        root=Path(__file__).parent/'profiles/mario/enemy-seeds'
        self.reference=cv2.imread(str(root/'large-projectile.png'))
        self.mask=cv2.imread(str(root/'large-projectile.mask.png'),0)
        if self.reference is None or self.mask is None:raise ValueError('Missing projectile reference')
        self.templates=[(self.reference,self.mask),(cv2.flip(self.reference,1),cv2.flip(self.mask,1))]

    def detect(self,frame,player_box=None):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        dark=(gray<65).astype(np.uint8);dark[:35]=0
        count,_,stats,_=cv2.connectedComponentsWithStats(dark,8)
        hits=[]
        for x,y,w,h,area in stats[1:count]:
            # Dark outlines can connect to Mario or the floor at contact.
            # Component extent is only a proposal; the masked sprite match
            # below must still verify the complete projectile on current pixels.
            if not (800<=area<=6500 and 42<=w<=frame.shape[1] and 42<=h<=180):continue
            left=max(0,int(x)-5);top=max(35,int(y)-5)
            right=min(frame.shape[1],int(x+w)+5);bottom=min(frame.shape[0],int(y+h)+5)
            for reference,mask in self.templates:
                rh,rw=reference.shape[:2]
                if bottom-top<rh or right-left<rw:continue
                scores=cv2.matchTemplate(gray[top:bottom,left:right],cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),
                                         cv2.TM_SQDIFF_NORMED,mask=mask)
                scores=np.nan_to_num(scores,nan=1.,posinf=1.,neginf=1.)
                error,_,location,_=cv2.minMaxLoc(scores)
                if error>.18:continue
                bx,by=left+location[0],top+location[1];crop=frame[by:by+rh,bx:bx+rw]
                visible=mask>0
                if player_box is not None:
                    px,py,pw,ph=map(int,player_box)
                    l,t=max(bx,px),max(by,py);r,b=min(bx+rw,px+pw),min(by+rh,py+ph)
                    if r>l and b>t:visible[t-by:b-by,l-bx:r-bx]=False
                if np.count_nonzero(visible)<mask.size*.55:continue
                color_error=float(np.mean(np.abs(crop[visible].astype(float)-reference[visible].astype(float))))/255
                hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                red=((hsv[...,0]<10)|(hsv[...,0]>170))&(hsv[...,1]>110)&visible
                white=(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)>180)&visible
                if color_error>.18 or np.count_nonzero(red)<15 or np.count_nonzero(white)<50:continue
                box=(bx,by,rw,rh)
                if any(iou(box,item['bbox'])>.4 for item in hits):continue
                hits.append({'bbox':box,'label':'enemy_projectile_large',
                             'confidence':round(min(.99,1-float(error)),4)})
        return hits


class MarioObjectDetector:
    def __init__(self):
        self.rex=RexDetector();self.projectile=LargeProjectileDetector()

    def detect(self,frame,player_box=None):
        return self.rex.detect(frame,player_box)+self.projectile.detect(frame,player_box)
