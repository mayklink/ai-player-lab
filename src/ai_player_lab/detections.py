"""Display motion estimates on a COPY, never on the model's input pixels."""
import cv2
from .perception import VisualPipeline
from .types import separate_actor


class DetectionOverlay:
    def __init__(self):self.pipeline=None;self.scene=None

    def update(self,frame,now:float,scene:str,player_box=None,*,observation=None,reuse=False,player_mask=None):
        annotated=frame.copy();detections=[]
        if scene not in {'gameplay','hub','unknown'}:
            self.pipeline=None;self.scene=scene
            return annotated,detections
        if not reuse and (self.pipeline is None or self.scene!=scene):
            self.pipeline=VisualPipeline(frame,None,hud_px=34)
        self.scene=scene
        obs=observation if reuse else self.pipeline.process(frame,now,player_bbox=tuple(player_box) if player_box else None)
        if player_box:
            detections.append({'label':'Mario?' if scene=='unknown' else 'Mario','kind':'player','bbox':list(player_box),'color':[80,230,100]})
        for obj in sorted((obs.objects if obs is not None else []),key=lambda item:not item.label.startswith('enemy_')):
            x,y,w,h=obj.bbox
            if obj.label in {'unknown_motion','background_consistent','predicted_motion'}:continue
            if (obj.age<2 or y<34 or not 3<=w<=128 or not 3<=h<=128 or w*h>12000
                    or not separate_actor(obj,tuple(player_box) if player_box else None)):continue
            predicted = obj.missed > 0 or obj.label.endswith('_predicted')
            enemy=obj.label.startswith('enemy_')
            name='Projetil' if obj.label.startswith('enemy_projectile_large') else 'Rex'
            label = (f'{name} {obj.id}' if enemy else f'Movimento {obj.id}') + (' (previsto)' if predicted else '')
            detections.append({'label':label,'kind':'enemy' if enemy else 'motion','bbox':list(obj.bbox),
                               'color':[255,190,60] if predicted else ([80,230,100] if enemy else [50,210,255]),
                               'predicted':predicted,'missed':obj.missed,
                               'confidence':obj.confidence})
            if len(detections)>=12:break
        for obj in detections:
            x,y,w,h=map(int,obj['bbox']);color=tuple(obj['color'])
            if obj.get('predicted'):
                for x0 in range(x, x+w, 6):
                    cv2.line(annotated,(x0,y),(min(x0+3,x+w),y),color,1)
                    cv2.line(annotated,(x0,y+h),(min(x0+3,x+w),y+h),color,1)
                for y0 in range(y, y+h, 6):
                    cv2.line(annotated,(x,y0),(x,min(y0+3,y+h)),color,1)
                    cv2.line(annotated,(x+w,y0),(x+w,min(y0+3,y+h)),color,1)
            else:
                cv2.rectangle(annotated,(x,y),(x+w,y+h),color,1)
            cv2.putText(annotated,obj['label'],(max(0,x),max(42,y-3)),cv2.FONT_HERSHEY_SIMPLEX,.25,color,1,cv2.LINE_AA)
        if player_box and player_mask is not None:
            x,y,w,h=map(int,player_box)
            if player_mask.shape==(h,w):
                contours,_=cv2.findContours(player_mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated,contours,-1,(80,230,100),1,offset=(x,y))
        return annotated,detections
