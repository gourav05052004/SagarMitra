"""Portable inference-only RGB/gradient profile extraction for the pixel head."""
import cv2
import numpy as np


def image_fields(bgr):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)*.25
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)*.25
    return np.concatenate([rgb,gray[:,:,None],gx[:,:,None],gy[:,:,None]],2)


def extract_profiles(fields,boxes):
    boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    if not len(boxes):return np.empty((0,4,6,24,9),np.float16)
    output=[];along=np.linspace(.05,.95,24,dtype=np.float32)[None,:,None]
    across=np.linspace(-.06,.06,9,dtype=np.float32)[None,None,:]
    for start in range(0,len(boxes),128):
        b=boxes[start:start+128];x0,y0,x1,y1=[b[:,i,None,None] for i in range(4)]
        w=np.maximum(x1-x0,1.);h=np.maximum(y1-y0,1.)
        along_x=np.broadcast_to(x0+w*along,(len(b),24,9));along_y=np.broadcast_to(y0+h*along,(len(b),24,9))
        grids=[(np.broadcast_to(x0+w*across,along_y.shape),along_y),
            (along_x,np.broadcast_to(y0+h*across,along_x.shape)),
            (np.broadcast_to(x1+w*across,along_y.shape),along_y),
            (along_x,np.broadcast_to(y1+h*across,along_x.shape))]
        sides=[]
        for x,y in grids:
            values=cv2.remap(fields,(x-.5).reshape(-1,9).astype(np.float32),
                (y-.5).reshape(-1,9).astype(np.float32),cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE)
            sides.append(values.reshape(len(b),24,9,6).transpose(0,3,1,2))
        output.append(np.stack(sides,1).astype(np.float16))
    return np.concatenate(output)
