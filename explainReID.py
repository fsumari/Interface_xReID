import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
from pReID.personReID import personReIdentifier
import cv2


class ReID_Model():
    def __init__(self, batch_size):
        self.model = personReIdentifier(batch_size)
        self.input_size = (160, 60)
        
    def run_on_batch(self ,image_1, image_2):
        return self.model.predict(image_1, image_2)

def generate_masks(model, N, sf, sc, p1, progressMasks):
    cell_size = np.ceil(np.array([model.input_size[0] / sf ,\
                                  model.input_size[1] / sc ]) )
    up_size = [(sf + 1) * cell_size[0],\
               (sc + 1) * cell_size[1]]
    
    grid = np.random.rand(N, sf, sc) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *model.input_size))
    print('shape masks: ',np.shape(masks))
    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + model.input_size[0], y:y + model.input_size[1]]
        progressMasks.setValue(i) # para mostrar barra de loading
    masks = masks.reshape(-1, *model.input_size, 1)
    print('shape masks: ',np.shape(masks))
    return masks

def load_img(model, path):
    image1 = cv2.imread(path)
    image1 = cv2.resize(image1, (model.input_size[1], model.input_size[0]))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    x = np.reshape(image1, (1, model.input_size[0], model.input_size[1], 3)).astype(float)
    return image1, x

N = 2000
s = 8
p1 = 0.5

def explain_reid(model, batch_size, query, gallery, masks, progressExplain):
    preds = []
    
    count = 0
    aumen = 100./ len(gallery)
    for img in gallery:
        masked = img[2] * masks
        for i in tqdm(range(0, N, batch_size), desc='Explaining'):
            preds.append(model.run_on_batch(query , masked[i:min(i+batch_size, N)]) )
            
        preds = np.concatenate(preds)
        sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, * model.input_size)
        sal = sal / N / p1
        #sal_gallery.append(sal)
        img[3] = sal[0]
        sal = []
        preds = []
        masked = []
        count+= aumen
        progressExplain.setValue(count)

