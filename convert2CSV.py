import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

labels_celeb = ['background','skin','nose',
        'eye_g','l_eye','r_eye','l_brow',
        'r_brow','l_ear','r_ear','mouth',
        'u_lip','l_lip','hair','hat',
        'ear_r','neck_l','neck','cloth']
def read_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if type(img) is type(None):
        return np.zeros((256, 256, 1), dtype=np.uint8)
    return img

def mask2binary(path):
    mask = read_mask(path)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 0,1,0)
    return mask

def rle_encode(img): 
    pixels = img.flatten()
    if np.sum(pixels)==0:
        return '0'
    pixels = np.concatenate([[0], pixels, [0]]) 
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1 
    runs[1::2] -= runs[::2]
    # to string sep='_'
    runs = '_'.join(str(x) for x in runs)
    return runs

def rle_decode(mask_rle, shape): 
    s = mask_rle.split('_')
    s = [0 if x=='' else int(x) for x in s]
    if np.sum(s)==0:
        return np.zeros(shape, dtype=np.uint8)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])] 
    starts -= 1 
    ends = starts + lengths 
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8) 
    for lo, hi in zip(starts, ends): 
         img[lo:hi] = 255
    return img.reshape(shape)

def mask2csv(mask_paths, csv_path='mask.csv',image_id=1,header=False):
    """
        mask_paths: dict of label:mask_paths
        ['label1':path1,'label2':path2,...]
    """
    results = []
    for i, label in enumerate(labels_celeb):
        try:
            mask = mask2binary(mask_paths[label])
        except:
            mask = np.zeros((256, 256), dtype=np.uint8)
        mask = rle_encode(mask)
        results.append(mask)
    df = pd.DataFrame(results)
    df.insert(0,'label',labels_celeb)
    df.insert(0,'Usage',["Public" for i in range(len(results))])
    df.insert(0,'ID',[image_id*19+i for i in range(19)])
    if header:
        df.columns = ['ID','Usage','label','segmentation']
    # print()
    # print(df)
    df.to_csv(csv_path,mode='a',header=header,index=False)

def mask2csv2(masks, csv_path='mask.csv',image_id=1,header=False):
    """
        mask_paths: dict of label:mask
        ['label1':mask1,'label2':mask2,...]
    """
    results = []
    for i, label in enumerate(labels_celeb):
        try:
            mask = masks[label]
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        except:
            mask = np.zeros((256, 256), dtype=np.uint8)
        mask = rle_encode(mask)
        results.append(mask)
    df = pd.DataFrame(results)
    df.insert(0,'label',labels_celeb)
    df.insert(0,'Usage',["Public" for i in range(len(results))])
    df.insert(0,'ID',[image_id*19+i for i in range(19)])
    
    if header:
        df.columns = ['ID','Usage','label','segmentation']
    #print()
    #print(df)
    df.to_csv(csv_path,mode='a',header=header,index=False)

def convert_2_masks(pred_label):

    masks = dict()
    ones_array = np.ones_like(pred_label)
    zeros_array = np.zeros_like(pred_label)
    for i, label in enumerate(labels_celeb):
        if label == 'background':
            masks[label] = zeros_array
        else:
            masks[label] = np.where(pred_label == i, ones_array, zeros_array)
    return masks