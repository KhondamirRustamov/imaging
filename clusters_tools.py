import glob
from PIL import Image
import numpy as np


def download_images(datadir, RGB = 2, contrast = 1):
    im_list = []
    RGB -= 1
    for i in glob.glob(f'{datadir}/*'):
        if os.path.isfile(i):
            if contrast == 0:
                image_pil = np.asarray(Image.open(i))[:,:,RGB]*1
            else:
                image_pil = np.asarray(Image.open(i))[:,:,RGB]*contrast
            im_list.append(image_pil)
    if contrast == 0:
        max = max([np.max(x) for x in im_list])
        min = min([np.min(x) for x in im_list])
        for i, x in enumerate(im_list):
            im_list[i] = (im_list[i] - min)/(max-min)
    return im_list


def calculate_changes_downsample(start, end, im_list, thre=0.8):
    changes = []
    start = start - 1
    end = end - 1
    for i in range(start, end):
        c = 1/(1+np.exp(-(np.sqrt((im_list[i]-im_list[i+1])**2))))
        c[c<=np.mean(c)]=0
        kernel = 5
        ct = np.array([[np.mean(c[x*kernel:(x+1)*kernel,y*kernel:(y+1)*kernel]) for y in range(int(int(c.shape[1])/kernel))] for x in range(int(int(c.shape[0])/kernel))])
        #kernel2 = 5
        #ctt = np.array([[np.max(ct[x*kernel2:(x+1)*kernel2,y*kernel2:(y+1)*kernel2]) for y in range(int(int(ct.shape[1])/kernel2))] for x in range(int(int(ct.shape[0])/kernel2))])
        ct[ct<=thre]=0
        changes.append(ct)
        
    sums = np.zeros(changes[0].shape)
    for i in changes:
        sums+=i

    sums = (sums-np.min(sums))/np.max(sums)
    return sums


def calculate_changes_upsample(cc, im_list, sums):
    cc = np.zeros(im_list[0].shape)
    tt = np.argwhere(sums!=0)
    kernel2 = 5
    for x in tt:
        cc[x[0]*kernel2:(x[0]+1)*kernel2, x[1]*kernel2:(x[1]+1)*kernel2]=sums[x[0],x[1]]


def calculate_complexes(sums, size=25):
    sums_ones = sums
    no_zeros = np.argwhere(sums_ones!=0)
    no_zeros = [[x[0],x[1]] for x in no_zeros]
    
    def complex_grow(sums_ones, non_zeros_common, growing_array, working_array):
        for x in growing_array:
            for i in np.argwhere(sums_ones[x[0]-1:x[0]+2,x[1]-1:x[1]+2]!=0):
                if [i[0]-1+x[0], i[1]-1+x[1]] not in working_array:
                    working_array.append([i[0]-1+x[0], i[1]-1+x[1]])
                    growing_array.append([i[0]-1+x[0], i[1]-1+x[1]])
            if x in growing_array:
                growing_array.remove(x)
            if x in non_zeros_common:
                non_zeros_common.remove(x)
        while len(growing_array)>0:
            complex_grow(sums_ones, non_zeros_common, growing_array, working_array)
    
    complexes = []
    for x in no_zeros:
        growing_array = []
        working_array = []
        for i in np.argwhere(sums_ones[x[0]-1:x[0]+2,x[1]-1:x[1]+2]!=0):
            if [i[0]-1+x[0], i[1]-1+x[1]] not in working_array:
                working_array.append([i[0]-1+x[0], i[1]-1+x[1]])
                growing_array.append([i[0]-1+x[0], i[1]-1+x[1]])
            if x in no_zeros:
                no_zeros.remove(x)
        complex_grow(sums_ones, no_zeros, growing_array, working_array)
        complexes.append(working_array)
        
    big_complexes = [x for x in complexes if len(x)>=size]    
    
    return big_complexes




def complex_to_img(sums_ones, big_complexes):
    new_array = np.zeros(sums_ones.shape)
    for i, z in enumerate(big_complexes):
        for x in z:
            new_array[x[0],x[1]]=i+1
    return new_array


def complexes_extended(new_array):
    cc_complexes_extended = np.zeros((new_array.shape[0]*5,new_array.shape[1]*5))
    tt = np.argwhere(new_array!=0)
    kernel2 = 5
    for x in tt:
        cc_complexes_extended[x[0]*kernel2:(x[0]+1)*kernel2, x[1]*kernel2:(x[1]+1)*kernel2]=new_array[x[0],x[1]]
    return cc_complexes_extended


def all_mean_slices(slice_list, coords_list):
    line = []
    for i in slice_list:
        mean = np.mean(np.array([i[x[0],x[1]] for x in coords_list]))
        line.append(mean)
    return line


def calculate_lines(cc_complexes_extended, im_list):
    number_of_complexes=np.max(cc_complexes_extended)
    lines = []
    for x in range(int(number_of_complexes)):
        lines.append(all_mean_slices(im_list, np.argwhere(cc_complexes_extended==x+1)))
    return lines
