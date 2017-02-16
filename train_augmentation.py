from name_to_num import name_to_num_dict, filename_to_pokemon_dict
import numpy as np
from scipy.misc import imread, imsave, imresize
from scipy import ndimage
import os

IMGSIZE = 224
text_file=open('dataset/augmented_train_labels.txt','w')

dir_train = "dataset/train"
dir_aug_train = "dataset/augmented_train/"
cntr = 1
for file_name in os.listdir(dir_train):

    print "filename=",file_name
    #extension = file_name.split('.')[1]
    extension = "jpg"

    img = imread(dir_train + "/" + file_name)
    pokeid = int(file_name.split('.')[0])


    if len(img.shape) == 0:
        continue

    img = imresize(img, (IMGSIZE, IMGSIZE))


    if len(img.shape) == 2:  #grayscale
        img_rec = np.zeros((img.shape[0],img.shape[1],3))
        img_rec[:,:,0]+=img
        img_rec[:,:,1]+=img
        img_rec[:,:,2]+=img
        img0 = img_rec

    else:
        img0 = img


    img1 = np.fliplr(img0)  #rotate image

    img2 = ndimage.rotate(img0, np.random.randint(-10,10)) #randomly rotate image
    img2 = imresize(img2, (IMGSIZE, IMGSIZE))
    
    img3 = ndimage.rotate(img1, np.random.randint(-10,10)) #randomly rotate mirror image
    img3 = imresize(img3, (IMGSIZE, IMGSIZE))

    random_shift = np.random.randint(1,15)
    img4 = np.zeros_like(img0)
    img4[:224-random_shift,:224-random_shift,:] += img0[random_shift:,random_shift:,:]  # translate image

    random_shift = np.random.randint(1,15)
    img5 = np.zeros_like(img0)
    img5[:224-random_shift,:224-random_shift,:] += img1[random_shift:,random_shift:,:]  # translate mirror image
    
    random_shift = np.random.randint(1,15)
    img6 = np.zeros_like(img0)
    img6[:224-random_shift,:224-random_shift,:] += img2[random_shift:,random_shift:,:]  # translate rotated image


    imsave(dir_aug_train + str(cntr+0)+ "." + extension, img0)
    imsave(dir_aug_train + str(cntr+1)+ "." + extension, img1)
    imsave(dir_aug_train + str(cntr+2)+ "." + extension, img2)
    imsave(dir_aug_train + str(cntr+3)+ "." + extension, img3)
    imsave(dir_aug_train + str(cntr+4)+ "." + extension, img4)
    imsave(dir_aug_train + str(cntr+5)+ "." + extension, img5)
    imsave(dir_aug_train + str(cntr+6)+ "." + extension, img6)

    for i in range(7):
        text_str = dir_aug_train + str(cntr) + "." + extension + " " + str(name_to_num_dict[filename_to_pokemon_dict[pokeid]])
        text_file.write(text_str)
        text_file.write("\n")
        cntr += 1

    

    



