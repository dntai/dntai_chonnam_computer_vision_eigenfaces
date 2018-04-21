import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob

class YaleFaceDb(object):
    def __init__(self, image_width = 100, image_height = 100, image_dir = 'datasets/yalefaces/centered'):
        self.image_dir = image_dir
        self.image_width = image_width
        self.image_height = image_height
        self.type = type
        
        self.image_list_person  = { }
        self.image_list_subject = { }
        self.image_list_person_subject  = { }
        
        self.image_label = None
        self.image_list = None

        self.load()
    # __init__
    
    def load(self):
        self.image_list_person.clear()
        self.image_list_subject.clear()
        self.image_list_person_subject.clear()
        
        image_real_dir = os.path.realpath(self.image_dir)
        image_names = glob(os.path.join(self.image_dir,'*.*'))

        image_list = []
        image_label = []
        for image_path in image_names:
            (_, image_name) = os.path.split(image_path)
            names  = image_name.split('.') # perrson.subject(.pgm)
            person  = names[0]
            subject = names[1]

            image = Image.open(image_path)
            image = image.resize((self.image_width,self.image_height),Image.ANTIALIAS)
            image = np.expand_dims(np.asarray(image), axis=3)

            image_label.append([person, subject, image_name])
            image_list.append(image)

            if self.image_list_subject.get(subject)==None:
                self.image_list_subject[subject] = []
            self.image_list_subject[subject].append(image)

            if self.image_list_person.get(person)==None:
                self.image_list_person[person] = []
            self.image_list_person[person].append(image)

            self.image_list_person_subject[image_name] = image
        # for
        self.image_list = np.array(image_list)
        self.image_label = np.array(image_label)
    # load

    def get_list(self):
        return self.image_list
    
    def get_label(self):
        return self.image_label

    def get_dataset(self): # (x, y) with y containing [person, subject, image_name]
        return  (self.image_list, self.image_label)
    
    def get_person(self,person):
        return self.image_list_person.get(person)
    
    def get_subject(self,subject):
        return self.image_list_subject.get(subject)
    
    def get_person_subject(self, person, subject):
        return self.image_list_person_subject.get(person + '.' + subject)
    
    def get_category_person(self):
        return sorted(list(self.image_list_person.keys()))
    
    def get_category_subject(self):
        return sorted(list(self.image_list_subject.keys()))
    

    def get_random_train_test(self, percent = 0.8): # (x_train, y_train,x_test,y_test)
        total  = len(self.image_list)
        mask = np.random.random_sample(total)<=percent
        return  ((np.array(self.image_list)[mask[:]], np.array(self.image_label)[mask[:]]), \
                (np.array(self.image_list)[mask[:]==False], np.array(self.image_label)[mask[:]==False]))

    def plot_image(self, cnt):
        plot_image(self.image_list[cnt, :, :, :], self.image_label[cnt, :])
    # plot_image

    def plot_images(self, tfrom = 0, size = [4,4]):
        plot_images(self.image_list, self.image_label, tfrom, size=[4,4])
    # plot_image

def test_db():
    db = YaleFaceDb()
    images = db.get_list()
    labels = db.get_label()

    plot_images(images, labels)
    plot_image(images[10, :, :, :], labels[10, :])
# test_db

def plot_images(images, labels, start = 0, size = [4,4], wspace=1.5, hspace=1.5):
    r, c = size
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if images.shape[3] == 1:
                axs[i,j].imshow(images[start+cnt, :,:, 0], cmap='gray')
            else:
                axs[i,j].imshow(images[start+cnt, :,:, :])
            axs[i,j].axis('off')
            axs[i,j].set_title('%s'%(labels[cnt, 0]))
            cnt += 1
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()
# sample_images

def plot_image(image, label):
    plt.imshow(image[:,:,0], cmap = 'gray')
    plt.title('%s - %s'%(label[0], label[1]))
    plt.axis('off')
    plt.show()
# sample_images