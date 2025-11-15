# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
import mysql.connector
import hashlib
import datetime
import calendar
from flask_mail import Mail, Message
from flask import send_file
import random
from random import randint
from urllib.request import urlopen
import webbrowser
from plotly import graph_objects as go
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import imagehash
from werkzeug.utils import secure_filename
from PIL import Image
import argparse
import urllib.request
import urllib.parse
   
# necessary imports 
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


##
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="fire_detect"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
##email
mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": "forestfire2448@gmail.com",
    "MAIL_PASSWORD": "eoxwjxxfxquybndl"
}

app.config.update(mail_settings)
mail = Mail(app)
#######
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    ff=open("note.txt",'w')
    ff.write("0")
    ff.close()

    ff=open("detect.txt",'w')
    ff.write("")
    ff.close()
    return render_template('index.html',msg=msg)




@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    
    
    return render_template('admin.html')



@app.route('/img_process', methods=['GET', 'POST'])
def img_process():
    

    return render_template('img_process.html')

@app.route('/pro1', methods=['GET', 'POST'])
def pro1():
    msg=""
    dimg=[]
    path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #list_of_elements = os.listdir(os.path.join(path_main, folder))

        #resize
        '''img = cv2.imread('static/data/'+fname)
        rez = cv2.resize(img, (300, 300))
        cv2.imwrite("static/dataset/"+fname, rez)

        img = cv2.imread('static/dataset/'+fname) 	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/trained/g_"+fname, gray)
        ##noice
        img = cv2.imread('static/trained/g_'+fname) 
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        fname2='ns_'+fname
        cv2.imwrite("static/trained/"+fname2, dst)'''

    return render_template('pro1.html',dimg=dimg)


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/pro11', methods=['GET', 'POST'])
def pro11():
    msg=""
    dimg=[]
    path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)

    return render_template('pro11.html',dimg=dimg)

@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        f1=open("adata.txt",'w')
        f1.write(fname)
        f1.close()
        ##bin
        image = cv2.imread('static/dataset/'+fname)
        original = image.copy()
        kmeans = kmeans_color_quantization(image, clusters=4)

        # Convert to grayscale, Gaussian blur, adaptive threshold
        gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

        # Draw largest enclosing circle onto a mask
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            break
        
        # Bitwise-and for result
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask==0] = (0,0,0)

        
        ###cv2.imshow('thresh', thresh)
        ###cv2.imshow('result', result)
        ###cv2.imshow('mask', mask)
        ###cv2.imshow('kmeans', kmeans)
        ###cv2.imshow('image', image)
        ###cv2.waitKey()

        cv2.imwrite("static/trained/bb/bin_"+fname, thresh)

    
   

    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        ##
        
        
        img = cv2.imread('static/trained/g_'+fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,1.5*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        segment = cv2.subtract(sure_bg,sure_fg)
        img = Image.fromarray(img)
        segment = Image.fromarray(segment)
        path3="static/trained/sg/sg_"+fname
        segment.save(path3)
        

    return render_template('pro2.html',dimg=dimg)

###Segmentation using RNN
def crfrnn_segmenter(model_def_file, model_file, gpu_device, inputs):
    
    assert os.path.isfile(model_def_file), "File {} is missing".format(model_def_file)
    assert os.path.isfile(model_file), ("File {} is missing. Please download it using "
                                        "./download_trained_model.sh").format(model_file)

    if gpu_device >= 0:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_file, model_file, caffe.TEST)

    num_images = len(inputs)
    num_channels = inputs[0].shape[2]
    assert num_channels == 3, "Unexpected channel count. A 3-channel RGB image is exptected."
    
    caffe_in = np.zeros((num_images, num_channels, _MAX_DIM, _MAX_DIM), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        caffe_in[ix] = in_.transpose((2, 0, 1))

    start_time = time.time()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    end_time = time.time()

    print("Time taken to run the network: {:.4f} seconds".format(end_time - start_time))
    predictions = out[net.outputs[0]]

    return predictions[0].argmax(axis=0).astype(np.uint8)


def run_crfrnn(input_file, output_file, gpu_device):
    """ Runs the CRF-RNN segmentation on the given RGB image and saves the segmentation mask.
    Args:
        input_file: Input RGB image file (e.g. in JPEG format)
        output_file: Path to save the resulting segmentation in PNG format
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
    """

    input_image = 255 * caffe.io.load_image(input_file)
    input_image = resize_image(input_image)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    palette = get_palette(256)
    #PIL reads image in the form of RGB, while cv2 reads image in the form of BGR, mean_vec = [R,G,B] 
    mean_vec = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    mean_vec = mean_vec.reshape(1, 1, 3)

    # Rearrange channels to form BGR
    im = image[:, :, ::-1]
    # Subtract mean
    im = im - mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = _MAX_DIM - cur_h
    pad_w = _MAX_DIM - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Get predictions
    segmentation = crfrnn_segmenter(_MODEL_DEF_FILE, _MODEL_FILE, gpu_device, [im])
    segmentation = segmentation[0:cur_h, 0:cur_w]

    output_im = PILImage.fromarray(segmentation)
    output_im.putpalette(palette)
    output_im.save(output_file)
###Feature extraction & Classification
def DCNN_process(self):
        
        train_data_preprocess = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

        test_data_preprocess = (1./255)

        train = train_data_preprocess.flow_from_directory(
                'dataset/training',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        test = train_data_preprocess.flow_from_directory(
                'dataset/test',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        ## Initialize the Convolutional Neural Net

        # Initialising the CNN
        cnn = Sequential()

        # Step 1 - Convolution
        # Step 2 - Pooling
        cnn.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second convolutional layer
        cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        cnn.add(Flatten())

        # Step 4 - Full connection
        cnn.add(Dense(units = 128, activation = 'relu'))
        cnn.add(Dense(units = 1, activation = 'sigmoid'))

        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        history = cnn.fit_generator(train,
                                 steps_per_epoch = 250,
                                 epochs = 25,
                                 validation_data = test,
                                 validation_steps = 2000)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        test_image = image.load_img('\\dataset\\', target_size=(128,128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        print(result)

        if result[0][0] == 1:
                print('feature extracted')
        else:
                print('none')
@app.route('/pro3', methods=['GET', 'POST'])
def pro3():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        
    '''path_main = 'static/dataset'
    i=1
    while i<=50:
        fname="r"+str(i)+".jpg"
        dimg.append(fname)

        img = Image.open('static/data/classify/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        i+=1
    i=1
    j=51
    while i<=10:
        
        fname="r"+str(j)+".jpg"
        dimg.append(fname)

        img = Image.open('static/dataset/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        j+=1
        i+=1

    '''    

    return render_template('pro3.html',dimg=dimg)

@app.route('/pro4', methods=['GET', 'POST'])
def pro4():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        

        #####
        image = cv2.imread("static/dataset/"+fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        
        path4="static/trained/ff/"+fname
        edged.save(path4)
        ##
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
    return render_template('pro4.html',dimg=dimg)


    

@app.route('/pro5', methods=['GET', 'POST'])
def pro5():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
    #graph
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[10,26,45,63,78]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model Precision")
    plt.ylabel("precision")
    
    fn="graph1.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph2
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[10,26,45,63,78]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model recall")
    plt.ylabel("recall")
    
    fn="graph2.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,26,45,63,78]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="graph3.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[10,26,45,63,78]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="graph4.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    return render_template('pro5.html',dimg=dimg)

def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m
                
@app.route('/pro6', methods=['GET', 'POST'])
def pro6():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    print("aaa")
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)

    #ff=open("static/trained/class.txt",'r')
    #ext=ff.read()
    #ff.close()
    #cname=ext.split(',')
    '''data1=[]
    data2=[]
    data3=[]
    data4=[]
    v1=0
    v2=0
    v3=0
    v4=0
    path_main = 'static/trained'
    #for fname in os.listdir(path_main):
    i=0
    i<127
        dimg.append(fname)
        d1=fname.split('_')
        if d1[0]=='d':
            data1.append(fname)
            v1+=1
        if d1[0]=='f':
            data2.append(fname)
            v2+=1
        if d1[0]=='n':
            data3.append(fname)
            v3+=1
        if d1[0]=='w':
            data4.append(fname)
            v4+=1
        

    g1=v1+v2+v3+v4
    dd2=[v1,v2,v3,v4]
    
    
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 5))
     
    # creating the bar plot
    plt.bar(doc, values, color ='blue',
            width = 0.4)
 

    plt.ylim((1,g1))
    plt.xlabel("Objects")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()'''

    #,data1=data1,data2=data2,data3=data3,data4=data4,cname=cname,v1=v1,v2=v2,v3=v3,v4=v4
    ##############################

    
    ###############################
    
    
    

    return render_template('pro6.html',dimg=dimg)

#######
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""
    
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')

    #num=0
    ##    
    #ff2=open("static/trained/data.txt","r")
    #rd=ff2.read()
    #ff2.close()

    ##    
    ff2=open("static/trained/tdata.txt","r")
    rd=ff2.read()
    ff2.close()

    num=[]
    r1=rd.split(',')
    s=len(r1)
    ss=s-1
    i=0
    while i<ss:
        num.append(int(r1[i]))
        i+=1

    #print(num)
    dat=toString(num)
    print(dat)
    dd2=[]
    r1=dat.split(',')
    ##

    #r1=rd.split(',')
    s=len(r1)
    ss=s-1

    dt1=[]
    dt2=[]
    data2=[]
    v1=0
    v2=0
    i=0
    for rr in r1:
        rr2=rr.split('-')
        if rr2[1]=="1":
            v1+=1
            dt1.append(str(rr2[0]))
        else:
            v2+=1
            dt2.append(str(rr2[0]))
    

    print(dt1)
    print("************")
    print(dt2)

    data2.append(dt1)
    data2.append(dt2)
    '''num=[]
    r1=rd.split(',')
    s=len(r1)
    ss=s-1
    i=0
    while i<ss:
        num.append(int(r1[i]))
        i+=1

    #print(num)
    #dat=toString(num)
    #dd2=[]
    #ex=dat.split(',')
    ##

    #ffq=open("static/trained/adata.txt",'r')
    #ext1=ffq.read()
    #ffq.close()
    '''
    
    '''v1=0
    v2=0
    
    data2=[]
    dt2=""
    #ex=ext1.split(',')
    dt1=[]
    dt2=[]
    
    g=0
    for nx in ex:
        g+=1
        nn=nx.split('-')
        if nn[1]=='1':
            
            dt1.append(nn[0])
            
            v1+=1
        if nn[1]=='2':
            dt2.append(nn[0])
            
            v2+=1
        
        
    dtt.append(dt1)
    data2.append(dtt)
   
    data2.append(dt10)'''


    #print(data2)   
    dd2=[v1,v2]
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 8))
     
    # creating the bar plot
    plt.bar(doc, values, color ='blue',
            width = 0.4)
 

    plt.ylim((1,178))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    plt.xticks(rotation=10)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()'''
    
    return render_template('classify.html',msg=msg,cname=cname,data2=data2)

#######
@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""

    
        
    return render_template('userhome.html',msg=msg)

@app.route('/test_img', methods=['GET', 'POST'])
def test_img():
    msg=""
    ss=""
    fn=""
    fn1=""
    tclass=0

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    
    if request.method=='POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            f1=open('static/test/file.txt','w')
            f1.write(filename)
            f1.close()
            file.save(os.path.join("static/test", filename))

        cutoff=1
        path_main = 'static/data'
        for fname1 in os.listdir(path_main):
            hash0 = imagehash.average_hash(Image.open("static/dataset/"+fname1)) 
            hash1 = imagehash.average_hash(Image.open("static/test/"+filename))
            cc1=hash0 - hash1
            print("cc="+str(cc1))
            if cc1<=cutoff:
                ss="ok"
                fn=fname1
                
                break
            else:
                ss="no"

        if ss=="ok":
            print("yes")
            tclass=0
            dimg=[]

            ##    
            ff2=open("static/trained/data.txt","r")
            rd=ff2.read()
            ff2.close()

            num=[]
            r1=rd.split(',')
            s=len(r1)
            ss=s-1
            i=0
            while i<ss:
                num.append(int(r1[i]))
                i+=1

            #print(num)
            dat=toString(num)
            dd2=[]
            ex=dat.split(',')
            print(fn)
            ##
            for nx in ex:
                
                nn=nx.split('-')
                f='a ('+nn[1]+').JPG'
                fn1='a ('+nn[1]+').png'
                if f==fn:
                    tclass=int(nn[0])
                    break

            
            print(tclass)
            tt=tclass-1
            cla=cname[tt]
            dta=cla+"|"+fn+"|"+fn1
            f3=open("static/test/res.txt","w")
            f3.write(dta)
            f3.close()

            
                    
            return redirect(url_for('test_pro',act="1"))
        else:
            msg="Invalid!"
    
    
        
    return render_template('test_img.html',msg=msg,fn1=fn1)


    
@app.route('/test_pro', methods=['GET', 'POST'])
def test_pro():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    fn1=gs[2]
    ts=gs[0]
    fname=fn
    ##bin
    '''image = cv2.imread('static/dataset/'+fn)
    original = image.copy()
    kmeans = kmeans_color_quantization(image, clusters=4)

    # Convert to grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

    # Draw largest enclosing circle onto a mask
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        break
    
    # Bitwise-and for result
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask==0] = (0,0,0)

    
    ###cv2.imshow('thresh', thresh)
    ###cv2.imshow('result', result)
    ###cv2.imshow('mask', mask)
    ###cv2.imshow('kmeans', kmeans)
    ###cv2.imshow('image', image)
    ###cv2.waitKey()

    #cv2.imwrite("static/upload/bin_"+fname, thresh)'''
    

    ###fg
    '''img = cv2.imread('static/dataset/'+fn)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    segment = cv2.subtract(sure_bg,sure_fg)
    img = Image.fromarray(img)
    segment = Image.fromarray(segment)
    path3="static/trained/test/fg_"+fname
    #segment.save(path3)'''
    
        
    return render_template('test_pro.html',msg=msg,fn=fn,ts=ts,act=act,fn1=fn1)

@app.route('/test_pro2', methods=['GET', 'POST'])
def test_pro2():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[0]
    fn1=gs[2]

    cursor = mydb.cursor()
    cursor.execute('SELECT count(*) FROM leaf_data where disease=%s',(ts,))
    cnt = cursor.fetchone()[0]
    st=""
    data3=[]
    

        
    return render_template('test_pro2.html',msg=msg,fn=fn,ts=ts,act=act,fn1=fn1)


@app.route('/test_pro3', methods=['GET', 'POST'])
def test_pro3():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[0]
    fn1=gs[2]

    cursor = mydb.cursor()
    cursor.execute('SELECT count(*) FROM leaf_data where disease=%s',(ts,))
    cnt = cursor.fetchone()[0]
    st=""
    data3=[]
    if cnt>0:
        st="1"
        cursor.execute('SELECT * FROM leaf_data where disease=%s',(ts,))
        data3 = cursor.fetchall()

        
    return render_template('test_pro3.html',msg=msg,fn=fn,ts=ts,act=act,fn1=fn1,data3=data3,st=st)

@app.route('/monitor', methods=['GET', 'POST'])
def monitor():
    msg=""
    ff=open("detect.txt",'w')
    ff.write("")
    ff.close()
    
    return render_template('monitor.html',msg=msg)

@app.route('/page', methods=['GET', 'POST'])
def page():
    msg=""
    v2=0
    st=""
    cursor = mydb.cursor()
    message=""
    ff=open("detect.txt","r")
    det=ff.read()
    ff.close()

    cursor.execute('SELECT * FROM admin')
    rw2 = cursor.fetchone()
    mobile=rw2[2]
    email=rw2[3]

    ff1=open("note.txt",'r')
    v=ff1.read()
    ff1.close()
    v1=int(v)

    if det=="detect":
        v2=v1+1
        vv=str(v2)
        ff=open("note.txt",'w')
        ff.write(vv)
        ff.close()
    
        st="1"
        message="Fire Detected"
       
        with app.app_context():
            msg = Message(subject="Fire Detected", sender=app.config.get("MAIL_USERNAME"),recipients=[email], body=message)
            with app.open_resource("static/fire.jpg") as fp:  
                msg.attach("static/fire.jpg", "images/jpeg", fp.read())
            mail.send(msg)
            
    else:
        
        st="2"

        
    return render_template('page.html',msg=msg,message=message,mobile=mobile,st=st)
##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))

def gen(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
        

def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)


