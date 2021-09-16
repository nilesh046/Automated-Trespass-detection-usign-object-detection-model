import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import mysql.connector
import datetime
import tkinter.font as font
import time 

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfile 
from imutils.video import FPS

# MODEL AND DEFINTION

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_FROZEN_GRAPH = 'c:/Tensorflow/models/research/'+MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8) 

def run_inference_for_single_image(image, graph,sess):
	ops = tf.get_default_graph().get_operations()
	all_tensor_names = {output.name for op in ops for output in op.outputs}
	tensor_dict = {}
	for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
		tensor_name = key + ':0'
		if tensor_name in all_tensor_names:
			tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
	if 'detection_masks' in tensor_dict:
		detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
		detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
		real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
		detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
		detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
		detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
		detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
		tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
	image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
	output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})
	output_dict['num_detections'] = int(output_dict['num_detections'][0])
	output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
	output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
	output_dict['detection_scores'] = output_dict['detection_scores'][0]
	if 'detection_masks' in output_dict:
		output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict

# DATABASE

mydb=mysql.connector.connect(host='localhost', user='root', password='', database='project')

def next_button_func(x):
	x=x+1
	mycursor=mydb.cursor()
	mycursor.execute('SELECT COUNT(DISTINCT DATESTMP) FROM TEST')
	myresult=mycursor.fetchone()
	if(myresult[0]<=x*8):
		x=x-1
	update_past_surv(x)

def prev_button_func(x):
	x=x-1
	if(x<0):
		x=0
	update_past_surv(x)

def update_past_surv(x):
	for widget in past_surv_pane.winfo_children():
		widget.destroy()
	mycursor=mydb.cursor()
	mycursor.execute('SELECT DISTINCT DATESTMP FROM TEST ORDER BY DATESTMP DESC LIMIT 8 OFFSET '+str(x*8))
	myresult = mycursor.fetchall()
	r=len(myresult)
	if(r==0):
		Label(past_surv_pane,text='No database exists.',font=font.Font(size=10)).grid(row=0,column=0,columnspan=2,pady=4,padx=10,ipadx=10,ipady=2)
	if(r>0):
		Button(past_surv_pane,text='DATE: '+myresult[0][0],command=lambda: update_past_surv_extd(myresult[0][0],0),font=font.Font(size=10)).grid(row=0,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>1):
		Button(past_surv_pane,text='DATE: '+myresult[1][0],command=lambda: update_past_surv_extd(myresult[1][0],0),font=font.Font(size=10)).grid(row=1,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>2):
		Button(past_surv_pane,text='DATE: '+myresult[2][0],command=lambda: update_past_surv_extd(myresult[2][0],0),font=font.Font(size=10)).grid(row=2,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>3):
		Button(past_surv_pane,text='DATE: '+myresult[3][0],command=lambda: update_past_surv_extd(myresult[3][0],0),font=font.Font(size=10)).grid(row=3,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>4):
		Button(past_surv_pane,text='DATE: '+myresult[4][0],command=lambda: update_past_surv_extd(myresult[4][0],0),font=font.Font(size=10)).grid(row=4,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>5):
		Button(past_surv_pane,text='DATE: '+myresult[5][0],command=lambda: update_past_surv_extd(myresult[5][0],0),font=font.Font(size=10)).grid(row=5,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>6):
		Button(past_surv_pane,text='DATE: '+myresult[6][0],command=lambda: update_past_surv_extd(myresult[6][0],0),font=font.Font(size=10)).grid(row=6,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>7):
		Button(past_surv_pane,text='DATE: '+myresult[7][0],command=lambda: update_past_surv_extd(myresult[7][0],0),font=font.Font(size=10)).grid(row=7,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)

	prev_button=Button(past_surv_pane,text='Prev',command=lambda: prev_button_func(x),font=font.Font(size=10))
	prev_button.grid(row=max(r,1), column=0,pady=4,padx=5,ipadx=10,ipady=2,stick=E)
	next_button=Button(past_surv_pane,text='Next',command=lambda: next_button_func(x),font=font.Font(size=10))
	next_button.grid(row=max(r,1), column=1,pady=4,padx=5,ipadx=10,ipady=2,stick=E)


def next_button_func_extd(var,x):
	x=x+1
	mycursor=mydb.cursor()
	mycursor.execute("SELECT COUNT(*) FROM TEST WHERE DATESTMP='"+var+"'")
	myresult=mycursor.fetchone()
	if(myresult[0]<=x*7):
		x=x-1
	update_past_surv_extd(var,x)

def prev_button_func_extd(var,x):
	x=x-1
	if(x<0):
		x=0
	update_past_surv_extd(var,x)

def update_past_surv_extd(var,x):
	for widget in past_surv_pane.winfo_children():
		widget.destroy()
	mycursor=mydb.cursor()
	mycursor.execute("SELECT DISTINCT TIMESTMP FROM TEST WHERE DATESTMP='"+var+"' ORDER BY TIMESTMP DESC LIMIT 7 OFFSET "+str(x*7))
	myresult = mycursor.fetchall()
	r=len(myresult)
	Button(past_surv_pane,text=' DATE: '+var,font=font.Font(size=10),bg='lightgrey').grid(row=0,column=0,columnspan=2,pady=4,padx=10,ipadx=15,ipady=2)
	if(r==0):
		Label(past_surv_pane,text='No object was \ndetected at \nthis instance.',font=font.Font(size=10)).grid(row=1,column=0,columnspan=2,pady=4,padx=10,ipadx=10,ipady=2)
	if(r>0):
		Button(past_surv_pane,text='TIME: '+myresult[0][0],command=lambda: update_past_surv_double_extd(var,myresult[0][0],0),font=font.Font(size=10)).grid(row=1,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>1):
		Button(past_surv_pane,text='TIME: '+myresult[1][0],command=lambda: update_past_surv_double_extd(var,myresult[1][0],0),font=font.Font(size=10)).grid(row=2,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>2):
		Button(past_surv_pane,text='TIME: '+myresult[2][0],command=lambda: update_past_surv_double_extd(var,myresult[2][0],0),font=font.Font(size=10)).grid(row=3,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>3):
		Button(past_surv_pane,text='TIME: '+myresult[3][0],command=lambda: update_past_surv_double_extd(var,myresult[3][0],0),font=font.Font(size=10)).grid(row=4,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>4):
		Button(past_surv_pane,text='TIME: '+myresult[4][0],command=lambda: update_past_surv_double_extd(var,myresult[4][0],0),font=font.Font(size=10)).grid(row=5,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>5):
		Button(past_surv_pane,text='TIME: '+myresult[5][0],command=lambda: update_past_surv_double_extd(var,myresult[5][0],0),font=font.Font(size=10)).grid(row=6,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>6):
		Button(past_surv_pane,text='TIME: '+myresult[6][0],command=lambda: update_past_surv_double_extd(var,myresult[6][0],0),font=font.Font(size=10)).grid(row=7,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)

	prev_button=Button(past_surv_pane,text='Prev',command=lambda: prev_button_func_extd(var,x),font=font.Font(size=10))
	prev_button.grid(row=max(r+1,2), column=0,pady=4,padx=5,ipadx=10,ipady=2,stick=E)
	next_button=Button(past_surv_pane,text='Next',command=lambda: next_button_func_extd(var,x),font=font.Font(size=10))
	next_button.grid(row=max(r+1,2), column=1,pady=4,padx=5,ipadx=10,ipady=2,stick=E)
	next_button=Button(past_surv_pane,text='Back',command=lambda: update_past_surv(0),font=font.Font(size=10))
	next_button.grid(row=max(r+2,3), column=0,columnspan=2,pady=4,padx=5,ipadx=10,ipady=2)

def next_button_func_double_extd(var1,var2,x):
	x=x+1
	mycursor=mydb.cursor()
	mycursor.execute("SELECT COUNT(*) FROM TEST WHERE DATESTMP='"+var1+"' AND TIMESTMP='"+var2+"'")
	myresult=mycursor.fetchone()
	if(myresult[0]<=x*6):
		x=x-1
	update_past_surv_double_extd(var1,var2,x)

def prev_button_func_double_extd(var1,var2,x):
	x=x-1
	if(x<0):
		x=0
	update_past_surv_double_extd(var1,var2,x)

def update_past_surv_double_extd(var1,var2,x):
	for widget in past_surv_pane.winfo_children():
		widget.destroy()
	mycursor=mydb.cursor()
	mycursor.execute("SELECT STORAGE FROM TEST WHERE DATESTMP='"+var1+"' AND TIMESTMP='"+var2+"' LIMIT 6 OFFSET "+str(x*6))
	myresult = mycursor.fetchall()
	r=len(myresult)
	
	Button(past_surv_pane,text=' DATE: '+var1,font=font.Font(size=10),bg='lightgrey').grid(row=0,column=0,columnspan=2,pady=4,padx=10,ipadx=15,ipady=2)

	Button(past_surv_pane,text=' TIME: '+var2,font=font.Font(size=10),bg='lightgrey').grid(row=1,column=0,columnspan=2,pady=4,padx=10,ipadx=15,ipady=2)
	
	if(r==0):
		Label(past_surv_pane,text='No object was \ndetected at \nthis instance.',font=font.Font(size=10)).grid(row=1,column=0,columnspan=2,pady=4,padx=10,ipadx=10,ipady=2)
	if(r>0):
		Button(past_surv_pane,text=myresult[0][0][39:],command=lambda: img_func(myresult[0][0]),font=font.Font(size=10)).grid(row=2,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>1):
		Button(past_surv_pane,text=myresult[1][0][39:],command=lambda: img_func(myresult[1][0]),font=font.Font(size=10)).grid(row=3,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>2):
		Button(past_surv_pane,text=myresult[2][0][39:],command=lambda: img_func(myresult[2][0]),font=font.Font(size=10)).grid(row=4,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>3):
		Button(past_surv_pane,text=myresult[3][0][39:],command=lambda: img_func(myresult[3][0]),font=font.Font(size=10)).grid(row=5,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>4):
		Button(past_surv_pane,text=myresult[4][0][39:],command=lambda: img_func(myresult[4][0]),font=font.Font(size=10)).grid(row=6,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)
	if(r>5):
		Button(past_surv_pane,text=myresult[5][0][39:],command=lambda: img_func(myresult[5][0]),font=font.Font(size=10)).grid(row=7,column=0,columnspan=2,pady=4,padx=10,ipadx=30,ipady=2)

	prev_button=Button(past_surv_pane,text='Prev',command=lambda: prev_button_func_double_extd(var1,var2,x),font=font.Font(size=10))
	prev_button.grid(row=max(r+2,3), column=0,pady=4,padx=5,ipadx=10,ipady=2,stick=E)
	next_button=Button(past_surv_pane,text='Next',command=lambda: next_button_func_double_extd(var1,var2,x),font=font.Font(size=10))
	next_button.grid(row=max(r+2,3), column=1,pady=4,padx=5,ipadx=10,ipady=2,stick=E)
	next_button=Button(past_surv_pane,text='Back',command=lambda: update_past_surv_extd(var1,0),font=font.Font(size=10))
	next_button.grid(row=max(r+3,4), column=0,columnspan=2,pady=4,padx=5,ipadx=10,ipady=2)

def img_func(s):
	details_pane_canvas.delete('all')
	global image
	image =Image.open(s)
	global img
	img = ImageTk.PhotoImage(image.resize((400, 250), Image.ANTIALIAS))  
	details_pane_canvas.create_image(0, 0, image=img, anchor=NW)  
	details_pane_canvas.update()
	mycursor=mydb.cursor()
	mycursor.execute('SELECT * FROM TEST WHERE STORAGE=\''+s+'\'')
	myresult=mycursor.fetchone()
	info=""
	if(myresult[5]=='0'):
		info+='WEBCAM '
	else:
		info+='UPLOADED '
	if(myresult[6]=='0'):
		info+='VIDEO'
	else:
		info+='GRAPHIC'
	details_pane_info['text']=info+'\n'+'DATESTAMP : '+str(myresult[0])+'    '+'TIMESTAMP : '+str(myresult[2])+'\n'+'STORAGE : '+str(myresult[4])+'\n'+'DETAILS : '+str(myresult[3])

#START WEBCAM VIDEO

def start_func():
	mysess=str(datetime.datetime.now()).split()
	dirname="C:/Tensorflow/models/research/object_detection/project_ouput/User/"+mysess[0]
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	dirname=dirname+"/"+re.sub(':','-',mysess[1][:8])
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	y=0
	pset=set()
	cap = cv2.VideoCapture(0)
	with detection_graph.as_default():
		with tf.Session() as sess:
			fps=FPS()
			fps.start()
			while True:
			    ret, image_np = cap.read()
			    image_np_expanded = np.expand_dims(image_np, axis=0)
			    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph,sess)
			    vis_util.visualize_boxes_and_labels_on_image_array(
			        image_np,
			        output_dict['detection_boxes'],
			        output_dict['detection_classes'],
			        output_dict['detection_scores'],
			        category_index,
			        instance_masks=output_dict.get('detection_masks'),
			        use_normalized_coordinates=True,
			        line_thickness=8)
			    j=0
			    find=""
			    cset=set()
			    while(j<10 and output_dict['detection_scores'][j]>0.5):
			    	cset.add(category_index.get(output_dict['detection_classes'][j])['name'])
			    	if(j!=0 and j%3==2):
			    		find=find+'\n'
			    	find=find+str(category_index.get(output_dict['detection_classes'][j])['name'])+" : "+str((output_dict['detection_scores'][j])*100)[:5]+"% ; "
			    	j=j+1
			    if(len(pset-cset)!=0 or len(cset-pset)!=0):
			    	pset.clear()
			    	for x in cset:
			    		pset.add(x)
			    	if(len(cset)!=0):
				    	mycursor=mydb.cursor()
				    	timestmp=str(datetime.datetime.now())
				    	storage=dirname[47:]+"/IMG_"+str(y)+".jpg"
				    	sql="INSERT INTO TEST VALUES(%s,%s,%s,%s,%s,%s,%s)"
				    	val=(mysess[0],mysess[1][:8],timestmp[11:19],find,storage,str(0),str(0))
				    	mycursor.execute(sql,val)
				    	mydb.commit()
				    	location=dirname+"/IMG_"+str(y)+".jpg"
				    	y=y+1
				    	cv2.imwrite(location,image_np)
			    cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
			    if cv2.waitKey(1) == ord('q'):
			        cv2.destroyAllWindows()
			        break
			    fps.update()
			fps.stop()
			print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
			print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	update_past_surv(0)

# UPLOAD VIDEO FUNCTION

def upload_func():
	file = askopenfile(mode ='r') 
	mysess=str(datetime.datetime.now()).split()
	dirname="C:/Tensorflow/models/research/object_detection/project_ouput/User/"+mysess[0]
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	dirname=dirname+"/"+re.sub(':','-',mysess[1][:8])
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	y=0
	pset=set()
	cnt=0
	cap = cv2.VideoCapture(file.name)
	with detection_graph.as_default():
		with tf.Session() as sess:
			while True:
			    ret, image_np = cap.read()
			    cap.set(cv2.CAP_PROP_POS_MSEC,(cnt*1000))
			    image_np_expanded = np.expand_dims(image_np, axis=0)
			    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph,sess)
			    vis_util.visualize_boxes_and_labels_on_image_array(
			        image_np,
			        output_dict['detection_boxes'],
			        output_dict['detection_classes'],
			        output_dict['detection_scores'],
			        category_index,
			        instance_masks=output_dict.get('detection_masks'),
			        use_normalized_coordinates=True,
			        line_thickness=8)
			    j=0
			    cset=set()
			    find='Time: '+str(cnt*1)+'s >> ['
			    while(j<10 and output_dict['detection_scores'][j]>0.5):
			    	cset.add(category_index.get(output_dict['detection_classes'][j])['name'])
			    	if(j!=0 and j%3==2):
			    		find=find+'\n'
			    	find=find+str(category_index.get(output_dict['detection_classes'][j])['name'])+" : "+str((output_dict['detection_scores'][j])*100)[:5]+"% ; "
			    	j=j+1
			    find=find+']'
			    if(len(pset-cset)!=0 or len(cset-pset)!=0):
			    	pset.clear()
			    	for x in cset:
			    		pset.add(x)
			    	if(len(cset)!=0):
				    	mycursor=mydb.cursor()
				    	timestmp=str(datetime.datetime.now())
				    	storage=dirname[47:]+"/IMG_"+str(y)+".jpg"
				    	sql="INSERT INTO TEST VALUES(%s,%s,%s,%s,%s,%s,%s)"
				    	val=(mysess[0],mysess[1],timestmp[11:19]+' s',find,storage,str(1),str(0))
				    	mycursor.execute(sql,val)
				    	mydb.commit()
				    	location=dirname+"/IMG_"+str(y)+".jpg"
				    	y=y+1
				    	cv2.imwrite(location,image_np)
			    cnt=cnt+1
			    cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
			    if cv2.waitKey(1) == ord('q'):
			        cv2.destroyAllWindows()
			        break
	cv2.destroyAllWindows()
	update_past_surv(0)

# UPLOAD IMAGE FUNCTION

def upload_img_func():
	file = askopenfile(mode ='r') 
	mysess=str(datetime.datetime.now()).split()
	dirname="C:/Tensorflow/models/research/object_detection/project_ouput/User/"+mysess[0]
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	dirname=dirname+"/"+re.sub(':','-',mysess[1][:8])
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	with detection_graph.as_default():
		with tf.Session() as sess:
			cap = cv2.VideoCapture(file.name)
			ret, image_np = cap.read()
			image_np_expanded = np.expand_dims(image_np, axis=0)
			output_dict = run_inference_for_single_image(image_np_expanded, detection_graph,sess)
			vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				output_dict['detection_boxes'],
				output_dict['detection_classes'],
				output_dict['detection_scores'],
				category_index,
				instance_masks=output_dict.get('detection_masks'),
				use_normalized_coordinates=True,
				line_thickness=8)
			j=0
			find=""
			while(j<10 and output_dict['detection_scores'][j]>0.5):
				if(j!=0 and j%3==2):
					find=find+'\n'
				find=find+str(category_index.get(output_dict['detection_classes'][j])['name'])+" : "+str((output_dict['detection_scores'][j])*100)[:5]+"% ; "
				j=j+1
			mycursor=mydb.cursor()
			timestmp=str(datetime.datetime.now())
			storage=dirname[47:]+"/IMG.jpg"
			sql="INSERT INTO TEST VALUES(%s,%s,%s,%s,%s,%s,%s)"
			val=(mysess[0],mysess[1],timestmp[11:19],find,storage,str(1),str(1))
			mycursor.execute(sql,val)
			mydb.commit()
			location=dirname+"/IMG.jpg"
			cv2.imwrite(location,image_np)
	update_past_surv(0)

# START WEBCAM PHOTO

def start_img_func():
	mysess=str(datetime.datetime.now()).split()
	dirname="C:/Tensorflow/models/research/object_detection/project_ouput/User/"+mysess[0]
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	dirname=dirname+"/"+re.sub(':','-',mysess[1][:8])
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	with detection_graph.as_default():
		with tf.Session() as sess:
			cap = cv2.VideoCapture(0)
			time.sleep(1)
			ret, image_np = cap.read()
			image_np_expanded = np.expand_dims(image_np, axis=0)
			output_dict = run_inference_for_single_image(image_np_expanded, detection_graph,sess)
			vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				output_dict['detection_boxes'],
				output_dict['detection_classes'],
				output_dict['detection_scores'],
				category_index,
				instance_masks=output_dict.get('detection_masks'),
				use_normalized_coordinates=True,
				line_thickness=8)
			j=0
			find=""
			while(j<10 and output_dict['detection_scores'][j]>0.5):
				if(j!=0 and j%3==2):
					find=find+'\n'
				find=find+str(category_index.get(output_dict['detection_classes'][j])['name'])+" : "+str((output_dict['detection_scores'][j])*100)[:5]+"% ; "
				j=j+1
			mycursor=mydb.cursor()
			timestmp=str(datetime.datetime.now())
			storage=dirname[47:]+"/IMG.jpg"
			sql="INSERT INTO TEST VALUES(%s,%s,%s,%s,%s,%s,%s)"
			val=(mysess[0],mysess[1],timestmp[11:19],find,storage,str(0),str(1))
			mycursor.execute(sql,val)
			mydb.commit()
			location=dirname+"/IMG.jpg"
			cv2.imwrite(location,image_np)
			cv2.destroyAllWindows()
	update_past_surv(0)

# TAB SELECTED

def tab_func(x):
	global start_button
	global upload_button
	global vag_tab
	global video_button
	global graphic_button
	if(x==1):
		start_button['text']='TAKE  PHOTO'
		start_button['command']=start_img_func
		upload_button['text']='UPLOAD IMAGE'
		upload_button['command']=upload_img_func
	else:
		start_button['text']='START VIDEO'
		start_button['command']=start_func
		upload_button['text']='UPLOAD VIDEO'
		upload_button['command']=upload_func
	for widget in vag_tab.winfo_children():
		widget.destroy()
	video_button=Button(vag_tab,text='VIDEO',command=lambda: tab_func(0),font=font.Font(size=14))
	graphic_button=Button(vag_tab,text='GRAPHIC',command=lambda: tab_func(1),font=font.Font(size=14))
	if(x==0):
		video_button['bg']='green'
	else:
		graphic_button['bg']='green'
	video_button.grid(row=0,column=0,padx=5)
	graphic_button.grid(row=0,column=1,padx=5)

# MAIN PROGRAM AND GUI

root=Tk()

top_label=Label(root,text='AUTOMATED TRESPASS DETECTION USING OBJECT DETECTION MODELS',font=font.Font(size=20),bd=1,bg='gray',fg='white')
top_label.grid(row=0,column=0,columnspan=2,ipadx=50,pady=10)

user_label=Label(root,text='Welcome, User    ',font=font.Font(size=12))
user_label.grid(row=1,column=0,columnspan=2,padx=20,pady=10,sticky=E)

surv_tab_label=Label(root,text='Surveillance Tab',bg='black',fg='white',font=font.Font(size=14))
past_surv_tab_label=Label(root,text='Past Surveillances',bg='black',fg='white',font=font.Font(size=14))
surv_tab_label.grid(row=2,column=0,ipadx=10,ipady=4)
past_surv_tab_label.grid(row=2,column=1,ipadx=10,ipady=4)

# SURVEILLANCE TAB
surv_frame=Frame(root)
web_cam_label=Label(surv_frame,text='Start a new surveillance\n session.(press \'q\' to stop)',font=font.Font(size=12))
upload_video_label=Label(surv_frame,text='Upload a video to spot\n any tresspassing.',font=font.Font(size=12))
web_cam_label.grid(row=0,column=0,sticky=N,pady=10,padx=10)
upload_video_label.grid(row=0,column=1,sticky=N,pady=10,padx=10)
start_button=Button(surv_frame,text='START VIDEO',command=start_func,bg='navyblue',fg='white',font=font.Font(size=14))
upload_button=Button(surv_frame,text='UPLOAD VIDEO',command=upload_func,bg='navyblue',fg='white',font=font.Font(size=14))
start_button.grid(row=1,column=0,sticky=N,padx=15,pady=50)
upload_button.grid(row=1,column=1,sticky=N,padx=15,pady=50)
surv_frame.grid(row=3,column=0,padx=30,pady=12)

# PAST SURVEILLANCES
past_surv_frame=Frame(root)
past_surv_pane=Frame(past_surv_frame,bg='white')
update_past_surv(0)
past_surv_pane.grid(row=0,column=0,padx=5,pady=5)
details_pane=Frame(past_surv_frame,bg='white')
details_pane.grid(row=0,column=1,padx=5,pady=5)
image=Image.open('test_image_1.jpg')
details_pane_canvas = Canvas(details_pane)
details_pane_canvas.grid(row=0,column=0,padx=10,pady=10)  
img = ImageTk.PhotoImage(image.resize((400, 250), Image.ANTIALIAS))  
details_pane_canvas.create_image(0, 0, image=img, anchor=NW)  
details_pane_info=Label(details_pane,bg='white',text='The information regarding the\n image will be displayed here.',font=font.Font(size=11))
details_pane_info.grid(row=1,column=0,ipadx=10,ipady=10)
past_surv_frame.grid(row=3,column=1,padx=10,pady=10,ipadx=10,ipady=10)

# VIDEO AND GRAPHICS TAB
vag_tab=Frame(root)
video_button=Button(vag_tab,text='VIDEO',command=lambda: tab_func(0),font=font.Font(size=14),bg='green')
graphic_button=Button(vag_tab,text='GRAPHIC',command=lambda: tab_func(1),font=font.Font(size=14))
video_button.grid(row=0,column=0,padx=5)
graphic_button.grid(row=0,column=1,padx=5)
vag_tab.grid(row=4,column=0,columnspan=2,sticky=E,padx=30)

# STATUS BAR
status_bar=Label(root,text='Coded and Developed by : Nilesh Tanwar',relief=SUNKEN,bd=1,font=font.Font(size=12))
status_bar.grid(row=5,column=0,columnspan=2,sticky=W,padx=5,pady=10)

root.mainloop()
