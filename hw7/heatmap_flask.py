from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2, re
import tensorflow as tf
PIC_SIZE = 128


####
from tensorflow.python.keras.backend import set_session

sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

####

class conv2d_heatmap:
    def __init__(self, model_path, graph):
        ###
        tf.compat.v1.disable_eager_execution()
        print(tf.executing_eagerly())
        set_session(sess)
        ###
        
        
        self.model = load_model(model_path)
        
#        self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        ##
        

        self.graph = graph
        
        #取得class對應關係
        self.class_list = None
        with open('class_list.txt', 'r') as f:
            s = f.read()
            self.class_list = s.split(',')
        
        
    
    def _preprocessing(self, input_img_path):
        img = image.load_img(input_img_path, target_size=(PIC_SIZE, PIC_SIZE))    
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        
        return img_tensor
    
    #取得所有Con2v layers的名字
    def _get_con2v_layers_name(self):
        con2v_models = []
        def find_last_model(input_str):
            if re.search('\(Conv2D\)', input_str):
                con2v_models.append(input_str.split(' ')[0])
        self.model.summary(print_fn=find_last_model)
        return con2v_models
    
    #取得預測的class
    def _pred_class(self, img_tensor):
        #使用訓練好的model進行圖片分類，取出最有可能的類別
        
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            preds = self.model.predict(img_tensor)
            pred_class = np.argmax(preds[0])
        
            #回傳class label和label名稱
            return (pred_class, self.class_list[pred_class])
    

    def _GRAD_CAM(self, pred_class, img_tensor, layer_name):
        with self.graph.as_default():
            pred_output = self.model.output[:, pred_class]
            last_conv_layer = self.model.get_layer(layer_name)
            grads = K.gradients(pred_output, last_conv_layer.output)[0]
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            iterate = K.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])
            pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
            for i in range(pooled_grads_value.shape[0]):
                conv_layer_output_value[:, :, i] = conv_layer_output_value[:, :, i] * pooled_grads_value[i]
            
            return np.mean(conv_layer_output_value, axis=-1)
    
    
    def _make_heatmap(self, heatmap, input_img_path):        
    
        heatmap = np.maximum(heatmap, 0)
        #normalize the heatmap between 0 and 1
        heatmap /= np.max(heatmap)
        
        img = cv2.imread(input_img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        #RGB -> BGR
        heatmap = np.uint8(255 * heatmap)
        
        #apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap * 0.4 + img
        
        return superimposed_img
        
    def get_all_conv_layers(self):
        return self._get_con2v_layers_name()
    
    def get_heatmap(self, input_img_path, layer_name=None, save=True):
        if layer_name is None:
            layer_name = self._get_con2v_layers_name()[-1]
        
        img = self._preprocessing(input_img_path)    
        pred_class, pred_class_name = self._pred_class(img)
        raw_heatmap = self._GRAD_CAM(pred_class, img, layer_name)
        
        out_img = self._make_heatmap(raw_heatmap, input_img_path)
        
        if save:        
            cv2.imwrite('%s_%s_output.jpg' % (pred_class_name, layer_name), out_img)
        
        return (out_img, pred_class_name)

if __name__ == '__main__':
    myHeatmap = conv2d_heatmap('weights.h5')
    
    for i in myHeatmap.get_all_conv_layers():  
        myHeatmap.get_heatmap('server_store/bd29ce6406e11c13d027435b6fb85198/bd29ce6406e11c13d027435b6fb85198.jpg', layer_name=i)