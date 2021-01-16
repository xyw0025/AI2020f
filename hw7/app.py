from heatmap_flask import conv2d_heatmap
from flask import Flask, request, Response, redirect, send_file
import cv2, hashlib, os, json, shutil
import tensorflow as tf



import threading, webbrowser


graph = tf.compat.v1.get_default_graph() 

app = Flask(__name__, static_url_path='')

hm = conv2d_heatmap('./weights.h5', graph=graph)

def calculate_file_name(f):

    md5_hash = hashlib.md5()
    md5_hash.update(f)
    hashstr = md5_hash.hexdigest()
    
    if not os.path.exists('server_store/%s' % hashstr):
        os.mkdir('server_store/%s' % hashstr)
    
    with open('server_store/%s/%s.jpg' % (hashstr,hashstr), 'wb') as fp:
        fp.write(f)
    return hashstr
    
    

@app.route("/")
def index():
    return redirect('/index.html')

    
@app.route('/makeHeatmap', methods=['POST', 'OPTIONS'])
def makeHeatmap():
    img = request.files['img'].read()
    
    hashstr = calculate_file_name(img)
    
    if not os.path.exists('server_store/%s.zip' % hashstr):
        layers = hm.get_all_conv_layers()
        pred_name = ''
        for i in range(len(layers)):
            out_img, pred_name = hm.get_heatmap('server_store/%s/%s.jpg' % (hashstr,hashstr), layer_name=layers[i], save=False)
            cv2.imwrite('server_store/%s/%s.jpg' % (hashstr, layers[i]) ,out_img)
            
        with open('server_store/%s/%s.json' % (hashstr,hashstr), 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'layers': layers,
                'predict': pred_name
            }))
        shutil.make_archive('server_store/%s' % hashstr, 'zip', 'server_store/%s' % hashstr)
        shutil.rmtree('server_store/%s' % hashstr)

    return send_file('server_store/%s.zip' % hashstr)

if __name__ == '__main__':   
    threading.Timer(1, lambda: webbrowser.open('http://127.0.0.1:5000') ).start()
    app.run(port=5000)
    