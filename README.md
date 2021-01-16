#AI2020f  

|作業編號|作業說明| 
|--- |--- |
|作業一| mnist手寫數字辨識 |
|作業二、三| nn結構visualization |
|作業四| Autoencoder實作 |
|作業五| VAE實作|
|作業六 | CNN Multi-Class Classification|
|作業七 | heat-map |




##HW1
![title](MD_images/hw1.png)

##HW2

Lit up activated neurons. Visualize how neural network (pretty shallow in this case though) works.

![title](MD_images/hw2.png)  
##HW3 
Shows that what it looks like in the dataset. For exmaple, we can sum up and get average look of each handwritten digits in the training dataset. 

![title](MD_images/hw3/hw3_digits.png)

We also get to know the distribution in the dataset. (x-axis=labels, y-aixs=number of data)
![title](MD_images/hw3/distribution.png)
  
##HW4
#### Implement autoencoder.
![title](MD_images/hw4/hw4.png)
### ENCODER
![title](MD_images/hw4/encoder.png)  
### DECODER
![title](MD_images/hw4/decoder.png)


##HW5

####Implement VAE.
![title](MD_images/hw5/hw5.png)

###ENCODER
![title](MD_images/hw5/vae_encoder_plot.png)
###DECODER
![title](MD_images/hw5/vae_decoder_plot.png)  


##HW6 

Implement a multi-label (class) classifier to classify 11 kinds of food (Dataset: food11). The model here is kind of a VGG16 mockup, not entirely the same structure. Validation accuracy stayed around ~80% after 80 epoches. Snatched 2nd place (private leaderboard) in kaggle in-class competition.


![title](MD_images/hw6/kaggle.png)
 
![title](MD_images/hw6/VGG16_implement.png)

##HW7

Load the weights file from HW6 and use Flask to implement heat-map on web. (The .h5 file is removed since its size exceeds github limit)

![title](MD_images/hw7.png)
