# Face-Style-transform-by-CycleGAN-and-VGG
Use CycleGAN model to transform real face pictures into cartoon, sketch and oil painting styles.

This project trained 3 different CycleGAN models to transform a real face picture into other 3 styles.

To collect data(human faces), used the styleGAN model to generate 1000 model faces.

![image](https://github.com/JunanPan/pics/raw/main/2210230.png)  

As for corresponding pictures of other styles, use API interface from Tencent and Baidu AI platforms to make those.

![image](https://github.com/JunanPan/pics/raw/main/2210231.png)  

For details of CycleGAN, reference [my blog](https://blog.csdn.net/weixin_44492824/article/details/124943553?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166651876816782428678954%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=166651876816782428678954&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-124943553-null-null.nonecase&utm_term=Cycle&spm=1018.2226.3001.4450) here. 

After training, I deploy the model on Flask and make a webpage by Vue to display the result. Rented a Ubuntu from Alibaba Cloud Server to run the service.

![image](https://github.com/JunanPan/pics/raw/main/2210232.png)  
![image](https://github.com/JunanPan/pics/raw/main/2210233.png)  

VGG network is another common network to convert style, the result of VGG:  

![image](https://github.com/JunanPan/pics/raw/main/2210234.png)  
