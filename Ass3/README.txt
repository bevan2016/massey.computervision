The train image can be downloaded from 
http://www.massey.ac.nz/~albarcza/gesture_dataset2012.html

1. put main. cpp and mlp_model.xml in the same folder
2. compile: g++ -std=c++11 main.cpp -o gr `pkg-config --cflags --libs opencv`
3. for static image: ./gr hand_img.jpg
4. for camera: ./gr