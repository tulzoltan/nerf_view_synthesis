# nerf_view_synthesis

Perform camera calibration by placing pictures taken of a checkerboard in the directory ./images/calibration and running 
```
python calibration.py
```

Next the input images should be processed. Camera positions will be inferred using structure from motion method. The command 
```
python process_images.py
```
will produce the ray origin and direction vectors for every pixel and print them along with the RGB value of corresponding pixel in training_data.pkl

Train neural network by running
```
python nerf.py
```

Generate novel views with the code 
```
python generate_views.py
```

Visualize camera path and orientations with
```
python visualization.py
```
The camera position and direction for novel views will be shown in blue.
