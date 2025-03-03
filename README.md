# A Plug and Play Algorithm for 3D Video Super-Resolution of single-photon data
Repository of the paper 'A Plug and Play Algorithm for 3D Video Super-Resolution of single-photon data'

![alt text](https://github.com/aliceruget/PlugandPlay_Algorithm/blob/main/NewConcept2.png?raw=true)

## Prerequisites 
Download Super-resolution network HVSR from: https://github.com/facebookresearch/DVSR

## Simulation of dataset
The script create_dataset.ipynb to simulate the scenes from the Middlebury dataset. 
Adjust the paths, and the parameters. 

## Run algorithm 
Once the dataset is simulated, run plug_and_play_algorithm.ipynb. 
Adjust the paths and parameters. 


## Plot results
Use the script plot_results.ipynb to plot the final results against the other methods Naive averaging and HVSR. 
![alt text](https://github.com/aliceruget/PlugandPlay_Algorithm/blob/main/output.png?raw=true)
