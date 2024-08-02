# Predicting chemotherapy-induced thrombotoxicity by NARX neural networks and transfer learning
This repository contains a demonstration for learning an individual patients time series with NARX networks. It accompanies the puplication "Predicting chemotherapy-induced thrombotoxicity by NARX neural networks and transfer learning" by M. Steinacker, Y. Kheifetz and M. Scholz. 

# Usage
In the notebook `NARX_showcase.ipynb` it is demonstrated how individual 
patient dynamics can be learned with NARX neural networks, and how transfer learning via a semi-mechanistic model can be utilized. This is shown with a simulated individual patient time series, which is generated with the model of Friberg et al., 2002 (https://doi.org/10.1200/jco.2002.02.140), and an added simulated noise.  

# How To
1. Clone this environment to your computer.    
2. Create Python virtual environment (Python 3.9) and install dependencies.   
   I recommend using pip with the given requirements in requirements.txt.    
3. Activate the new virtual environment.   
4. Execute the jupyter notebook `NARX_showcase.ipynb` from the main folder. 
   This will take around 10 minutes on modern hardware.

### Hardware Requirements
The demonstrator notebook runs on a local pc with 8 cores and 16 GB RAM.

### Software Versions Used 
   - Python 3.9   
   - NumPy 1.26.3   
   - TensorFlow 2.14.1   
   - Keras 2.14   
   - scikit-learn 1.4   
   - SciPy 1.11.4    
   - Matplotlib 3.5.1
   - pandas 1.5.3
   
# File description
- `NARX_showcase.ipynb` Demonstration notebook. A simulated patients 
  dynamics are learned with a NARX network and compared with a calibrated 
  semi-mechanistic model. The NARX framework and the learning scenario can 
  be adjusted.
- `friberg.py` Implementation of the model of Friberg et al.
- `NARX.py` Implementation of ARX-RNN and ARX-FNN model classes.
- `utils.py` Different helper and utility functions for model training.
- `therapies_all.csv` Generated treatment scenarios for transfer learning, 
  as used in Steinacker et al., 2023 (https://doi.org/10.1016/j.heliyon.2023.e17890). 
- `index_weights_FNN.h5` FNN starting weights for transfer learning.
- `index_weights_GRU.h5` GRU starting weights for transfer learning.

# Issues 
If you encounter any issues in this notebook or have questions, please 
open an issue in the repository. 