"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Paper link: 

Last updated Date: April 28th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_release.py

(1) data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
(2) data_save: Save preprocessed data into public and private datasets
"""

## Necessary Packages
import numpy as np
import pickle
import sys
sys.path.append('../')
from data.data_utils import MinMaxScaler, data_division


def data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('raw_data/stock_data.csv', delimiter = ",",skiprows = 1)
        
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data


def data_save():
  """Save the preprocessed data to public and private datasets
  """
  
  # Set parameters
  data_name = 'stock'
  seq_len = 7
  seed = 0
  divide_rates = [(1./3), (2./3)]

  # Load data
  data = data_loading(data_name, seq_len)
  
  # Divide data
  divided_data, divided_index = data_division(data, seed, divide_rates)
  
  # Save the public and private data
  with open('../data/public_data/public_' + data_name + '_data.txt', 'wb') as fp:
    pickle.dump(divided_data[0], fp)
  with open('private_data/private_' + data_name + '_data.txt', 'wb') as fp:
    pickle.dump(divided_data[1], fp)  