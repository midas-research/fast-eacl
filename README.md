# FAST: Financial news and tweet based time Aware network for Stock Trading

This codebase contains the python scripts for FAST: Financial News and Tweet Based Time Aware Network for Stock Trading.

EACL - The 16th Conference of the European Chapter of the Association for Computational Linguistics 2021 paper [coming soon](#)

## Environment & Installation Steps
Create an environment having Python 3.6 and install the following dependencies
```python
pip install -r requirements.txt
```
## Contents
1. train.py consists of the training, validation, and testing scripts.
2. model.py comprises the definition of FAST and its components.
3. evaluator.py contains the script to evaluate model performance.
4. processed_data comprises the processed data in .npy files.

## Data
Find the US S&P 500 data [here](https://github.com/yumoxu/stocknet-dataset), and the China & Hong Kong data [here](https://pan.baidu.com/s/1mhCLJJi).

## Training
Execute the following command in the same environment to train and test FAST:
```bash
python3 train.py
```
## Processed data
Each .npy file corresponds to data about one lookback period.
FAST takes the following data as inputs for training and evaluation:
1. Text: Text embeddings for each day in the lookback for all stocks in the dataset. Dimensions: (number of stocks, length of lookback period, number of texts, embedding size)
2. Timestamps: Inverse of time intervals between release of consecutive texts in a day, for all days, for all stocks. Dimensions: (number of stocks, length of lookback period, number of texts, 1)
3. Mask: A single value for each stock, set to 0 if the daily return ratio is small, or 1 otherwise. Dimensions: (number of stocks, 1)
4. Price: A single value for each stock, indicating the normalized adjusted closing price across all stocks. Dimensions: (number of stocks, 1)
5. Ground Truth: A single value for each stock, indicating the trading day's actual normalized return ratio. Dimensions: (number of stocks, 1)
In the folder titled processed_data, we provide some sample data for clarity. 

## Preprocessing
To encode the texts, we use the 768-dimensional embedding obtained per news item or tweet by averaging the token-level outputs from the final layer of BERT. However, FAST is compatible with any and all 1-D text embeddings.
To extract the timestamp input, we obtain the time interval (in minutes) between the release of two consecutive texts and compute its inverse.
For masking to 0, we keep the return ratio threshold of 0.05.
We normalize the adjusted closing prices and the return ratios across all stocks in the dataset.

## Cite

If our work was helpful for your research, please kindly cite our work.
