# Multi-Attention-Spatiotemporal-Network-for-mobile-traffic-prediction
Implementing Multi-level attention spatio-temporal network for mobile traffic prediction using Keras. 

#### Download the data 

Mobile traffic data is released by Telecom Italia and the data can be acquired [here](https://dandelion.eu/datagems/SpazioDati/telecom-sms-call-internet-mi/description/). 

You can download the first 7 days of November as demo for testing the code.

#### Cleaning data and making data set

###### data preprocesing and generating h5 File

- run : src> python datapreprocessing.py ../data/raw ../data/processed

###### making dataset

- to calculate series distance, run: src> python calSeriesDis.py 

  _notice that series distance is calculated based on weekly average traffic series, so if using only 7 days as demo for testing, the whole demo data will be used to calculate series distance_

- to generate weight matrix, run: src>python genweight.py

- to train and test data set for the model, run : src >python makedataset.py

  _test_len is specified in this py file_

#### MASTNN Model

##### Framework

![image](https://github.com/Elliebababa/Multi-Attention-Spatiotemporal-Network-for-mobile-traffic-prediction/blob/master/MASTNN_Framework.png)

##### Train and evaluate model

- src> python train.py

  _model type is specified by the parameter **modelbase**_

- baseline model STN(spatio-temporal network that incoporating 3Dconv and convlstm for forecasting)$[3]$ 

  - make dataset: src > python stn_makedataset.py
  - train and evaluate: src > python stn_model_train.py

- baseline mode ARIMA:

  - run: src > python arima_train_evaluate.py

#### Reference

- 1.[Barlacchi G , De Nadai M , Larcher R , et al. A multi-source dataset of urban life in the city of Milan and the Province of Trentino[J]. Scientific Data, 2015, 2:150055.](https://www.nature.com/articles/sdata201555) 
- 2.[Yuxuan Liang, Songyu Ke, Junbo Zhang, Xiuwen Yi, Yu Zheng, "GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction", IJCAI, 2018.](https://www.ijcai.org/proceedings/2018/0476.pdf)
- 3.[Zhang C , Patras P . Long-Term Mobile Traffic Forecasting Using Deep Spatio-Temporal Neural Networks[J]. 2017.](https://arxiv.org/abs/1712.08083)
