Multivariate Models for Photovoltaic Power Forecasting with Non-climatic Exogenous Variables
 Manuscript ID: IEEE LATAM Submission ID: 9697 Authors:
•	Isidro Fraga Hurtado 
•	Julio R. Gómez Sarduy 
•	Zaid García Sánchez 
•	Roy Reyes Calvo 
•	Yuri U. López 
•	Enrique C. Quispe 
________________________________________
📁 Included Scripts
This repository contains all scripts required to reproduce the simulation and numerical results presented in the article.
Script	Description

G_24h.mat	The file contains the electricity generation records for each park, organized into columns in the following order: Caguaguas (P1), Frigorífico (P2), Guasimal (P3), Marrero (P4), Mayajigüa-1 (P5), Mayajigüa-2 (P6), Venegas (P7), and Yaguaramas (P8). Each column represents the energy production of one of these parks, allowing for a detailed and comparative analysis of the generation at each facility.

MODEL_MEJOR_univ.m	This code implements a univariate prediction system for photovoltaic power generation, which means it uses data from a single park to predict the generation of that same park, using a bidirectional LSTM neural network. First, the data is divided into training, validation, and test sets. Subsequently, the data is normalized and organized into temporal sequences to feed the model. The network architecture includes a BiLSTM layer, a dropout layer to avoid overfitting, a ReLU layer, and a final regression layer. Training is performed, periodically validating the performance. Finally, the model is evaluated using metrics such as RMSE, MAE, MAPE, and the correlation coefficient, and the predictions are visualized against actual values to analyze the model's accuracy.

MODEL_MEJOR_multiv.m	This code implements a multivariate prediction system for photovoltaic power generation, which means it uses data from a single park and several adjacent parks to predict the generation of that same park, using a bidirectional LSTM neural network. First, the data is divided into training, validation, and test sets. Subsequently, the data is normalized and organized into temporal sequences to feed the model. The network architecture includes a BiLSTM layer, a dropout layer to avoid overfitting, a ReLU layer, and a final regression layer. Training is performed, periodically validating performance. Finally, the model is evaluated using metrics such as RMSE, MAE, MAPE, and the correlation coefficient, and the predictions are visualized against actual values to analyze the model's accuracy.

________________________________________
📂 Required Files
•	G_24h.mat: Required for MODEL_MEJOR_univ.m   and MODEL_MEJOR_multiv.m . Place it in the same folder as the script.
________________________________________
💻 Requirements
•	MATLAB R2024a or later.
•	No additional toolboxes are required.
________________________________________
✉️ Contact
For questions or replication of results:
ifraga@ucf.edu.cu
isidro712@gmail.com


