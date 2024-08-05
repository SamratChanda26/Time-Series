Time Series Weather Analysis and Forecasting

Project Description
This project focuses on analyzing and forecasting weather data from Sydney, Australia. The model uses the NeuralProphet library to predict future temperatures based on historical data.

Table of Contents

1. Introduction
2. Technologies Used
3. Requirements
4. Installation Instructions
5. Usage Instructions
6. Features
7. Dataset
8. Model
9. Result
10. Documentation
11. Visuals
12. Conclusion

Introduction
This project aims to predict the temperature in Sydney, Australia, using historical weather data. The dataset contains various weather-related metrics, and the model is trained to forecast future temperatures.

Technologies Used
Python
Pandas
NeuralProphet
Matplotlib
Pickle

Requirements
Python 3.x
Pandas
NeuralProphet
Matplotlib
Pickle

Installation Instructions
1. Clone the repository.
git clone https://github.com/SamratChanda26/Time-Series.git

2. Install the required dependencies.
pip install pandas neuralprophet matplotlib pickle

Usage Instructions

1. Run the script to read and preprocess the data.
df = pd.read_csv('weatherAUS.csv')
syd = df[df['Location'] == 'Sydney']
syd['Date'] = pd.to_datetime(syd['Date'])
data = syd[['Date', 'Temp3pm']]
data.dropna(inplace=True)
data.columns = ['ds', 'y']

2. Train the NeuralProphet model.
m = NeuralProphet()
m.fit(data, freq='D', epochs=1000)

3. Forecast future temperatures.
future = m.make_future_dataframe(data, periods=900)
forecast = m.predict(future)

4. Save and load the trained model.
with open('forecast_model.pkl', "wb") as f:
    pickle.dump(m, f)
with open('forecast_model.pkl', "rb") as f:
    m = pickle.load(f)

Features
Data preprocessing and cleaning
Temperature prediction using NeuralProphet
Forecast visualization

Dataset
The dataset used in this project is the "weatherAUS.csv" file, which contains weather data for various locations in Australia.

Model
The NeuralProphet model is used for forecasting temperatures. It is trained on historical temperature data from Sydney.

Result
The model forecasts future temperatures for Sydney. The results can be visualized using Matplotlib.

Documentation
Pandas Documentation
NeuralProphet Documentation
Matplotlib Documentation

Visuals
Temperature data plot:
plt.plot(syd['Date'], syd['Temp3pm'])
plt.show()

Forecast plot:
plot1 = m.plot(forecast)
plot1.show()
plot2 = m.plot_components(forecast)
plot2.show()

Conclusion
This project demonstrates the process of analyzing and forecasting weather data using NeuralProphet. The model successfully predicts future temperatures based on historical data, providing valuable insights for weather analysis.