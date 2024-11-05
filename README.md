# MLOps Tool Demonstration with Aim
Note: Instead of forking my entire group project in this public repository, I have included only the models.py script, which contains the demonstration code for the MLOps tool, Aim, along with a parquet data file (20241027_203929.parquet).

The parquet file contains data from Kafka stream logs in the format <time>,<userid>,GET /rate/<movieid>=<rating>. These logs are aggregated into a dictionary, converted into a Pandas DataFrame, and stored in a parquet file. The data is collected chronologically in the order received from the Kafka stream, representing a sequence of user interactions in a movie recommendation scenario.

## Overview
The models.py Python script is used to demonstrate the capabilities of Aim in tracking machine learning experiment metrics. Aim allows us to log and visualize metrics such as training time, inference time, RMSE, and model size. This script illustrates the use of Aim in a simulated production environment, utilizing data from the movie streaming scenario.

You can read more about the implementation and the role of Aim in my blog post here - https://medium.com/@shreya.tallam/ml-experiment-tracking-with-aim-f77499ee5e3c

