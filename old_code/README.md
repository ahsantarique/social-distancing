# Game-theory-BII
Pandemic Vaccine and Social Distancing Game Models

* Install libraries
```shell script
pip install -r requirements.txt
```

* Command to run the code
```shell script
python SD_variance_random_ntimes.py T epsilon alphavals num_nodes num_edge_per_node avg_data_filename raw_data_filename

To Run Social Distancing Game
python SD_variance_random_ntimes.py 50 0.0001 1,2,3,4,5,10,20,50,75,100,200,500,1000,2000,5000 100 3 avg.txt raw.txt

To Run Vaccination Game
python best_response.py 50 0.0001 1,2,3,4,5,10,20,50,75,100,200,500,1000,2000,5000 100 3 avg.txt raw.txt
```