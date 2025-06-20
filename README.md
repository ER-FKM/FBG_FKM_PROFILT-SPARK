# Fuzzy Blocking for Entity Resolution (on Apache Spark)

## Project Description
In the entity resolution field, in the blocking step, grouping similar records within the same block considerably reduces the number of comparisons and, subsequently, the required running time. However, due to errors in the dataset, such as typographical mistakes or missing values, true duplicate records may be assigned to different blocks, and/or non-matching records may be erroneously grouped within the same block. This uncertainty in the blocking step is also a result of the strict/binary assignment, where a record is assigned to a block with complete certainty. This approach lacks precision and leads to ambiguity, as a record may realistically belong to multiple candidate blocks.
To address this challenge, this method introduces a soft assignment method to overcome the uncertainty and ambiguity associated with strict/binary blocking. The soft assignment aims to mitigate this problem by assigning each record to multiple possible blocks, along with a probability of belonging. This method offers greater flexibility and precision, balancing the trade off between efficiency and effectiveness. It consists of two steps:
•	Fuzzy Block Generator (FBG): Constructs efficient overlapping blocks.
•	Probabilistic Filtering (ProbFilt): Filters out unnecessary comparisons.

## Installation & Usage
### Prerequisites
- Docker Engine
- Docker Compose
- Apache Spark 3.5.3 (included in the Docker container)
- A dataset named 'dataset.csv' placed in the root directory
- dataset ('matched_pairs.csv') in the root directory.

### Setup Instructions
1. Clone the Repository:
    git clone https://github.com/ER-FKM/FBG_FKM_PROFILT-SPARK
    cd fuzzy-kmodes-spark
2. Add the Dataset:
    Place your dataset ('dataset.csv') in the root directory.
3. Build and Start the Spark Cluster:
    docker compose up --build
4. Access the Spark Master UI (optional): http://localhost:8080
5. Access the Master Container:
    docker exec -it spark-master /bin/bash
6. Run the Application:
    spark-submit --master spark://spark-master:7077 /opt/bitnami/spark/app/ER_FKM_JAR_JAR.py


## Input 
Input: dataset.csv (your data file), required schema:
    id,name,attribute1,attribute2,...
	matched_pairs.csv



## Contributing
Contributions are welcome! If you’d like to help:
- Open an Issue
- Create a Pull Request
- Discuss ideas in the Discussions tab

## Acknowledgements
This work is built upon:
- Apache Spark
- An Adapted and Distributed version of the Fuzzy K‑Modes clustering algorithm


