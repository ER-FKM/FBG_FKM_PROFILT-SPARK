FROM bitnami/spark:latest

#RUN apt-get update
#RUN apt-get clean
# Switch to root user if not already
USER root

# Create the necessary directory and install vim
RUN apt-get update \
    && apt-get install -y vim \
    && apt-get clean
RUN apt update \
    && apt install -y net-tools

RUN pip3 install pandas
RUN pip3 install Levenshtein
RUN pip3 install jaro-winkler
RUN pip3 install fuzzywuzzy
RUN pip3 install jellyfish
RUN pip3 install scipy

COPY matched_pairs.csv /opt/bitnami/spark/app/matched_pairs.csv
COPY ER_FKM_JAR_JAR.py /opt/bitnami/spark/app/ER_FKM_JAR_JAR.py
COPY dataset.csv /opt/bitnami/spark/app/dataset.csv
COPY matched_pairs.csv /opt/bitnami/spark/app/matched_pairs.csv


