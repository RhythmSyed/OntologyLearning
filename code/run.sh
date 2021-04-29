#!/bin/bash
## Start Topic Name
startTopic="Star Wars"
corpus="starwars"
taxonName="our-l3-0.25"
taxonomyPath="./data/starwars/our-l3-0.25"
## Number of Topics for Dataset
numOfTopics=100
## Dataset Filepath
dataset_path="./star_wars.txt"
## Cleaned Dataset Filepath
cleaned_dataset_path="./star_wars_cleaned.txt"
## Embedding Filepath
embed_filepath="./embeddings.txt"
## Ontology Filepath
ontology_path="./ontology.txt"
echo 'Starting Ontology Constructor'
echo 'Running Dataset Creation'
python dataset_creation.py "$startTopic" $numOfTopics
echo 'Running Dataset Preprocessing'
python preprocessing.py "$dataset_path"
echo 'Training word2vec'
./taxogen/word2vec -train $cleaned_dataset_path -output $embed_filepath
mkdir ./data
mkdir ./data/$corpus
mkdir ./data/$corpus/init
mkdir ./data/$corpus/input
mkdir ./data/$corpus/raw
mkdir ./data/$corpus/$taxonName
cp $embed_filepath ./data/$corpus/init/$embed_filepath
cp $embed_filepath ./data/$corpus/input/$embed_filepath
cp $embed_filepath ./data/$corpus/raw/$embed_filepath
cp ./keywords.txt ./data/$corpus/raw/keywords.txt
cp ./keywords.txt ./data/$corpus/input/keywords.txt
cp $cleaned_dataset_path ./data/$corpus/raw/papers.txt
echo 'Starting Cluster Preprocessing'
python ./taxogen/cluster-preprocess.py "$corpus"
python ./taxogen/preprocess.py "$corpus"
cp ./keywords.txt ./data/$corpus/init/seed_keywords.txt
echo "Starting Taxogen"
python ./taxogen/main.py ""
echo "Building Ontology"
python process_taxonomy.py $taxonomyPath $cleaned_dataset_path
echo "Visualizing Ontology"
python visualize_graph.py $ontology_path