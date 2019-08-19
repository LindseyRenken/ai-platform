# Recommendation System

## Summary

This basic collaborative filtering model is used to recommend items to users based on their history with similar items. Instead of only considering ratings for items (for example on a scale from 0 to 5), this system takes into account any positive integer value regardless of whether it's on a bound scale.

A data normalization method is used to scale the integer variables linearly between 0 and 1 depending on the value and how close it is to the max or min value for the set, and a pearson correlation is used to calculate the similarity between items.

## Inputs

The inputs of this function are as follows:

- dataset_file_name: the path or name of the file containing the data is in the format "user item variable" separated by tabs with each entry on a new line. The file sould be located within the .data/ directory in the root folder of this project and should be zipped with a ".zip" extension. The 'user' and 'item' column fields can be an id or another type of unique string. The variable column fields are positive integers.
- recom_n: the number of items to recommend to each user, such as 5 or 10.

## Outputs

- recomendations_by_user.csv written to the 'output/{formatted datetime}/' folder in the root of this project with csv format and includes a header.

## Examples

> mlflow run . -P dataset_file_name='subsampled_triplets.txt.zip' -P recom_n='10'

> mlflow run . -P dataset_file_name='train_triplets.txt.zip' -P recom_n='3'

## Future Work

- Add option to force variable integer data to 1 or 0, instead of keeping it on an unbound scale, and add an option to set cutoff for casting data to 1 or 0 (e.g. more than 2 plays of a song is a 1 and anything less is a 0).
- Experiment with other model algorithms, like FunkSVD model
- Add content filtering model by comparing word embeddings of text tags and average the score with the results of the collaborative filtering model.
- Make the personalization score calculation more efficient
- Launch on Spark
- Add error handling and filetype handling
- Optionally output to json
- Add more evaluation and exploritory metrics and save to output
- Optionally load a pre-trained model and find recommendations
- Optionally pass in an array of users to receive recommendations for
