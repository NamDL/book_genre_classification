### CSCI 544 Applied Natural Language Processing Course Project

## Title:  *Book Genre Classification* 

The main goal of this project is to identify the genre(s) to which a book belongs to, given plot summaries and/or chapter extracts from the book, ie, a multi class text classification problem possibly requiring multiple labels/classes to be assigned to each document.

We limited the scope of the project to classifying books into five pre-determined classes:  
   * Humor  
   * Mystery  
   * Religion  
   * Romance  
   * Science Fiction  
 
The performance of three Machine Learning classifiers on a test data set of around 3000 books along with Naive Bayes as a baseline classifier is as follows:

| Genre/Classifier|Naive Bayes| Maximum Entropy| Random Forests| Support Vector Machines
|-----------------|-----------|----------------|---------------|------------------------|
| Humor           | 0.77      | 0.77           | 0.73          | 0.78                   |
| Mystery         | 0.82      | 0.88           | 0.85          | 0.87                   |
| Religion        | 0.89      | 0.87           | 0.84          | 0.88                   |
| Romance         | 0.71      | 0.76           | 0.71          | 0.75                   |
| Science Fiction | 0.80      | 0.81           | 0.78          | 0.82                   |

| Team Members             | Classifier |
|--------------------------|------------|
| Supriya Anand            |Support Vector Machines|
| Shobha Rani Dhalipathi   |Random Forests|
| Namratha Lakshminarayana |Maximum Entropy


