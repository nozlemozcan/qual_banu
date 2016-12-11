# QUALIFICATION EXAM
## December 2016
## Question by Banu Diri
## Text Classification with Naive Bayes

### how to use system

The system is implemented with python.<br />
The parameters for the system are:<br />

1. stem_flag<br />
   values: <br />
           0 for no stemming<br />
           1 for do stemming<br />

2. featureset_type<br />
   values: <br />
           1 for bow features<br />
           2 for bow features + function word features<br />
   
3. input_main_folder<br />
   The main folder for the dataset<br />
   
4. output_folder<br />
   The path for output folder that is used for output statistics file<br />

5. fw_filename<br />
   The full path and filename for txt function word list input file<br />


### files and folders

1. my_naive_bayes.py<br />
   The system is implemented with python. <br />
   Python version is 3.5.1.<br />

2. my_function_word_finder.py<br />
   This python script is used for finding the list of function words.<br />
   Python version is 3.5.1.<br />

3. dataset_details.xlsx and dataset<br />
   Dataset files<br />
   The dataset folder is the input for the system.<br />
   There is a main folder for the dataset and it includes three sub-folders, one for each class. The three class folders are labeled with class labels. These folders contain samples in separate text files.<br />
   
4. my_out_frequent_words.txt<br />
   txt file for function word list. It is the input for the system.<br />

5. sample_outputs<br />
   Output files that are used for experiment results in the report.<br />
   
6. example_calls.txt<br />
   The call commands for the system for the experiments presented in the report.<br />
