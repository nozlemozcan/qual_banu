"""
QUALIFICATION 2016
Nuriye Özlem Özcan Şimşek
2013800072
QUESTION: BANU DİRİ
SUBJECT: Text Classification with Naive Bayes
"""
import string
import sys, os
import random
import math
import shutil
import re
import collections
import traceback
import datetime
from collections import Counter
from nltk.stem.snowball import SnowballStemmer


def my_print(text):
    print(text)


def print_error(text):
    print('Error: ', text)

def read_file_lines(path, classes):
    my_print("read_file_lines")
    input_class_list = []
    read_lines = []
    file_list = [[os.path.join(dirpath, filename) for filename in filenames if filename[-4:].lower() == ".txt"] for (dirpath, dirnames, filenames) in os.walk(path)]
    file_list_len = len(file_list)
    file_counter = 0

    for d in range(1, file_list_len):
        current_class = classes[d-1]
        current_number_of_files = len(file_list[d])

        if current_number_of_files > 0:
            for f in range(current_number_of_files):
                with open(file_list[d][f], 'r', encoding='windows-1254') as current_opened_file:
                    read_lines.append(current_opened_file.readlines())
                current_opened_file.close()
                input_class_list.append(current_class)
                file_counter = file_counter+1

    return read_lines,input_class_list


def my_separate_samples(read_input_lines, stem_flag):
    my_print("my_separate_samples")

    input_splitted_list = []

    if stem_flag == '1':
        print('stemmer')
        stemmer = SnowballStemmer("english")

    for curr_file in read_input_lines:
        curr_line = ' '.join(curr_file)
        curr_line2 = curr_line.lower()
        exclude = string.punctuation
        curr_line3 = ''.join(ch for ch in curr_line2 if ch not in exclude)
        curr_sample = curr_line3.split()

        if stem_flag == '1':
            curr_sample_stemmed = []
            for w in curr_sample:
                curr_sample_stemmed.append(stemmer.stem(w))
            curr_sample = curr_sample_stemmed

        input_splitted_list.append(curr_sample)

    return input_splitted_list


def my_read_input(input_main_folder,stem_flag):
    my_print("my_read_input")

    words = []
    train_input_split_list = []
    train_input_class_list = []
    test_input_split_list = []
    test_input_class_list = []

    # read input files
    class_list = [[name for name in dirnames] for (dirpath, dirnames, filenames) in os.walk(input_main_folder)]
    class_list = class_list[0]
    [read_input_lines, input_class_list] = read_file_lines(input_main_folder, class_list)

    # separate samples
    input_splitted_list = my_separate_samples(read_input_lines, stem_flag)

    # shuffle
    input_all = list(zip(input_splitted_list, input_class_list))
    random.shuffle(input_all)
    random.shuffle(input_all)
    r_input_splitted_list,r_input_class_list = zip(*input_all)

    # separate into train-test pairs (10-fold) and find vocabulary
    N = len(r_input_splitted_list)  # all input size
    n = math.floor(N/10)  # fold size

    all_indexes = list(range(N))
    for f in range(10):  # starts from 0 upto 9
        testset_indexes = list(range(f*n,(f+1)*n))
        if f == 9:
            testset_indexes = list(range(f*n,N))
        trainset_indexes = list(set(all_indexes) - set(testset_indexes))
        trainset = [r_input_splitted_list[i] for i in trainset_indexes]
        trainset_labels = [r_input_class_list[i] for i in trainset_indexes]
        testset = [r_input_splitted_list[i] for i in testset_indexes]
        testset_labels = [r_input_class_list[i] for i in testset_indexes]
        train_input_split_list.append(trainset)
        train_input_class_list.append(trainset_labels)
        test_input_split_list.append(testset)
        test_input_class_list.append(testset_labels)

        words_train = []
        for t in trainset:
            for wt in t:
                if wt not in words_train:
                    words_train.append(wt)
        words.append(words_train)

    return train_input_split_list, train_input_class_list, test_input_split_list, test_input_class_list, words, class_list


def my_convert_to_bow(input_split_list, words):
    my_print("my_convert_to_bow")

    input_bow_list = []

    for f in range(10):
        curr_words = words[f]
        curr_sample_list = input_split_list[f]
        new_sample_list = [w for w in curr_sample_list if w in curr_words]
        curr_sample_counts = Counter(new_sample_list)
        input_bow_list.append(curr_sample_counts)

    return input_bow_list


def my_apply_NB(trainset,trainset_class_list,testset,words,classes):
    print("my_apply_NB")

    prediction_list = []
    total_word_count = len(words)
    total_class_count = len(classes)
    word_class_counts = collections.defaultdict(lambda: 0)
    class_file_counts = dict.fromkeys(classes,0)
    total_file_count = 0

    # TRAIN ------------------------------------------------------------------------
    N_train = len(trainset)
    for i in range(N_train):
        curr_train_sample = trainset[i]
        curr_class = trainset_class_list[i]

        class_file_counts[curr_class] += 1
        total_file_count += 1

        for curr_train_token in curr_train_sample:
            word_class_counts[curr_train_token, curr_class] += 1

    # find word and class frequencies in log form
    class_frequencies = {}
    word_class_frequencies = collections.defaultdict(lambda: 0)
    total_class_word_counts = {}
    default_word_class_frequency = {}
    for c in classes:
        class_frequencies[c] = round(math.log2(class_file_counts[c]/total_file_count),4)
        word_sum = 0
        for w in words:
            word_sum += word_class_counts[w, c]
        total_class_word_counts[c] = word_sum
        default_word_class_frequency[c] = round(math.log2(1/(word_sum+total_word_count)),4)

    for w in words:
        for c in classes:
            word_class_frequencies[w, c] = round(math.log2((word_class_counts[w, c] + 1) / (total_class_word_counts[c]+total_word_count)),4)

    # TEST -------------------------------------------------------------------------
    p = dict.fromkeys(classes,0)
    N_test = len(testset)
    for j in range(N_test):
        test_class_probabilities = p
        curr_test_sample = testset[j]

        for curr_test_token in curr_test_sample:
            if curr_test_token in words:
                for c2 in classes:
                    wcf = word_class_frequencies[curr_test_token, c2]
                    if wcf == 0:
                        wcf = default_word_class_frequency[c2]
                    test_class_probabilities[c2] += wcf
        for c3 in classes:
            test_class_probabilities[c3] += class_frequencies[c3]
            test_class_probabilities[c3] = round(test_class_probabilities[c3],4)

        test_max_prob = -99999999
        test_max_prob_class = ''
        for c4 in classes:
            test_c4_prob = test_class_probabilities[c4]
            if test_c4_prob > test_max_prob:
                test_max_prob = test_c4_prob
                test_max_prob_class = c4

        prediction_list.append(test_max_prob_class)

    return prediction_list


def my_evaluate_test(t,p):
    print("my_evaluate_test")
    N = len(t)

    curr_accuracy = 0
    for i in range(N):
        if t[i] == p[i]:
            curr_accuracy += 1
    curr_accuracy = round(curr_accuracy/N,4)

    curr_cm = [[0 for x in range(3)] for y in range(3)]

    class_dict = {'e':0, 'm':1, 's':2}
    for i in range(N):
        curr_cm[class_dict[t[i]]][class_dict[p[i]]] += 1

    return curr_accuracy, curr_cm


def my_calculate_statistics(accuracy_list,cm_list,output_folder):

    print("my_calculate_statistics")

    outfilename = output_folder+"/my_out_stats.txt"
    if os.path.exists(outfilename):
        os.remove(outfilename)
    output_file = open(outfilename, "a", encoding='utf-8')

    output_file.write("Classifier : NB\n")
    output_file.write("             10-fold cross-validation\n")

    N = len(accuracy_list)

    # accuracy
    overall_accuracy = round(sum(accuracy_list) / N,4)
    print(overall_accuracy)
    output_file.write("Accuracy : "+str(overall_accuracy)+'\n')

    # confusion matrix
    overall_cm = [[0 for x in range(3)] for y in range(3)]
    for i in range(N):
        for j in range(3):
            for k in range(3):
                overall_cm[j][k] += cm_list[i][j][k]
    for j2 in range(3):
        for k2 in range(3):
            overall_cm[j2][k2] = round(overall_cm[j2][k2] / N,4)
    print(overall_cm)
    output_file.write("Confusion Matrix : (Gold/Prediction)\n")
    output_file.write("  E    M    S\n")
    classes = ['E','M','S']
    for j3 in range(3):
        output_file.write(classes[j3]+" ")
        for k3 in range(3):
            output_file.write(str(overall_cm[j3][k3])+"  ")
        output_file.write("\n")

    # precision
    precision_E = my_calculate_precision('e','m','s',overall_cm)
    precision_M = my_calculate_precision('m','e','s',overall_cm)
    precision_S = my_calculate_precision('s','m','e',overall_cm)
    overall_precision = round((precision_E+precision_M+precision_S)/3, 4)

    # recall
    recall_E = my_calculate_recall('e','m','s',overall_cm)
    recall_M = my_calculate_recall('m','e','s',overall_cm)
    recall_S = my_calculate_recall('s','m','e',overall_cm)
    overall_recall = round((recall_E+recall_M+recall_S)/3,4)

    output_file.write("Overall Precision : "+str(overall_precision)+'\n')
    output_file.write("Overall Recall : "+str(overall_recall)+'\n')
    output_file.write("Precision E : "+str(precision_E)+'\n')
    output_file.write("Precision M : "+str(precision_M)+'\n')
    output_file.write("Precision S : "+str(precision_S)+'\n')
    output_file.write("Recall E : "+str(recall_E)+'\n')
    output_file.write("Recall M : "+str(recall_M)+'\n')
    output_file.write("Recall S : "+str(recall_S)+'\n')

    output_file.close()


def my_calculate_recall(curr_class,class1,class2,cm):
    recall=0
    class_dict = {'e':0, 'm':1, 's':2}

    a = cm[class_dict[curr_class]][class_dict[curr_class]]
    if a == 0:
        recall=0
    else:
        recall = round(a / (a + cm[class_dict[curr_class]][class_dict[class1]] + cm[class_dict[curr_class]][class_dict[class2]]),4)

    return recall

def my_calculate_precision(curr_class,class1,class2,cm):

    precision=0
    class_dict = {'e':0, 'm':1, 's':2}

    a = cm[class_dict[curr_class]][class_dict[curr_class]]
    if a == 0:
        precision=0
    else:
        precision = round(a / (a + cm[class_dict[class1]][class_dict[curr_class]] + cm[class_dict[class2]][class_dict[curr_class]]),4)

    return precision


# Define a main() function that manages requests
def main():

    try:
        start_time = datetime.datetime.now()
        main_exception_message = 'HOHOHO'

        stem_flag = sys.argv[1]
        featureset_type = sys.argv[2]  # 1:bow 2:tf-idf ??
        input_main_folder = sys.argv[3]
        output_folder = sys.argv[4]

        # parse input file and break into 10 folds train - test pairs
        [train_input_split_list, train_input_class_list,test_input_split_list, test_input_class_list, words, classes] = my_read_input(input_main_folder,stem_flag)
        # convert word lists to featureset
        if featureset_type == '1':
            train_input_bow_list = my_convert_to_bow(train_input_split_list, words)
            test_input_bow_list = my_convert_to_bow(test_input_split_list, words)


        # Naive Bayes
        cm_list = []
        accuracy_list = []
        for f in range(10):
            curr_trainset = train_input_split_list[f]
            curr_trainset_class_list = train_input_class_list[f]
            curr_testset = test_input_split_list[f]
            curr_testset_class_list = test_input_class_list[f]
            curr_words = words[f]

            curr_prediction_list = my_apply_NB(curr_trainset,curr_trainset_class_list,curr_testset,curr_words,classes)
            [curr_accuracy, curr_cm] = my_evaluate_test(curr_testset_class_list,curr_prediction_list)
            accuracy_list.append(curr_accuracy)
            cm_list.append(curr_cm)

        my_calculate_statistics(accuracy_list,cm_list,output_folder)

        my_print(main_exception_message)
        finish_time = datetime.datetime.now()
        time_difference = finish_time - start_time
        my_print("time: " + str(time_difference))
    except Exception:
        print("Exception in user code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)
        print(traceback._cause_message, file=sys.stderr)
        sys.exit(1)

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()