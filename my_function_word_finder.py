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
import itertools
import operator

# word frequency in corpus
def read_frequent_words(path):

    print("read_file_lines")

    read_lines = []
    file_list = [[os.path.join(dirpath, filename) for filename in filenames if filename[-4:].lower() == ".txt"] for (dirpath, dirnames, filenames) in os.walk(path)]
    file_list_len = len(file_list)

    for d in range(1, file_list_len):
        current_number_of_files = len(file_list[d])

        if current_number_of_files > 0:
            for f in range(current_number_of_files):
                with open(file_list[d][f], 'r', encoding='windows-1254') as current_opened_file:
                    read_lines.append(current_opened_file.readlines())
                current_opened_file.close()


    merged_lines = list(itertools.chain.from_iterable(read_lines))
    merged_lines2 = " ".join(merged_lines)
    merged_lines3 = merged_lines2.lower()
    exclude = string.punctuation
    merged_lines4 = ''.join(ch for ch in merged_lines3 if ch not in exclude)
    merged_lines5 = merged_lines4.split()

    print(len(merged_lines5))
    merged_line_counts = Counter(merged_lines5)
    frequent_words = [a for (a,b) in merged_line_counts.most_common(500)]

    return frequent_words

# idf in corpus
def read_frequent_words_idf(path):

    print("read_file_lines")

    read_lines = []
    file_list = [[os.path.join(dirpath, filename) for filename in filenames if filename[-4:].lower() == ".txt"] for (dirpath, dirnames, filenames) in os.walk(path)]
    file_list_len = len(file_list)

    for d in range(1, file_list_len):
        current_number_of_files = len(file_list[d])

        if current_number_of_files > 0:
            for f in range(current_number_of_files):
                with open(file_list[d][f], 'r', encoding='windows-1254') as current_opened_file:
                    read_lines.append(current_opened_file.readlines())
                current_opened_file.close()

    merged_lines = list(itertools.chain.from_iterable(read_lines))
    merged_lines2 = " ".join(merged_lines)
    merged_lines3 = merged_lines2.lower()
    exclude = string.punctuation
    merged_lines4 = ''.join(ch for ch in merged_lines3 if ch not in exclude)
    merged_lines5 = merged_lines4.split()
    all_words = list(set(merged_lines5))

    read_lines2 = []
    for d in read_lines:
        curr_d=" ".join(d).lower()
        exclude = string.punctuation
        curr_d = ''.join(ch for ch in curr_d if ch not in exclude)
        curr_d = curr_d.replace("\n","")
        read_lines2.append(curr_d)

    df = []
    for w in all_words:
        curr_df=0
        for d in read_lines2:
            curr_df += d.count(w)
        df.append(curr_df)

    N = len(read_lines2)
    idf = [math.log2(N/a) for a in df]


    idf_list = list(zip(all_words,idf))
    idf_list.sort(key=operator.itemgetter(1),reverse=True)

    frequent_words = [a for (a,b) in idf_list[:500]]
    return frequent_words


# Define a main() function that manages requests
def main():

    try:
        start_time = datetime.datetime.now()
        main_exception_message = 'HOHOHO'

        input_main_folder = sys.argv[1]
        out_frequent_words = read_frequent_words_idf(input_main_folder)

        output_folder = sys.argv[2]
        outfilename = output_folder+"/my_out_frequent_words.txt"
        if os.path.exists(outfilename):
            os.remove(outfilename)
        output_file = open(outfilename, "a", encoding='utf-8')

        for f in out_frequent_words:
            output_file.write(f+"\n")

        output_file.close()

        print(main_exception_message)
        finish_time = datetime.datetime.now()
        time_difference = finish_time - start_time
        print("time: " + str(time_difference))
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