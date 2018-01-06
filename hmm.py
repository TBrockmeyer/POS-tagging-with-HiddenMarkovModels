# -*- coding: utf-8 -*-

import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import math

def create_model_file (text_file, model_name):
    # Method for creating a list of emission-state pairs
    def txtfile_to_emission_state_list (path):
        txtfile = open(path, 'r')
        emission_state_list = [line.split(' ') for line in txtfile.readlines()]
        return emission_state_list

    # Method for creating succeeding state pairs (T_i|T_(i-1)) from a given set of emission-state pairs
    # Single elements of the set are interpreted as follows:
    #   they mark the end of a sequence of the source emission-state pairs
    #   they are taken as delimiters before a new sequence of the source emission-state pairs begins
    def create_pairs_state_state (emission_state_list):
        state_state_list = []
        for i in range (0, len(emission_state_list)-1):
            # Only if these two conditions are fulfilled:
            #   the element in question itself is an emission-state pair
            #   after the element in question there's an emission-state pair (as opposite to a single element),
            # then create a state-state tuple
            if (len(emission_state_list[i+1]) == 2):
                state_1 = emission_state_list[i][0]
                if (len(emission_state_list[i]) == 2):
                    state_1 = emission_state_list[i][1]
                elif (len(emission_state_list[i]) == 1):
                    state_1 = emission_state_list[i][0]
                state_2 = emission_state_list[i+1][1]
                pair_state_state_tmp = [state_1, state_2]
                state_state_list.append(pair_state_state_tmp)
        return state_state_list

    # Method for deleting the '\n' substring from every sublist element in a list of sublists
    def delete_linebreak_in_sublists (list_of_sublists):
        try:
            for i in range (0, len(list_of_sublists)):
                for j in range (0, len(list_of_sublists[i])):
                    for k in range (0, len(list_of_sublists[i][j])):
                        # replace '\n' by ''
                        s_tmp = list_of_sublists[i][j].replace('\n', '')
                        list_of_sublists[i][j] = s_tmp
        except TypeError:
            print ("Error in <def delete_linebreak_in_sublists>: The input element seems to be invalid. Needs to be a two-dimensional list of sublists, e.g. [['abc','def\n'],['ghi','jkl']]")
        except AttributeError:
            print ("Error in <def delete_linebreak_in_sublists>: The input element seems to be invalid. Needs to be a two-dimensional list of sublists, e.g. [['abc','def\n'],['ghi','jkl']]")
        return list_of_sublists

    # Method for replacing the first element of a tag tuple with '<s' for sentence begin if it is equal to ''
    def mark_first_space_as_begin (tuple_list):
        for i in range (0, len(tuple_list)):
            if (tuple_list[i][0] == ''):
                tuple_list[i][0] = '<s'
        return tuple_list

    # Method for creating a 2D-list as a base for calculating emission and transition probabilities:
    #   "sentence start" and all possible tags as rows and columns
    #   all tag-transitions will be counted here
    def create_2D_list_from_tuple_list (tuple_list, tag_list_1, tag_list_2):
        # Use tag sets given to this function as input and dimension indicators
        w, h = len(tag_list_2), len(tag_list_1)
        # Create a Matrix for counting transitions between all tags:
        tag_tag_list = [[0 for x in range(w)] for y in range(h)]
        # Create an empty list for counting all occurences of the first tuple elements
        # (we will need these counts for calculating P(T_i|T_(i-1)) later)
        tag_list_1_counts = [0 for x in range(h)]
        tag_list_2_counts = [0 for z in range(w)]
        # Go through the whole input list and
        #   increase value at position of tag tuple_list[i][0], tuple_list[i][1]
        #   catch error if list has not "list-of-tuple-sublists" format
        #   catch StopIteration error if element not found, as described in link
        try:
            for i in range (0, len(tuple_list)):
                # If currently analyzed tuple consists of two elements
                # (because we do not want to count any single elements,
                # which do not provide a base for emission or transition probabilities)
                if (len(tuple_list[i]) == 2):
                    # Find index of first and second tag
                    first_tag_index = tag_list_1.index(tuple_list[i][0])
                    second_tag_index = tag_list_2.index(tuple_list[i][1])
                    # Add count to corresponding entry in tag_tag_list
                    tag_tag_list[first_tag_index][second_tag_index] += 1
                    # Add count to corresponding entry in tag_list_1_counts
                    tag_list_1_counts[first_tag_index] += 1
                    tag_list_2_counts[second_tag_index] += 1
        # An error exception is thrown if an element looked for cannot be found. For-loop continues then with "pass"
        except StopIteration:
            print ("Error in <create_dict_list_from_tuple_list>: Searched element not found in list")
            pass
        # Error exceptions are thrown if given type is not a valid two-dimensional element
        except TypeError:
            print ("Error in <create_dict_list_from_tuple_list>: The input element seems to be invalid. Needs to be a two-dimensional list of sublists, e.g. [['abc','def\n'],['ghi','jkl']]")
        except AttributeError:
            print ("Error in <create_dict_list_from_tuple_list>: The input element seems to be invalid. Needs to be a two-dimensional list of sublists, e.g. [['abc','def\n'],['ghi','jkl']]")
        # Package tag_tag_list and tag_list_1_counts into one output list
        output_list_tags = [tag_tag_list, tag_list_1_counts, tag_list_2_counts]
        return output_list_tags

    # Method for creating a 2D-list ... WITH Add-One Smoothing
    def create_2D_list_from_tuple_list_w_AOS (tuple_list, tag_list_1, tag_list_2):
        # Use tag sets given to this function as input and dimension indicators
        w, h = len(tag_list_2), len(tag_list_1)
        # Create a Matrix for counting transitions between all tags:
        # (Add-One smoothing already considered, by creating a matrix of ones)
        tag_tag_list = [[1 for x in range(w)] for y in range(h)]
        # Create an empty list for counting all occurences of the first tuple elements
        # (we will need these counts for calculating P(T_i|T_(i-1)) later)
        tag_list_1_counts = [len(tag_list_1) for x in range(h)]
        tag_list_2_counts = [len(tag_list_2) for z in range(w)]
        # Go through the whole input list and
        #   increase value at position of tag tuple_list[i][0], tuple_list[i][1]
        #   catch error if list has not "list-of-tuple-sublists" format
        #   catch StopIteration error if element not found, as described in link
        try:
            for i in range (0, len(tuple_list)):
                # If currently analyzed tuple consists of two elements
                # (because we do not want to count any single elements,
                # which do not provide a base for emission or transition probabilities)
                if (len(tuple_list[i]) == 2):
                    # Find index of first and second tag
                    first_tag_index = tag_list_1.index(tuple_list[i][0])
                    second_tag_index = tag_list_2.index(tuple_list[i][1])
                    # Add count to corresponding entry in tag_tag_list
                    tag_tag_list[first_tag_index][second_tag_index] += 1
                    # Add count to corresponding entry in tag_list_1_counts
                    tag_list_1_counts[first_tag_index] += 1
                    tag_list_2_counts[second_tag_index] += 1
        # An error exception is thrown if an element looked for cannot be found. For-loop continues then with "pass"
        except StopIteration:
            print ("Error in <create_dict_list_from_tuple_list>: Searched element not found in list")
            pass
        # Error exceptions are thrown if given type is not a valid two-dimensional element
        except TypeError:
            print ("Error in <create_dict_list_from_tuple_list>: The input element seems to be invalid. Needs to be a two-dimensional list of sublists, e.g. [['abc','def\n'],['ghi','jkl']]")
        except AttributeError:
            print ("Error in <create_dict_list_from_tuple_list>: The input element seems to be invalid. Needs to be a two-dimensional list of sublists, e.g. [['abc','def\n'],['ghi','jkl']]")
        # Package tag_tag_list and tag_list_1_counts into one output list
        output_list_tags = [tag_tag_list, tag_list_1_counts, tag_list_2_counts]
        return output_list_tags

    # Method for adding 1 to every sublist element in a list of lists
    def add_one_to_every_sublist_element (list_of_lists):
        # Determine whether list_of_lists is really a list of lists:
        lol_indicator = any(isinstance(el, list) for el in list_of_lists)
        if (lol_indicator == True):
            w, h = len(list_of_lists[0]), len(list_of_lists)
            # Add-One smoothing (adding 1 to every sublist element)
            for d in range (0, h):
                for e in range (0, w):
                    list_of_lists[d][e] = list_of_lists[d][e] + 1
        else:
            w = len(list_of_lists)
            for e in range (0, w):
                list_of_lists[e] = list_of_lists[e] + 1
        return list_of_lists

    # Method for creating a list of emissions or states tags from a given list of tuples
    # The list will be a one-each representation of all elements occurring
    # as the first, second or both elements in the list of tuples
    # This allows the creation of headers for rows and columns of
    # matrices being the base for calculating emission and transition probabilities
    def create_tagsets_acc_to_key (tuple_list, key_1, key_2 = None):
        tag_list = []
        if key_2 is None:
            # Catch case that key_1 exceeds dimensions
            if (int(key_1) > len(tuple_list[0])):
                raise IndexError ("Error in <create_tagsets_acc_to_key>: Key cannot exceed dimensions of input list elements")
            # Create tag_list only from key_1-th element
            # Go through all elements of tuple_list
            for i in range (0, len(tuple_list)):
                # If the key_1-th element of tuple_list[i] is not in tag_list yet, add it
                if (tuple_list[i][int(key_1)] not in tag_list):
                    tag_list.append(tuple_list[i][int(key_1)])
        else:
            # If keys exceeding tuple indices are given, throw error
            if (int(key_1) > (len(tuple_list[0])-1) or int(key_2) > (len(tuple_list[0])-1)):
                raise IndexError ("Error in <create_tagsets_acc_to_key>: Key cannot exceed dimensions of input list elements")
            # Create tag_list from both key_1-th and key_2-th element
            # Go through all elements of tuple_list
            for i in range (0, len(tuple_list)):
                # If the key_1-th or key_2-th element of tuple_list[i] is not in tag_list yet, add it
                if (tuple_list[i][int(key_1)] not in tag_list):
                    tag_list.append(tuple_list[i][int(key_1)])
                if (tuple_list[i][int(key_2)] not in tag_list):
                    tag_list.append(tuple_list[i][int(key_2)])
        return tag_list

    # Method for creating a matrix of conditional probabilities
    # from a given matrix of occurences
    # The conditional part ("where_conditionals": "rows" or "columns") needs to be specified when the function is called
    # E.g., for the conditional probability P(W_i|T_i)
    # we demand a matrix of occurences "counts_matrix" as input element with
    # rows with words W_i as heads and
    # columns with tags T_i as heads and
    # occurences of words W_i, given that they are tagged with a tag T_i (and, of course, vice versa) as entries
    # We then calculate P(W_i|T_i) by using knowledge on how many occurences were recorded for each T_i
    # (this information, "summed_conditionals", has been calculated in function create_2D_list_from_tuple_list
    # and returned as element [1](sums of each row) or [2](sums of each column) of the package)
    # In the exemplary case of P(W_i|T_i), the words W_i constitute the rows, so the
    # sum of occurences for each conditional T_i is package element number [2]
    # In the exemplary case of P(T_i|T_(i-1), the tags T_(i-1) constitute the columns, so the
    # sum of occurences for each conditional T_i is package element number [1]
    # The function allows both variants

    def calculate_log_cond_probabilities (counts_matrix, where_conditionals, summed_conditionals):
        # Prepare later index allocation of summed_conditionals
        cond_in_rows = 1
        cond_in_columns = 1
        if (where_conditionals == "rows"):
            cond_in_columns = 0
        elif (where_conditionals == "columns"):
            cond_in_rows = 0
        # Create an empty matrix with equal dimensions as counts_matrix
        w, h = len(counts_matrix[0]), len(counts_matrix)
        probabilities_matrix = [[0 for x in range(w)] for y in range(h)]
        # Catch if input matrix is not 2-dimensional
        try:
            # Go through whole counts_matrix input element
            for i in range (0, len(counts_matrix)):
                for j in range (0, len(counts_matrix[i])):
                    # Write probability (count from counts_matrix divided by conditional sum) to entry
                    #   if the conditionals are in the rows, the index does not change while i doesn't change
                    #   if in the columns, it does change with every j
                    current_sums_index = i*cond_in_rows+j*cond_in_columns
                    # If summed_conditionals[current_sums_index] is == 0, calculate a marginal probability: 1/(1+h)
                    #   in case of e.g. a long list of words as emissions, this is equivalent to (1)/(1+number_of_known_emissions)
                    if (summed_conditionals[current_sums_index] == 0):
                        probability = 1/(1+h)
                    else:
                        probability = counts_matrix[i][j] / summed_conditionals[current_sums_index]
                    # If probability is == 0, calculate a marginal probability: 1/(1+h)
                    #   in case of e.g. a long list of words as emissions, this is equivalent to (1)/(1+number_of_known_emissions)
                    if (probability == 0):
                        probability = 1/(1+h)
                    log_probability = math.log(probability)
                    probabilities_matrix[i][j] = log_probability
        except TypeError:
            print ("Error in <calculate_log_cond_probabilities>: The input matrix element seems to be invalid. Needs to be a two-dimensional list of sublists, e.g. [['abc','def\n'],['ghi','jkl']]")
        except AttributeError:
            print ("Error in <calculate_log_cond_probabilities>: The input matrix element seems to be invalid. Needs to be a two-dimensional list of sublists, e.g. [['abc','def\n'],['ghi','jkl']]")
        return probabilities_matrix

    # Create a list of emission-state pairs from the 'training_pos.txt' file
    emission_state_pairs_list = txtfile_to_emission_state_list(text_file)

    # Show number of emission-state pairs (including single elements, mostly '\n', for now; we'll clean that later)
    # print (len(emission_state_pairs_list))

    # print ("")

    # Clean emission_state_pairs_list of '\n' substrings
    emission_state_pairs_list = delete_linebreak_in_sublists (emission_state_pairs_list)

    # Show a sample range of emission-state pairs
    #for a in range (0, 300):
    #    ### print ("Emission w/wo tag: ", emission_state_pairs_list[a], "; Length:", len(emission_state_pairs_list[a]))

    # print ("")

    # Create a list of state-state pairs
    state_state_pairs_list = create_pairs_state_state(emission_state_pairs_list)

    # Clean state_state_pairs_list of '\n' substrings
    state_state_pairs_list = delete_linebreak_in_sublists (state_state_pairs_list)

    # Replace the first element of a tag tuple with '<s' for sentence begin if it is equal to ''
    state_state_pairs_list = mark_first_space_as_begin (state_state_pairs_list)

    # Show an excerpt of the state_state_pairs_list
    # print (state_state_pairs_list[0:10])

    # print ("")

    # Create the base for calculating emission and transition probabilities

    # Create two 2D-lists as a base for calculating emission and transition probabilities:
    # One with "sentence start" and all possible tags as rows and columns
    #   where all tag-transitions will be counted
    # One with "sentence start" and all possible tags as columns
    # and all occurring words as rows
    #   where all words occurring with a specific tag will be counted

    # Create two 2D-lists for calculating emission and transition probabilities
    # (from the 2D-lists with the counts):
    # Sum up all values in each tag column and divide every element in this column by this value
    # One for the emission probabilites; entries will be P(W_i|T_i)
    # One for the transition probabilities; entries will be P(T_i|T_(i-1))

    ################ First preparatory measure for building the HMM model:
    # counting tag-tag combinations and then
    # calculate transition propabilities

    # Create a list of states tags
    tag_list = create_tagsets_acc_to_key(state_state_pairs_list, '0', '1')
    tag_list = sorted(tag_list, key=str.lower)

    # print ("")

    # Create and unpack output of the tag-tag transition matrix (tag_tag_list)
    # and the accumulated occurences of first element tags (tag_list_1_counts)
    # tag_list_1 was given to the function as key-list for both condition and events
    # (spoken in conditional probability terminology)
    tag_tag_package = create_2D_list_from_tuple_list_w_AOS (state_state_pairs_list, tag_list, tag_list)
    #   Matrix of tag-tag transition counts
    tag_tag_list = tag_tag_package[0]
    #   Accumulated occurences of first element (tags)
    tag_1_counts = tag_tag_package[1]
    #   Accumulated occurences of second element (tags)
    tag_2_counts = tag_tag_package[2]


    # print ("tag_1_counts", tag_1_counts)
    # print ("tag_2_counts", tag_2_counts)

    # print ("")

    # Create matrix of (conditional) transition probabilities
    # The "conditional element"
    # transition_probabilities = calculate_log_cond_probabilities (tag_tag_list, "rows", tag_1_counts)
    transition_probabilities = calculate_log_cond_probabilities (tag_tag_list, "columns", tag_2_counts)
    # print ("transition_probabilities[0:9]: ", transition_probabilities[0:9])

    # print ("")

    ################ Second preparatory measure for building the HMM model:
    # counting emission-tag combinations and then
    # calculate emission propabilities

    # Create a list of word tags
    # We order create_tagsets_acc_to_key to create the word_list
    # only from the first elements of the tuple_list. i.e. the 'real words
    word_list = create_tagsets_acc_to_key(emission_state_pairs_list, '0')
    word_list = sorted(word_list, key=str.lower)

    # print ("")

    # Create and unpack output of the word-tag emission matrix (word_tag_list)
    # and the accumulated occurences of first element tags (tag_list_1_counts)
    # tag_list_1 was given to the function as key-list for both condition and events
    # (spoken in conditional probability terminology)
    word_tag_package = create_2D_list_from_tuple_list (emission_state_pairs_list, word_list, tag_list)
    #   Matrix of word-tag emission counts
    word_tag_list = word_tag_package[0]
    #   Accumulated occurences of first element (words)
    word_counts = word_tag_package[1]
    #   Accumulated occurences of second element (tags)
    tag_word_counts = word_tag_package[2]

    # print ("")

    # Create matrix of (conditional) emission probabilities
    # emission_probabilities = calculate_log_cond_probabilities (word_tag_list, "columns", tag_word_counts)
    emission_probabilities = calculate_log_cond_probabilities (word_tag_list, "rows", word_counts)

    # print ("")

    ################ Compression of elements necessary for HMM to a compressed file "model.npz"

    # compressing code adapted from https://docs.python.org/3/library/lzma.html#examples

    # Prepare compressed file for HMM calculations:
    # Prepare adding data to compressed file:
    #   List with all valid states
    out1 = tag_list
    #   Matrix with transition probabilities
    out2 = transition_probabilities
    #   List with all valid emissions
    out3 = word_list
    #   Matrix with emission probabilities
    out4 = emission_probabilities
    #   List with number of words occurred per tag
    out5 = tag_word_counts
    # Compress data
    np.savez_compressed(model_name, out1, out2, out3, out4, out5)

    # Prepare compressed file for checking counts and how probabilities may have been calculated:
    # Prepare adding data to compressed file:
    out21 = out1
    out22 = out3
    out23 = word_tag_list
    out24 = tag_word_counts
    out25 = word_counts
    np.savez_compressed('compressed_file_counts.npz', out21, out22, out23, out24, out25)

def classify_sentence_with_model (model_name, sentence):
    # Define viterbi algorithm
    # code from https://gist.github.com/nkt1546789/fa238a168c2c2f84babce71f9f5d5ccd
    def viterbi(P, A):
        """
        P: log probability matrix (n_samples by n_states)
        A: log transition probability matrix (n_states by n_states)
        """
        # n_samples is the number of words
        n_samples = P.shape[0]
        # states is an array, with the ascending numbers of 1 to the number of states as elements: [1 2 3 ... n_states]
        states = np.arange(P.shape[1])
        len_states = len(states)

        V = np.zeros((n_samples, len_states))
        S = np.zeros((n_samples, len_states), dtype=int)
        V[0] = P[0] + A[3]
        # V[0] = P[0]

        ### print ("P: \n", P)
        ### print ("A: \n", A)

        for t in range(1, n_samples):
            ### print ("-----t: ", t)
            values = V[t-1] + A
            ### print ("-----values: \n", values)
            S[t-1] = np.argmax(values, axis=1)
            ### print ("-----P[t]: \n", P[t])
            V[t] = P[t] + np.max(values, axis=1)
            ### print ("-----V[t]: \n", V[t])

        y = np.zeros(n_samples, dtype=int)
        y[-1] = V[-1].argmax()
        log_prob = V[-1].max()
        ### print ("log_prob: ", log_prob)
        Vindex = V[-1].argmax()
        ### print ("Location of argmax in V[-1]: ", Vindex , "; value:", V[-1][Vindex])
        for t in range(2, n_samples+1):
            y[-t] = S[-t, y[-t+1]]
        viterbi_package = [y, log_prob]
        return viterbi_package

    # Get data from compressed file:
    model_name_npz = model_name[0]
    loaded = np.load(model_name_npz)
    #   List with all valid states
    tag_list = loaded['arr_0']
    #   Matrix with transition probabilities
    transition_probabilities = loaded['arr_1']
    #   List with all valid emissions
    word_list = loaded['arr_2']
    #   Matrix with emission probabilities
    emission_probabilities = loaded['arr_3']
    #   List with number of words occurred per tag
    tag_word_counts = loaded['arr_4']
    #   Add
    #       an element 'unknown emission' to word_list
    word_list = np.append(word_list, ['unknown_emission'])
    #       and a line with marginal probabilities to emission_probabilities
    marginal_probability = []
    for f in range (0, emission_probabilities.shape[1]):
        # Add a marginal probability representing the distribution of tags occurred throughout the training set
        curr_prob = tag_word_counts[f]/emission_probabilities.shape[0]
        marginal_probability.append(curr_prob)
    ### print (" marginal_probability: ", marginal_probability)
    #       or: a line with values (1)/(1+number_of_known_emissions) and shape[0] = number_of_states to emission_probabilities
    # marginal_probability = emission_probabilities.shape[1]*[1/(1+emission_probabilities.shape[0])]
    ### print ("emission_probabilities.shape[0] (aka. number of unique words): ", emission_probabilities.shape[0])
    ### print ("marginal_probability: ", marginal_probability)
    emission_probabilities = np.append(emission_probabilities, [marginal_probability], axis=0)

    ### print ("tag_list: \n", tag_list)

    # Create an emission_probabilities_sentence matrix that only contains the emission probs for the words analyzed
    # then conduct viterbi only on this matrix
    # for creating emission_probabilities_sentence,
    #   determine indices of all words in sentence and add relevant lines of emission_probabilities to emission_probabilities_sentence
    sentence_split = sentence.split()
    ### print ("sentence_split: ", sentence_split)
    emission_probabilities_sentence = []
    for b in range (0, len(sentence_split)):
        # determine index of current word in emission_probabilities, i.e. in word_list
        ### print ("sentence_split[", b, "]: ", sentence_split[b])
        #### print ("word_list: ", word_list[0:4])
        word_index = np.where(word_list == sentence_split[b])
        ### print ("word_index[0]: ", word_index[0])
        if (len(word_index[0]) == 0):
            word_index = np.where(word_list == 'unknown_emission')
        # If a word has more than one character, ensure that it cannot be labeled with $( $. $, tags
        # by reducing probabilities at the corresponding positions
        # in part of emission_probabilities given to shortened array
        emission_probabilities_excerpt = emission_probabilities[word_index[0][0]]
        if (len(word_list[word_index[0][0]]) > 1):
            # Determine positions of tags $( $. $, in tag_list
            bracket_index = np.where(tag_list == '$(')
            point_index = np.where(tag_list == '$.')
            komma_index = np.where(tag_list == '$,')
            senbegin_index = np.where(tag_list == '<s')
            # Set emission probabilities for these tags to very low levels (to approximately -14.05, far lower than other marginal values)
            low_prob = math.log(1/(1+100*emission_probabilities.shape[0]))
            if not (len(bracket_index[0]) == 0): emission_probabilities_excerpt[bracket_index] = low_prob
            if not (len(point_index[0]) == 0): emission_probabilities_excerpt[point_index] = low_prob
            if not (len(komma_index[0]) == 0): emission_probabilities_excerpt[komma_index] = low_prob
            if not (len(senbegin_index[0]) == 0): emission_probabilities_excerpt[senbegin_index] = low_prob
        emission_probabilities_sentence.append(emission_probabilities_excerpt)
    emission_probabilities_sentence = np.array(emission_probabilities_sentence)
    ### print ("emission_probabilities_sentence: ", emission_probabilities_sentence)

    # Conduct viterbi with given parameters
    ##### viterbi_result = 'placeholder viterbi_result'
    viterbi_package = viterbi(emission_probabilities_sentence, transition_probabilities)
    viterbi_result = viterbi_package[0]
    viterbi_prob = viterbi_package[1]
    ### print ("viterbi_result = ", viterbi_result)
    viterbi_result_tags = []
    for c in range (0, len(viterbi_result)):
        # Find tag corresponding to state number
        ### print ("viterbi_result[c]: ", type(viterbi_result[c]))
        tag_index_corresp = viterbi_result[c]
        tag_corresp_state = tag_list[tag_index_corresp]
        viterbi_result_tags.append(tag_corresp_state)

    viterbi_result_output = ' '.join(viterbi_result_tags)
    viterbi_result_package = [viterbi_result_output, viterbi_prob]
    return viterbi_result_package

################ Allow command-line interface:
# "python hmm.py –train training_pos.txt model.npz" invokes the creation of the "model.npz" file with the data necessary for HMM
# "python hmm.py –classify model.npz "Wir rennen oft zum Bus . <or any other sentence>" invokes an output of the form
#   "PPER VVFIN ADV APPRART NN $ .
#   -52.549781228206065"

# argparse, inspired by https://docs.python.org/3/library/argparse.html

# definition to check whether argument at position 2 (after the 'train'/'classify' decision is a .txt-file name
def some_string(string):
    global opt1
    opt1 = False
    global opt2
    opt2 = False
    global given_textfile
    global given_modelfile
    if (string.find('.txt') >= 0):
        opt1 = True
        # print("opt1")
        # args.cmd_string is then a .txt file
        given_textfile = string.split()
    else:
        opt2 = True
        # print("opt2")
        # args.cmd_string is then the name of a compressed model file
        given_modelfile = string.split()
# Creation of a parser to expect arguments in command line
parser = argparse.ArgumentParser(description='Invoke either model.npz creation or classification of given sentence. \nFormat: either\n<train text_file_name.txt model_name> or\n<classify existing_model_name ``sentence to be classified, in quotation marks´´>', formatter_class=RawTextHelpFormatter)
parser.add_argument('cmd_choice')
parser.add_argument('cmd_string', type=some_string)
parser.add_argument('cmd_third')
args = parser.parse_args()
# Invoke definitions according to given arguments
if(opt1==True and args.cmd_choice == 'train'):
    given_modelname = args.cmd_third
    given_textfile = given_textfile[0]
    ### print ("given_textfile", given_textfile)
    ### print ("given_modelname", given_modelname)
    # Invoke model creation
    create_model_file (given_textfile, given_modelname)
    print ("Model file ", given_modelname, " created successfully in current folder.")
elif(opt2==True and args.cmd_choice == 'classify'):
    given_sentence = args.cmd_third
    ### print ("given_modelfile", given_modelfile)
    ### print ("given_sentence", given_sentence)
    # Invoke classification definition
    viterbi_result_package = classify_sentence_with_model (given_modelfile, given_sentence)
    # Output to console: viterbi result tagsequence; and the viterbi result log probability
    print (viterbi_result_package[0])
    print (viterbi_result_package[1])
else:
    print ("Something went wrong. Please check command for formats: Either\n<train text_file_name.txt model_name> or\n<classify existing_model_name 'sentence to be analyzed'>")
# add_argument with command to be processed
"""
parser1.add_argument(
    'command',
    choices = ['train', 'classify'],
    help='A keyword determining what shall be done: train or classify.  '
         '')
args1 = parser1.parse_args()
### print ("args1.command: ", args1.command)
parser2 = argparse.ArgumentParser(description='Invoke either model.npz creation or classify given sentence')

if (args1.command == 'train'):
    parser2.add_argument(
        'text_file',
        help='A file containing the document with emissions and states to process.  '
             'Should be encoded in UTF-8')
    parser2.add_argument(
        'model_name',
        help='A name for the compressed model file to be created from the input text file.  ')
    args2 = parser2.parse_args()
    ### print ("args.text_file: ", args2.text_file)
    ### print ("args.model_name: ", args2.model_name)
    ### print ("Let's build the model")
    #create_model_file (args.text_file, args.model_name)
elif (args1.command == 'classify'):
    parser2.add_argument(
        'model_name_1',
        help='A name for the compressed model file to be created from the input text file.  ')
    parser2.add_argument(
        'sentence',
        help='A sentence of words to be analyzed.  ')
    args2 = parser2.parse_args()
    ### print ("args.model_name: ", args2.model_name)
    ### print ("args.sentence: ", args2.sentence)
    ### print ("let's classify")
else:
    ### print ("wrong command entered")
"""

"""
# add_argument with text file to be processed or model file for classification
group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument(
    'text_file',
    help='A file containing the document with emissions and states to process.  '
         'Should be encoded in UTF-8')
group1.add_argument(
    'model_name_1',
    help='A name for the compressed model file to be created from the input text file.  ')

# add_argument with model file to be processed or sentence for classification
group2 = parser.add_mutually_exclusive_group(required=True)
group2.add_argument(
    'model_name_2',
    help='A name for the compressed model file to be created from the input text file.  ')
group2.add_argument(
    'sentence',
    help='A sentence of words to be analyzed.  ')

# add_argument with text file to be processed
parser.add_argument(
    'text_file',
    default='no text file given',
    required=False,
    help='A file containing the document with emissions and states to process.  '
         'Should be encoded in UTF-8')
# add_argument with name of model to be created
parser.add_argument(
    'model_name',
    help='A name for the compressed model file to be created from the input text file.  '
         'Should be encoded in UTF-8')
# add_argument with sentence to be analyzed
parser.add_argument(
    'sentence',
    default='no sentence given',
    required=False,
    help='A sentence of words to be analyzed.  '
         'Should be encoded in UTF-8')
"""

# invoke def with model creation

"""

################ Testing

# Test: find word index of searched word and go through all tag indices
# to see where emission probability is highest
word_ranking_tags = []
searched_word = 'zu'
ind_word = word_list.index(searched_word)
for k in range (0, len(tag_list)):
    ind_tag = k
    prob_word_tag = emission_probabilities[ind_word][ind_tag]
    word_occurrences_with_tag = word_tag_list[ind_word][ind_tag]
    word_ranking_tags.append([tag_list[ind_tag], word_occurrences_with_tag, prob_word_tag])
word_ranking_tags = sorted(word_ranking_tags, key=lambda x: x[1], reverse=True)
# print ("Ranking of most probable tags for given word:\n")
for l in range (0, len(word_ranking_tags)):
    ### print ("Rank ", l, ": ", word_ranking_tags[l])
    
# print ("")
    
# Test: Determine most likely transitions between tags
# Go through entire transition_probabilities and record
# tag_1 | tag_2 | probability
tag_ranking_tags = []
for m in range (0, len(transition_probabilities)):
    for n in range (0, len(transition_probabilities[m])):
        prob_tag_tag = transition_probabilities[m][n]
        tag_occurrences_with_tag = [tag_list[m], tag_list[n], prob_tag_tag]
        tag_ranking_tags.append(tag_occurrences_with_tag)
tag_ranking_tags = sorted(tag_ranking_tags, key=lambda x: x[2], reverse=True)
# print ("Ranking of most occurring tag-tag transitions:\n")
for o in range (0, 100):
#for o in range (0, len(word_ranking_tags)):
    ### print ("Rank ", o, ": ", tag_ranking_tags[o])

### print ("")

# Test: Create a ranking of how often a word occurred given that a certain TAG is present
# Loop through all 52 types of tags, and show three most probable words
import itertools
import heapq
# Transpose emission_probabilities, so that words are in columns and underlying tags in rows
emission_probabilities_T = list(itertools.zip_longest(*emission_probabilities, fillvalue=""))
# Loop through rows (tags) and reveal first, second, third maximum
#   create empty list
tag_max_words = []
for p in range (0, len(emission_probabilities_T)):
    tag_name = tag_list[p]
    first_maxes = heapq.nlargest(3, emission_probabilities_T[p])
    maxes_with_indices = []
    for q in range (0, len(first_maxes)):
        word_currmax_index = emission_probabilities_T[p].index(first_maxes[q])
        word_currmax = word_list[word_currmax_index]
        maxes_with_indices.append([word_currmax, first_maxes[q]])
    tag_max_words_bundle = [tag_name, maxes_with_indices]
    tag_max_words.append(tag_max_words_bundle)
for r in range (0, len(tag_max_words)):
    ### print (tag_max_words[r])

"""