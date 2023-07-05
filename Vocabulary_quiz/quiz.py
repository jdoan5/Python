# The purpose of this program is run and test user to select the right answer
# The topic is the Independence Day

import random

def get_def_and_pop(word_list, word_dictionary):
    random_index = random.randrange(len(word_list))
    word = word_list.pop(random_index)
    definition = word_dictionary.get(word)
    return word, definition

def get_word_and_definition(R):
    word, definition = R.split(',', 1)
    return word, definition
fh = open(r'C:\Users\john_\Documents\GitHub\Python\Vocabulary_quiz\Vocabulary.csv')
wd_list = fh.readlines()

# Removed duplicate voculary words
wd_list.pop(0)
wd_set = set(wd_list)
fh = open(r'C:\Users\john_\Documents\GitHub\Python\Vocabulary_quiz\Vocabulary_updated.csv', "w")
fh.writelines(wd_set)

word_dictionary = dict()
for R in wd_set:
    word, definition =  get_word_and_definition(R)
    word_dictionary[word] = definition
    #print (word)

while True:
    wd_list = list(word_dictionary)
    choice_list = []
    for x in range(4):
        word, definition = get_def_and_pop(wd_list, word_dictionary)
        choice_list.append(definition)
    random.shuffle(choice_list)

    print("Independence Day Vocabulary Words")
    print(word)
    print("*********************************")
    for idx, choice in enumerate(choice_list):
        print(idx+1, choice)
    choice = int(input('Select your answer or enter 0 to exit: '))
    if choice_list[choice - 1] == definition:
        print("Correct answer!\n")
    elif choice == 0:
        exit(0)
    else:
        print("Incorrect answer!")


    
