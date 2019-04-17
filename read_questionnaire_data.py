import datetime
import pandas as pd

### questions organizer dictionary
questions = {'conj': {'Q6': [0, 1],
                      'Q8': [2, 3],
                      'Q12': [0, 3],
                      'Q16': [1, 2]},
             'disj':{'Q10': [0, 2],
                     'Q18': [1, 3]},
             'trap': {'Q14': 2},
             'Gender': 'Q3',
             'Age' : 'Q4',
             'Education' : 'Q5'}

questions_options = {
    'Q6' : {'qubits': {'0': 0,
                       '1': 5,
                       '01': 6}}
}

### todo: add which options in each questions are the important ones represent each of the qubits.


### load the file
raw_df = pd.read_csv('Emma_and_Liz_april2019_no_slider_short.csv')
clms = raw_df.columns
raw_df = raw_df.iloc[2:]

### clear users that fail the trap question
raw_df = raw_df[raw_df[questions['trap'].keys()[0]].astype('float') == questions['trap'].values()[0]]

### order of the questions
### choose the columns of the order
rand_qs = ['Q10', 'Q12', 'Q14', 'Q16', 'Q18']
rand_qs = [x + '_order' for x in rand_qs]
order_cls = raw_df[clms[clms.str.contains('FL_')]]
renaming_dict = dict(zip(order_cls, rand_qs))
raw_df.rename(columns=(renaming_dict), inplace=True)

### remove all the order of the options inside each question
clms = raw_df.columns
clms = clms[~clms.str.contains('_DO_')]
raw_df = raw_df[clms]

# ### calculate all the aprameters and progpogate psi for the first 2 questions
# all_data, _ = calc_first_2_questions(raw_df)

cnames = []

### questions with fallacies
fallacy_qs = list(questions['conj'].keys()) + list(questions['disj'].keys())
all_cls = fallacy_qs + list(questions['trap'].keys()) + [questions['Gender']]+ [questions['Age']]+ [questions['Education']] + ['survey_code']

### subsample the columns that i need
for q in all_cls:
    cnames = cnames + list(clms[clms.str.contains(q)])

raw_df = raw_df[cnames]
clms = raw_df.columns

### change options numbering
for q in fallacy_qs:
    cc = clms[clms.str.contains(q)]
    cc = cc[~cc.str.contains('order')]

    a = cc.str.split('_', expand=True) ### current numbering
    list(a.levels[1]).sort()
    d = {}
    for i, j in enumerate(a.levels[1]):
        d[q+'_'+j] = q + '_' + str(i) ### new order

    raw_df.rename(columns=(d), inplace=True)

clms = raw_df.columns

### match option with which qubit and probability it is
for q in fallacy_qs:
    pass
print()