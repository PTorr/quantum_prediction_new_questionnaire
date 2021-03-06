from hamiltonian_prediction import *
import datetime
import pandas as pd
import numpy as np

### questions organizer dictionary
questions = {'conj': {'Q6': [0, 1],
                      'Q8': [2, 3],
                      'Q12': [0, 3],
                      'Q16': [1, 2]},
             'disj':{'Q10': [0, 2],
                     'Q18': [1, 3]},
             'trap': {'Q14': 1},
             'Gender': 'Q3',
             'Age' : 'Q4',
             'Education' : 'Q5'}

questions_dal = {'Q10': 2,
                 'Q18': 2,
                 'Q12': 1,
                 'Q16': 1}

### which options correspond to which qubits
questions_options = {
    'Q6' : {'0': 0,
            '1': 5,
            '01': 6},
    'Q8': {'2': 0,
           '3': 2,
           '23': 3},
    'Q10' : {'0': 1,
            '2': 0,
            '02': 4},
    'Q12' : {'0': 3,
            '3': 0,
            '03': 5},
    'Q16' : {'1': 2,
            '2': 0,
            '12': 4},
    'Q18' : {'1': 0,
            '3': 2,
            '13': 5}
}

def reformat_data_from_qualtrics(path):
    '''reformat the data from qualtrics to cleaner dataframe'''
    ### load the file
    raw_df = pd.read_csv(path)
    clms = raw_df.columns
    raw_df = raw_df.iloc[2:]

    ### clear users that fail the trap question
    ###cchange the range from qualtrics to [0,6]
    vd = dict(zip(np.sort(raw_df[list(questions['trap'].keys())[0]].astype('float').unique()), np.arange(6)))
    raw_df[list(questions['trap'].keys())[0]] = raw_df[list(questions['trap'].keys())[0]].astype('float').replace(vd)
    raw_df = raw_df[raw_df[list(questions['trap'].keys())[0]].astype('float') == list(questions['trap'].values())[0]]

    cnames = []

    ### questions with fallacies
    fallacy_qs = list(questions['conj'].keys()) + list(questions['disj'].keys())
    id_qs = list(questions['trap'].keys()) + [questions['Gender']]+ [questions['Age']]+ [questions['Education']] + ['survey_code']
    all_cls = fallacy_qs + id_qs

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

    ### match option with which qubit and probability it is
    q_dict = {}
    probs = ['pa','pb','pab']
    for q in fallacy_qs:
        for i, (qubit, option) in enumerate(questions_options[q].items()):
            current_name = q + '_' + str(option)
            new_name = q + '_' + 'q' + str(qubit)+ '_' + probs[i] + '_'
            q_dict[current_name] = new_name

    raw_df.rename(columns=(q_dict), inplace=True)

    raw_df = raw_df[list(q_dict.values()) + id_qs + list(raw_df.columns[raw_df.columns.str.contains('order')])]

    # raw_df[list(q_dict.values())] = raw_df[list(q_dict.values())].astype('float') / 100 ### todo: uncomment for real data
    # raw_df[list(q_dict.values())] = np.random.random(raw_df[list(q_dict.values())].shape)

    ### which question was third
    ### order of the questions
    ### choose the columns of the order
    rand_qs = ['Q10', 'Q12', 'Q16', 'Q18']
    raw_df['q3'] = ''
    for q in rand_qs:
        raw_df.loc[raw_df[raw_df.columns[raw_df.columns.str.contains(q)]].any(axis=1), 'q3'] = q

    raw_df.to_csv('data/clear_df_one_question.csv', index = False)

    return raw_df


def calc_first_2_questions(df):
    ### calculate all the parameters and psi for the first 2 questions

    all_data = {}
    for ui, u_id in enumerate(df['survey_code'].unique()):
        # go over questions 1 & 2

        ### init psi
        psi_0 = uniform_psi(n_qubits=4)
        sub_data = {
            'h_q': {}
        }
        d0 = df[(df['survey_code'] == u_id)]
        a = d0[d0.columns[d0.columns.str.contains('order')]].reset_index(drop=True) ### which columns were randomized --> to take only the one that was second
        for p_id, q in enumerate(list(questions['conj'].keys())[:2] + [a.idxmin(axis = 1)[0].split('_')[0]]):
            print(u_id, p_id, q)
            d = d0.copy()
            ### take the real probs of the user
            d = d[d.columns[d.columns.str.contains(q)]].reset_index(drop=True)
            p_real = {
                'A': d[d.columns[d.columns.str.contains('pa_')]].values,
                'B': d[d.columns[d.columns.str.contains('pb_')]].values,
                'A_B': d[d.columns[d.columns.str.contains('pab_')]].values
            }

            ### is the third question is conj/ disj
            all_q, fal = q_qubits_fal(q)

            sub_data[p_id] = get_question_H(psi_0, all_q, p_real,with_mixing = True, h_mix_type=0, fallacy_type=fal)

            psi_0 = sub_data[p_id]['psi']

            sub_data['h_q'][str(all_q[0])] = sub_data[p_id]['h_a']
            sub_data['h_q'][str(all_q[1])] = sub_data[p_id]['h_b']
            sub_data['h_q'][str(all_q[0])+str(all_q[1])] = sub_data[p_id]['h_ab']

        all_data[u_id] = sub_data
        t1 = time.time()

    ### save dict
    # all_data = pd.DataFrame(all_data)
    # all_data.to_csv('data/all_data_dict.csv', index = False)

    np.save('data/all_data_dict.npy', all_data)

    return all_data

def q_qubits_fal(q):
    if q in list(questions['conj'].keys()):
        all_q = questions['conj'][q]
        fal = 'C'
    elif q in list(questions['disj'].keys()):
        all_q = questions['disj'][q]
        fal = 'D'
    return all_q, fal

def main():
    ### running the script
    print('======= Started running at: %s =======' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    reformat, calc_first2  = True, False
    # reformat, calc_first2  = False, True
    # reformat, calc_first2  = False, False
    ### First reformat the data from qualtrics
    if reformat:
        raw_df = reformat_data_from_qualtrics('data/Emma_and_Liz_april2019_no_slider_short_one_question.csv')
    else:
        raw_df = pd.read_csv('data/clear_df_one_question.csv')
        ### calc the data of the first 2 questions
        if calc_first2:
            all_data = calc_first_2_questions(raw_df)
        else:
            all_data = np.load('data/all_data_dict.npy').item()

    print('======= Finished running at: %s =======' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':
    main()
