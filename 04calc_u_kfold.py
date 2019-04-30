import random
import numpy as np
import pandas as pd

from hamiltonian_prediction import *
from read_questionnaire_data import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import seaborn as sns
from scipy import stats

# psi_dyn = np.dot(U, psi_0)

qubits_dict = {0:'a', 1:'b', 2:'c', 3:'d'}
fal_dict = {1:'C', 2: 'D'}

def get_general_p_without_h_trial(all_q, which_prob, psi, n_qubits=4, is_normalized = False):
    '''calculate probability based on U and psi'''
    P_ = MultiProjection(which_prob, all_q, n_qubits)
    psi_final = np.dot(P_, psi)
    p_ = np.dot(np.conjugate(np.transpose(psi_final)), psi).real / np.dot(np.conjugate(np.transpose(psi_final)), psi_final).real
    return p_

def sub_sample_data(all_data, data_qn, df, users):
    '''return data'''
    for k, v in all_data.items():
        if k in users:
            p_real, d = sub_q_p(df, k, 2)
            data_qn[k] = deepcopy(v)
            data_qn[k][2] = {
                'p_a': p_real['A'],
                'p_b': p_real['B'],
                'p_ab': p_real['A_B']
            }

    return data_qn


def calculate_all_data_cross_val_kfold(use_U=True, with_mixing=True, use_neutral=False, h_mix_type=0):
    '''cross validation only for the third question'''

    ### load the dataframe containing all the data
    raw_df = pd.read_csv('data/clear_df.csv')
    raw_df.rename({'survey_code':'userID'},axis = 1, inplace=True)
    raw_df.reset_index(drop=True, inplace=True)

    ### loading all the dat of the firs 3 questions
    all_data = np.load('data/all_data_dict.npy').item()

    ### creating a dictionary with users that had the same question as the third question
    user_same_q_list = {}
    for q, g in raw_df.groupby('q3'):
        user_same_q_list[q] = g['userID']

    # third question
    ### creating a dataframe to save all the predictions error --> for specific question group by 'qn' --> agg('mean')
    prediction_errors = pd.DataFrame()

    q_info = {}
    ### Run on all users that have the same third question.
    for qn, user_list in user_same_q_list.items():
        # user_list = list(user_list)
        user_list = np.array(user_list)
        all_q, fal = q_qubits_fal(qn)
        # go over all 4 types of questions

        ### split the users to test and train using kfold - each user will be one time in test
        kf = KFold(n_splits=10)  # todo: num_of_repeats
        kf.get_n_splits(user_list)

        for i, (train_index, test_index) in enumerate(kf.split(user_list)):
            q_info[qn] = {}
            train_users, test_users = user_list[train_index], user_list[test_index]
            train_q_data_qn = {}
            test_q_data_qn = {}

            train_q_data_qn = sub_sample_data(all_data, train_q_data_qn, raw_df, train_users)
            test_q_data_qn = sub_sample_data(all_data, test_q_data_qn, raw_df, test_users)

            ### taking the mean of the probabilities of the 80 %
            p_a_80 = []
            p_b_80 = []
            p_ab_80 = []
            for u_id, tu in test_q_data_qn.items():
                p_a_80.append(tu[2]['p_a'][0])
                p_b_80.append(tu[2]['p_b'][0])
                p_ab_80.append(tu[2]['p_ab'][0])
            p_a_80 = np.array(p_a_80).mean()
            p_b_80 = np.array(p_b_80).mean()
            p_ab_80 = np.array(p_b_80).mean()

            if len(train_q_data_qn) > 0:
                ### question qubits (-1) because if the range inside of some function
                h_names = ['0', '1', '2', '3', '01', '23', str(all_q[0]) + str(all_q[1])]

                # find U for each question #
                start = time.clock()
                print('calculating U for %s on train data' % qn)

                ### set bounds to all parameters
                bounds = np.ones([10, 2])
                bounds[:, 1] = -1

                # res_temp = general_minimize(fun_to_minimize_grandH, args_=(all_q, train_q_data_qn, h_mix_type, fal),
                #                             x_0=np.zeros([10]), method='L-BFGS-B', bounds = bounds)

                res_temp = general_minimize(fun_to_minimize_grandH, args_=(all_q, train_q_data_qn, h_mix_type, fal),
                                            x_0=np.zeros([10]), method='Powell')

                end = time.clock()
                print('question %s, U optimization took %.2f s' % (qn, end - start))

                q_info[qn]['U'] = U_from_H(grandH_from_x(res_temp.x, fal))

                q_info[qn]['U_params_h'] = [res_temp.x]

                # calculate H_AB
                print('calculating h_ab')
                H_dict = {}
                for u_id, tu in test_q_data_qn.items():
                    if use_neutral:
                        psi_0 = uniform_psi(n_qubits=4)
                    else:
                        psi_0 = np.dot(q_info[qn]['U'], tu[1]['psi'])

                    p_real, d = sub_q_p(raw_df, u_id, 2)
                    sub_data_q = get_question_H(psi_0, all_q, p_real,
                                                [tu['h_q'][str(all_q[0])],
                                                 tu['h_q'][str(all_q[1])]],
                                                with_mixing, h_mix_type, fallacy_type=fal)
                    tu['h_q'][str(all_q[0]) + str(all_q[1])] = sub_data_q['h_ab']
                    # tu['h_q'][str(all_q[0])] = sub_data_q['h_a']
                    # tu['h_q'][str(all_q[1])] = sub_data_q['h_b']
                    H_dict[u_id] = []
                    for hs in h_names:
                        H_dict[u_id].append(tu['h_q'][hs])

                df_H = pd.DataFrame.from_dict(data=H_dict, orient='index')
                df_H.columns = ['A', 'B', 'C', 'D', 'AB', 'CD', 'pred']

                start = time.clock()
                mtd = 'lr'  # 'ANN'
                print('calculating h_ij' + mtd)
                est = pred_h_ij(df_H, method = mtd)
                end = time.clock()
                print('question %s, h_ij prediction took %.2f s' % (qn, end - start))

                q_info[qn]['H_ols'] = est

            ### predict on test users --> with NO {H_ab}
            print('calculating errors on test data')
            U = q_info[qn]['U']
            for u_id, tu in test_q_data_qn.items():
                temp = {}
                temp['id'] = [u_id]
                temp['qn'] = [qn]

                temp['q1'] = [all_q[0]]
                temp['q2'] = [all_q[1]]

                q1 = 'p_' + qubits_dict[temp['q1'][0]]
                q2 = 'p_' + qubits_dict[temp['q2'][0]]
                q12 = 'p_' + qubits_dict[temp['q1'][0]] + qubits_dict[temp['q2'][0]]

                temp['U'] = [use_U]

                ### psi after the 2nd question
                psi_0 = tu[1]['psi']

                ### propogate psi with the U of the 3rd question
                psi_dyn = np.dot(U, psi_0)

                ### probabilities from the 1st and 2nd question
                temp['p_a'] = [tu[0]['p_a'][0]]
                temp['p_b'] = [tu[0]['p_b'][0]]
                temp['p_c'] = [tu[1]['p_a'][0]]
                temp['p_d'] = [tu[1]['p_b'][0]]

                ### probs of the current question taken from previous questions
                temp['p_a_pre'] = temp[q1]
                temp['p_b_pre'] = temp[q2]

                ### real probabilities
                temp['p_a_real'] = [tu[2]['p_a'][0]]
                temp['p_b_real'] = [tu[2]['p_b'][0]]
                temp['p_ab_real'] = [tu[2]['p_ab'][0]]

                ### predicted probabilities with U
                # full_h = [tu['h_q'][str(int(temp['q1'][0]) - 1)], tu['h_q'][str(int(temp['q2'][0]) - 1)], None]
                h_a = [tu['h_q'][str(int(temp['q1'][0]))], None, None]
                h_b = [None, tu['h_q'][str(int(temp['q2'][0]))], None]
                temp['p_a_pred_U'] = [get_general_p(h_a, all_q, '0', psi_dyn, n_qubits=4).flatten()[0]]
                temp['p_b_pred_U'] = [get_general_p(h_b, all_q, '1', psi_dyn, n_qubits=4).flatten()[0]]

                ### predicted probabilities with I
                temp['p_a_pred_I'] = [get_general_p(h_a, all_q, '0', psi_0, n_qubits=4).flatten()[0]]
                temp['p_b_pred_I'] = [get_general_p(h_b, all_q, '1', psi_0, n_qubits=4).flatten()[0]]

                ### calculate the error from the previous probabilities with NO U.
                temp['p_a_err_real_pre'] = [np.abs(temp['p_a_real'][0] - temp['p_a_pre'][0])]
                temp['p_b_err_real_pre'] = [np.abs(temp['p_b_real'][0] - temp['p_b_pre'][0])]

                ### calculate the error from the propogated state with U
                temp['p_a_err_real_U'] = [np.abs(temp['p_a_real'][0] - temp['p_a_pred_U'][0])]
                temp['p_b_err_real_U'] = [np.abs(temp['p_b_real'][0] - temp['p_b_pred_U'][0])]

                ### calculate the error from the full 4 qubits state with I
                print('pred_I = ', temp['p_a_pred_I'][0])
                print('pred_U = ', temp['p_a_pred_U'][0])
                print('real = ', temp['p_a_real'][0])

                temp['p_a_err_real_I'] = [np.abs(temp['p_a_real'][0] - temp['p_a_pred_I'][0])]
                temp['p_b_err_real_I'] = [np.abs(temp['p_b_real'][0] - temp['p_b_pred_I'][0])]

                ### calculate the error from the mean of 80 precent
                temp['p_a_err_real_mean80'] = [np.abs(temp['p_a_real'][0] - p_a_80)]
                temp['p_b_err_real_mean80'] = [np.abs(temp['p_b_real'][0] - p_b_80)]

                ### calculate the error from uniform
                temp['p_a_err_real_uniform'] = [np.abs(temp['p_a_real'][0] - .5)]
                temp['p_b_err_real_uniform'] = [np.abs(temp['p_b_real'][0] - .5)]

                # use question H to generate h_ab
                # q_info[qn]['H_ols']
                h_names_gen = ['0', '1', '2', '3', '01', '23']
                if with_mixing:
                    all_h = {'one': []}
                    for hs in h_names_gen:
                        all_h['one'].append(tu['h_q'][hs])
                    df_H = pd.DataFrame.from_dict(data=all_h, orient='index')
                    df_H.columns = ['A', 'B', 'C', 'D', 'AB', 'CD']
                    try:
                        h_ab = q_info[qn]['H_ols'].predict(df_H).values[0]
                    except:
                        h_ab = q_info[qn]['H_ols'].predict(df_H)[0]
                else:
                    h_ab = 0.0

                full_h = [tu['h_q'][str(int(temp['q1'][0]))], tu['h_q'][str(int(temp['q2'][0]))], h_ab]
                temp['p_ab_ols']    = [get_general_p(full_h, all_q, fal, psi_dyn, n_qubits=4).flatten()[0]]
                temp['p_ab_eye'] = [get_general_p(full_h, all_q, fal, psi_0, n_qubits=4).flatten()[0]]

                ### prediction erros for p_ab
                temp['p_ab_err_u_mlr']    = [np.abs(temp['p_ab_real'][0] - temp['p_ab_ols'][0])]
                temp['p_ab_err_i'] = [np.abs(temp['p_ab_real'][0] - temp['p_ab_eye'][0])]
                temp['p_ab_err_real_m80'] = [np.abs(temp['p_ab_real'][0] - p_ab_80)]
                temp['p_ab_err_real_uni'] = [np.abs(temp['p_ab_real'][0] - .5)]

                prediction_errors = pd.concat([prediction_errors, pd.DataFrame(temp)], axis=0)

            print('end of cycle %d' % i)
            np.save('data/kfold_all_data_dict.npy', all_data)
            np.save('data/kfold_UbyQ.npy', q_info)

    prediction_errors.set_index('id', inplace=True)
    prediction_errors.to_csv('data/calc_U/kfold_prediction_errors.csv')  # index=False)


def plot_errors(df):
    '''Boxplot of the errors per question type.'''
    ### list of the columns of the errors
    err_cl = list(df.columns[df.columns.str.contains('err')])

    ### take only the columns with the errors and question number
    df1 = df[err_cl + ['qn']]

    ### melt the data frame to: qn, err_type, err_value
    df2 = pd.melt(df1, id_vars=['qn'], value_vars=err_cl, var_name='err_type', value_name='err_value')
    df2['err_value'] = pd.to_numeric(df2['err_value'].apply(lambda x: x.replace('[', '').replace(']', '')))

    ### boxplot of err to qn by err_type
    g = sns.factorplot(x="qn", hue='err_type', y="err_value", data=df2, kind="box", size=4, aspect=2)
    g.fig.suptitle('prediction error as function of which question was third\n by probability by prediction type (U, I, previous)')
    g.savefig('data/calc_U/err_boxplot_per_qn_per_prob.png')

    g1 = sns.factorplot(x="err_type", y="err_value", data=df2, kind="box", size=4, aspect=2)
    g1.set_xticklabels(rotation=45)
    g1.savefig('data/calc_U/err_boxplot_across_all_questions.png')

    df3 = df2.copy()
    df3['prob'] = 0
    df3['prob'][df3['err_type'].str.contains('p_b')] = 1 # p_a --> 0
    df3['err_type'][df3['err_type'].str.contains('_I')]      = 'I'
    df3['err_type'][df3['err_type'].str.contains('_U')]      = 'U'
    df3['err_type'][df3['err_type'].str.contains('_pre')]    = 'pre'
    df3['err_type'][df3['err_type'].str.contains('mean')]    = 'mean80'
    df3['err_type'][df3['err_type'].str.contains('uniform')] = 'uniform'
    df3['err'] = pd.Categorical(df3['err_type'], categories=df3['err_type'].unique()).codes
    df3.to_csv('data/calc_U/00pred_err_per_prediction_type.csv')#index=False)

    ### group per probability a/b.
    gg = df3.groupby(['prob'])
    ### group per probability a/b per question.
    gg1 = df3.groupby(['prob','qn'])
    ### run on all combinations of prob and q.
    for p, q in product(list(df3['prob'].unique()),list(df3['qn'].unique())):
        gg.get_group(p).to_csv('data/calc_U/00pred_err_per_prob_%d.csv' % p)
        gg1.get_group((p, q)).to_csv('data/calc_U/00pred_err_per_prob_%d_per_qn_%s.csv' %(p,q))

    g2 = sns.factorplot(x="err_type", y="err_value", data=df3, kind="box", size=4, aspect=2)
    g2.set_xticklabels(rotation=45)
    g2.savefig('data/calc_U/err_boxplot_across_all_questions_and_probs.png')

    df4 = df3.pivot(columns='err_type', values='err_value')
    df5 = pd.DataFrame()
    df5['U'] = df4['U'].dropna().reset_index(drop=True)
    df5['I'] = df4['I'].dropna().reset_index(drop=True)
    df5['pre'] = df4['pre'].dropna().reset_index(drop=True)

    df5.to_csv('data/calc_U/00cross_val_prediction_errors_per_prediction_type4anova.csv')#index=False)('data/calc_U/cross_val_prediction_errors_per_prediction_type.csv')#index=False)


def average_results(h_mix_type, use_U, use_neutral, with_mixing, num_of_repeats):
    '''Combine all the data from num_of_repeats cross validations to one dataframe'''
    control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (use_U, with_mixing, use_neutral, h_mix_type)
    df_mean = pd.DataFrame()

    ### adding one dataframe at a time.
    for i in range(num_of_repeats):
        prediction_errors = pd.read_csv('data/calc_U/cross_val_prediction_errors_%s_%d.csv' % (control_str, i))
        df_mean = pd.concat((df_mean,prediction_errors), axis = 0)

    df_mean.to_csv('data/calc_U/00cross_val_prediction_errors_sum.csv')#index=False)

    for i, g in df_mean.groupby('qn'):
        g.to_csv('data/calc_U/00cross_val_prediction_errors_qn_%d.csv' % (i))#index=False)

    return df_mean


def h_u():
    '''looking at the different h's for different questions from the kfold'''
    control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (True, True, False, 0)
    df_hs_u = pd.DataFrame()
    for qn in (2,3,4,5):
        for i in range(10):
            fname2read = 'data/calc_U/q_info%s_q_%d_%d.pkl' % (control_str, qn, i)
            q_info = pickle.load(open(fname2read, 'rb'))
            if len(q_info[qn]) > 3:
                ### saving to dataframe all the params
                temp = np.concatenate(
                    (np.array([qn]), q_info[qn]['fal'], np.array([i]),
                     q_info[qn]['q1'], q_info[qn]['q2'],
                     q_info[qn]['U_params_h'][0]),
                    axis=0).reshape(1,15)
                temp = pd.DataFrame(data= temp, columns = ['qn','fal','run','q1','q2'] + ['h_a', 'h_b', 'h_c', 'h_d', 'h_ab', 'h_ac', 'h_ad', 'h_bc', 'h_bd', 'h_cd'])
                df_hs_u = pd.concat((df_hs_u, temp),axis = 0)
    df_hs_u.reset_index(inplace=True,drop=True)
    df_hs_u1 = pd.melt(df_hs_u, id_vars=['qn', 'fal', 'run', 'q1', 'q2'],
                       value_vars=['h_a', 'h_b', 'h_c', 'h_d', 'h_ab', 'h_ac', 'h_ad', 'h_bc', 'h_bd', 'h_cd'],
                       var_name='h', value_name='h_value')
    df_hs_u.to_csv('data/calc_U/df_hs_u.csv')
    df_hs_u1.to_csv('data/calc_U/df_hs_u_melted.csv')

    ### remove outliers.
    df_hs_u1 = df_hs_u1[df_hs_u1['h_value'] > - 5]
    return df_hs_u, df_hs_u1


def my_plot(x, y, **kwargs):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, slope * x + intercept)
    p = ''
    if p_value < 0.05:
        p = '*'
    if p_value < 0.01:
        p = '**'
    if p_value < 0.001:
        p = '***'
    f = '%.2f x + %.2f' % (slope, intercept) + '{}'.format(p)
    props = dict(boxstyle='round', facecolor='gainsboro', alpha=.7)
    plt.text(0.0, 0.0, f, size=9, bbox=props)
    plt.scatter(x, y, s = .5,alpha = .5, **kwargs)


def main():
    h_type = [0]
    use_U_l = [True]
    use_neutral_l = [False]
    with_mixing_l = [True]
    comb = product(h_type, use_U_l, use_neutral_l, with_mixing_l)

    # calcU = True
    calcU = False

    ### How many times to repeat the cross validation
    if calcU:

        for h_mix_type, use_U, use_neutral, with_mixing in comb:

            print('Running:\tUse_U = {} |\tUse_Neutral = {} |\tWith_Mixing = {} |\th_mix_type = {}'.format(use_U,use_neutral,with_mixing, h_mix_type))

            # if os.path.isfile('./data/' + control_str):
            #     print('Already calculated everything for this combination')
            #     continue

            # for i in range(num_of_repeats):
            #     print('Performing cross validation number %d / %d'%(i, num_of_repeats))
            #     calculate_all_data_cross_val(use_U=use_U, use_neutral=use_neutral, with_mixing=with_mixing, h_mix_type=h_mix_type, i = i)

            calculate_all_data_cross_val_kfold(use_U=use_U, use_neutral=use_neutral, with_mixing=with_mixing, h_mix_type=h_mix_type)
    else:
        i = 0
        for h_mix_type, use_U, use_neutral, with_mixing in comb:
            # prediction_errors = pd.read_csv('data/calc_U_unbounded/kfold_prediction_errors.csv')
            prediction_errors = pd.read_csv('data/calc_U/kfold_prediction_errors.csv')

            plot_errors(prediction_errors)

            # ### plot real probs vs. predicted probs
            # for p in ['a', 'b']: #, 'ab'
            #     df = prediction_errors[prediction_errors.columns[prediction_errors.columns.str.contains('p_%s_'%p)]]
            #     g = sns.PairGrid(df)
            #     g.map(my_plot)

            # ### look at the behavior of the parameters of U.
            # df_hs_u, df_hs_u_melted = h_u()
            # ### plot {h} params of U per question for 10 kfolds
            # qns = df_hs_u_melted['qn'].unique()
            # s1 = ''
            # for i in qns:
            #     s = df_hs_u_melted[df_hs_u_melted['qn'] == i][['qn', 'fal', 'q1', 'q2']][-1:].values.flatten()
            #     s1 = s1 + 'qn = %d, fal = %d, q1,q2 = %d, %d \n' % (s[0], s[1], s[2], s[3])
            # print(s1)
            # g = sns.factorplot(x="qn", hue='h', y="h_value", data=df_hs_u_melted, kind="box", size=4, aspect=2)
            # g = sns.factorplot(x="h", hue='qn', y="h_value", data=df_hs_u_melted, kind="box", size=4, aspect=2)

            plt.show()

if __name__ == '__main__':
    main()
