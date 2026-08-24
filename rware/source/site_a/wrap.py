import pandas as pd
import numpy as np
import random
import time
import math
import warnings
import itertools
from pathlib import Path

from rware.data.cost_maps import load_cost_map

warnings.filterwarnings(action='ignore')


def order_generate_site_a(routing_node_list, date_test, ROBOT_CAPA):
    # Resolve CSV path relative to this module so running from any CWD works.
    # data_path = Path(__file__).resolve().parent / "site_a_test.csv"
    data_path = Path(__file__).resolve().parent / "site_a_2.csv"

    PCS_LIMIT = 20
    new_routing_node_list = routing_node_list.copy()
    new_routing_node_list.insert(5, [])
    ROBOT_CAPA = 8
    SEQ_STRATEGY = 0

    node_list = []
    zone_list = []
    for node in new_routing_node_list:
        if node == []:
            if node_list == []:
                pass
            else:
                zone_list.append(node_list)
                node_list = []

            continue
        else:
            node_list += node

    df_site_a = pd.read_csv(data_path)
    month = int(df_site_a.loc[0, 'ADDDATE'][5:7])
    if month in [6,7]:
        print(f'Month : {month}')
        columns_list = df_site_a.columns
        del_idx_list = [3, 4, 5]
        del_list = list()
        for idx in del_idx_list:
            del_list.append(columns_list[idx])
        df_site_a.drop(del_list, axis=1, inplace=True)
        df_site_a.dropna(axis=0, inplace=True)
        order_key_list = df_site_a.loc[:, 'ADDDATE']
        save_text = list()
        for name in order_key_list:
            save_text.append(str(name) + ' 11:11:11.111')

        df_site_a['ADDDATE'] = save_text
    elif month in [3,4,5]:

        columns_list = df_site_a.columns
        idx_list = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27]
        drop_list = list()
        for idx in idx_list:
            drop_list.append(columns_list[idx])
        df_site_a.drop(drop_list, axis=1, inplace=True)

    df_site_a_PCS_LIMIT = df_site_a[df_site_a['QTYPICKED'] >= 1]
    df_site_a_PCS_LIMIT = df_site_a_PCS_LIMIT[df_site_a_PCS_LIMIT['QTYPICKED'] <= PCS_LIMIT]

    # 특정 문자열 포함한 행 제거
    df_site_a_PCS_LIMIT = df_site_a_PCS_LIMIT[~df_site_a_PCS_LIMIT['LOC'].str.contains('A01', na=False, case=False)]
    df_site_a_PCS_LIMIT = df_site_a_PCS_LIMIT[~df_site_a_PCS_LIMIT['LOC'].str.contains('G', na=False, case=False)]
    df_site_a_PCS_LIMIT = df_site_a_PCS_LIMIT[~df_site_a_PCS_LIMIT['LOC'].str.contains('H', na=False, case=False)]
    df_site_a_PCS_LIMIT = df_site_a_PCS_LIMIT[~df_site_a_PCS_LIMIT['LOC'].str.contains('K', na=False, case=False)]
    df_site_a_PCS_LIMIT = df_site_a_PCS_LIMIT[~df_site_a_PCS_LIMIT['LOC'].str.contains('F', na=False, case=False)]
    df_site_a_PCS_LIMIT = df_site_a_PCS_LIMIT[~df_site_a_PCS_LIMIT['LOC'].str.contains('c', na=False, case=True)]

    date_time_list = list(df_site_a_PCS_LIMIT.loc[:, 'ADDDATE'])
    date_list = list()
    for time in date_time_list:
        date_list.append(time[5:-13])
    df_site_a_PCS_LIMIT['ORDERDATE'] = date_list
    date_list = list()
    for time in date_time_list:
        date_list.append(time[11:-10])
    df_site_a_PCS_LIMIT['ADDHOUR'] = date_list

    df_site_a_PCS_LIMIT.sort_values(by='ADDDATE', ascending=True, inplace=True)
    date_set_list = sorted(list(set(list(df_site_a_PCS_LIMIT.loc[:, 'ORDERDATE']))))
    # Set date
    date = date_set_list[date_test]
    df_site_a_PCS_LIMIT_date = df_site_a_PCS_LIMIT[df_site_a_PCS_LIMIT['ORDERDATE'] == date]

    # Rack generator
    A_line = 17
    A_init = 406
    A_line_length = 31
    A_line_gap = 55
    A_rack_list = list()

    for i in range(A_line):
        if i == (A_line - 1):
            A_line_length = 21
        sub_A_list = list(range(A_init + (A_line_gap * i), A_init + (A_line_gap * i) + A_line_length))

        if i == 0:
            A_rack_list.append(sub_A_list)
        elif (i % 2) != 0:
            A_rack_list.append(sub_A_list)
        else:
            A_rack_list[i // 2] = A_rack_list[i // 2] + sub_A_list

    B_line = 29
    B_init = 153
    B_line_length = 24
    B_rack_list = list()

    for i in range(B_line):
        if i == 0 or i == 1 or i == 2:
            B_init = B_init + 45
        elif i == 3 or i == 4:
            B_init = B_init + 47
        elif i == 21:
            B_init = B_init + 45
        elif i >= 22:
            B_init = B_init + B_line_length
        else:
            B_init = B_init + 55

        sub_B_list = list(range(B_init, B_init + B_line_length))

        if i == 0:
            B_rack_list.append(sub_B_list)
        elif (i % 2) != 0:
            B_rack_list.append(sub_B_list)
        else:
            B_rack_list[i // 2] = B_rack_list[i // 2] + sub_B_list

    C_line = 12
    C_init = 9
    C_line_length = 21
    C_rack_list = list()

    for i in range(C_line):
        if i == 8 or i == 9:
            C_init = C_init + 45
        elif i == 10:
            C_init = C_init + 45
            C_line_length = 23
        elif i == 11:
            C_init = C_init + 47
            C_line_length = 23
        else:
            C_init = C_init + 21
        sub_C_list = list(range(C_init, C_init + C_line_length))

        if i == 0:
            C_rack_list.append(sub_C_list)
        elif (i % 2) != 0:
            C_rack_list.append(sub_C_list)
        else:
            C_rack_list[i // 2] = C_rack_list[i // 2] + sub_C_list

    Z_line = 10
    Z_init = 472
    Z_line_length = 20
    Z_line_gap = 55
    Z_rack_list = list()

    for i in range(Z_line):
        if i == (Z_line - 1):
            Z_line_length = 20
        sub_Z_list = list(range(Z_init + (Z_line_gap * i), Z_init + (Z_line_gap * i) + Z_line_length))

        if (i % 2) == 0:
            Z_rack_list.append(sub_Z_list)
        else:
            Z_rack_list[i // 2] = Z_rack_list[i // 2] + sub_Z_list

    E_rack_list = list(range(1, 30)) + [50, 71, 92, 113, 134, 155]

    # Rack Mapping
    loc_list = list(df_site_a_PCS_LIMIT_date.loc[:, 'LOC'])
    loc_save_list = list()
    for loc in loc_list:
        if loc[0] == 'A':
            loc_save_list.append(A_rack_list[10 - int(loc[1:3])][int(loc[4:6]) - 1])
        elif loc[0] == 'B':
            loc_save_list.append(B_rack_list[15 - int(loc[1:3])][int(loc[4:6]) - 1])
        elif loc[0] == 'C':
            if int(loc[1:3]) == 1:
                loc_save_list.append(C_rack_list[7 - int(loc[1:3])][int(loc[4:6]) - 24])
            elif int(loc[1:3]) == 2:
                if (int(loc[4:6]) - 1) >= 24:
                    loc_save_list.append(C_rack_list[7 - int(loc[1:3])][int(loc[4:6]) - 3])
                else:
                    loc_save_list.append(C_rack_list[7 - int(loc[1:3])][int(loc[4:6]) - 1])
            else:
                loc_save_list.append(C_rack_list[7 - int(loc[1:3])][int(loc[4:6]) - 1])
        elif loc[0] == 'Z':
            loc_save_list.append(Z_rack_list[9 - int(loc[1:3])][(int(loc[4:6]) // 2) - 1])
        elif loc[0] == 'E':
            loc_save_list.append(E_rack_list[35 - int(loc[1:3]) - 1])
        else:
            #             print(loc[0])
            print('ERROR')

    df_site_a_PCS_LIMIT_date['RACK'] = loc_save_list

    node_list = list()
    for rack in loc_save_list:
        for idx, node_rack_list in enumerate(zone_list):
            if node_rack_list == []:
                pass
            else:
                if rack in node_rack_list:
                    node_list.append(idx)
    df_site_a_PCS_LIMIT_date['NODE'] = node_list

    old_node_list = list()
    for rack in loc_save_list:
        for idx, node_rack_list in enumerate(routing_node_list):
            if node_rack_list == []:
                pass
            else:
                if rack in node_rack_list:
                    old_node_list.append(idx)
    df_site_a_PCS_LIMIT_date['OLD_NODE'] = old_node_list

    #     FOR DEBUGGING
    #     if loc[:-7] == 'A10':
    #         loc_save_list.append(A_rack_list[0][int(loc[4:6]) - 1])

    #     if loc[:-7] == 'Z09':
    #         loc_save_list.append(Z_rack_list[0][int(loc[4:6])//2])

    #     elif loc[:-7] == 'Z08':
    #         loc_save_list.append(Z_rack_list[1][int(loc[4:6])//2])

    #     elif loc[:-7] == 'Z07':
    #         loc_save_list.append(Z_rack_list[2][int(loc[4:6])//2])

    #     elif loc[:-7] == 'Z06':
    #         loc_save_list.append(Z_rack_list[3][int(loc[4:6])//2])

    #     elif loc[:-7] == 'Z05':
    #         loc_save_list.append(Z_rack_list[4][int(loc[4:6])//2])

    #     elif loc[:-7] == 'C04':
    #         loc_save_list.append(C_rack_list[3][int(loc[4:6]) - 1])

    #     elif loc[:-7] == 'C03':
    #         loc_save_list.append(C_rack_list[4][int(loc[4:6]) - 1])

    #     elif loc[:-7] == 'C02':
    #         loc_save_list.append(C_rack_list[5][int(loc[4:6]) - 1])

    #     elif loc[:-7] == 'C01':
    #         print()
    #         print(int(loc[4:6]) - 1)
    #         print(C_rack_list[6])
    #         print(len(C_rack_list[6]))
    #         loc_save_list.append(C_rack_list[6][int(loc[4:6]) - 22])
    #     print(loc[:-4])
    #     print(loc[:-7])
    #     loc_save_list.append(loc[:-4])

    df_site_a_PCS_LIMIT_date.drop(['ADDDATE', 'LOC'], axis=1, inplace=True)
    df_site_a_PCS_LIMIT_date.sort_values(by='ORDERKEY', ascending=True, inplace=True)

    invoice_numbers = sorted(list(set(list(df_site_a_PCS_LIMIT_date.loc[:, 'ORDERKEY']))))
    str_set_node_list = list()
    for invoice_number in invoice_numbers:
        df_processing = df_site_a_PCS_LIMIT_date[df_site_a_PCS_LIMIT_date['ORDERKEY'] == invoice_number]
        for _ in range(len(df_processing)):
            str_set_node_list.append(str(sorted(list(set(list(df_processing.loc[:, 'NODE']))))))
    df_site_a_PCS_LIMIT_date['STR_SET_NODE'] = str_set_node_list
    #     print(df_site_a_PCS_LIMIT_date)
    return df_site_a_PCS_LIMIT_date, A_rack_list, B_rack_list, C_rack_list


##########################################################
def order_batch(df_site_a_PCS_LIMIT_date, ROBOT_CAPA):
    # 시간별 처리
    times = sorted(list(set(list(df_site_a_PCS_LIMIT_date.loc[:, 'ADDHOUR']))))
    total_order = []
    seq_total = list()  ######
    for time in times:
        #     time = times[1]
        df_site_a_PCS_LIMIT_date_hour = df_site_a_PCS_LIMIT_date[df_site_a_PCS_LIMIT_date['ADDHOUR'] == time]
        df_site_a_PCS_LIMIT_date_hour.sort_values(by=['STR_SET_NODE'], ascending=[True], inplace=True)

        batch_order_list = list()
        order_list = list()
        seq_list = list()  ##########
        seq_order_total = list()  ##########
        cnt = 0
        for i in range(len(df_site_a_PCS_LIMIT_date_hour)):
            if cnt == 0:
                order_list.append([df_site_a_PCS_LIMIT_date_hour.iloc[i, 4], df_site_a_PCS_LIMIT_date_hour.iloc[i, 1],
                                   df_site_a_PCS_LIMIT_date_hour.iloc[i, 6]])
                seq_list.append(df_site_a_PCS_LIMIT_date_hour.iloc[i, 4])  ##########
                cnt += 1
            elif cnt >= 1 and cnt <= ROBOT_CAPA - 1:
                if df_site_a_PCS_LIMIT_date_hour.iloc[i, 0] == df_site_a_PCS_LIMIT_date_hour.iloc[i - 1, 0]:
                    cnt = cnt
                else:
                    cnt += 1
                order_list.append([df_site_a_PCS_LIMIT_date_hour.iloc[i, 4], df_site_a_PCS_LIMIT_date_hour.iloc[i, 1],
                                   df_site_a_PCS_LIMIT_date_hour.iloc[i, 6]])
                seq_list.append(df_site_a_PCS_LIMIT_date_hour.iloc[i, 4])  ##########

            elif cnt == ROBOT_CAPA:
                batch_order_list.append(order_list)
                order_list = list()
                order_list.append([df_site_a_PCS_LIMIT_date_hour.iloc[i, 4], df_site_a_PCS_LIMIT_date_hour.iloc[i, 1],
                                   df_site_a_PCS_LIMIT_date_hour.iloc[i, 6]])
                seq_list = list(set(seq_list))  ##########
                seq_order_total.append(seq_list)  ##########
                seq_list = list()  ##########
                seq_list.append(df_site_a_PCS_LIMIT_date_hour.iloc[i, 4])  ##########
                cnt = 1
            else:
                print('idx {}  ERROR'.format(i))

        if len(order_list) >= 1:
            batch_order_list.append(order_list)
            seq_order_total.append(seq_list)
        else:
            pass

        input_order = []
        for rack_pcs_node_list in batch_order_list:
            rack_pcs_node_list = sorted(rack_pcs_node_list)
            order_list = []
            cnt = 0
            #         print(rack_pcs_node_list)
            for idx in range(len(rack_pcs_node_list)):
                if len(order_list) == 0:
                    order_list.append(rack_pcs_node_list[idx])
                else:
                    if order_list[idx - 1 - cnt][0] == rack_pcs_node_list[idx][0]:
                        order_list[idx - 1 - cnt][1] = order_list[idx - 1 - cnt][1] + rack_pcs_node_list[idx][1]
                        cnt += 1
                    else:
                        order_list.append(rack_pcs_node_list[idx])

            input_order.append(order_list)
        total_order = total_order + input_order
        seq_total = seq_total + seq_order_total  ##########
    return total_order, seq_total


#         seq_total = seq_total + seq_order_total

############################################################
#     Generate Seq
def order_sequencing(total_order, seq_list, SEQ_STRATEGY):
    if SEQ_STRATEGY == 0: pass
        # print('SiteA default')
        # seq_list = sorted(list(range(1, 30)))
        # for idx in range(len(C_rack_list)):
        #     if idx == 0:
        #         seq_list = seq_list + C_rack_list[idx]
        #     elif idx == (len(C_rack_list) - 1):
        #         seq_list = seq_list + sorted(C_rack_list[idx], reverse=True)
        #     else:
        #         seq_list = seq_list + sorted(C_rack_list[idx][:int(len(C_rack_list[idx]) / 2)], reverse=True) + \
        #                    C_rack_list[idx][int(len(C_rack_list[idx]) / 2):]
        #
        # for idx in range(len(A_rack_list)):
        #     if idx == 0:
        #         seq_list = seq_list + A_rack_list[idx]
        #     elif idx == (len(C_rack_list) - 1):
        #         seq_list = seq_list + sorted(A_rack_list[idx][:int(len(A_rack_list[idx - 1]) / 2)], reverse=True) + (
        #         A_rack_list[idx][int(len(A_rack_list[idx - 1]) / 2):])
        #     else:
        #         seq_list = seq_list + sorted(A_rack_list[idx][:int(len(A_rack_list[idx]) / 2)], reverse=True) + \
        #                    A_rack_list[idx][int(len(A_rack_list[idx]) / 2):]
        #
        # flag = 14
        # for idx in range(len(B_rack_list)):
        #     if idx == 0:
        #         seq_list = seq_list + B_rack_list[idx][:flag] + sorted(B_rack_list[idx][flag:], reverse=True)
        #     else:
        #         B_rack_up = B_rack_list[idx][:int(len(B_rack_list[idx]) / 2)]
        #         B_rack_down = B_rack_list[idx][int(len(B_rack_list[idx]) / 2):]
        #         seq_list = seq_list + sorted(B_rack_up[:flag], reverse=True) + B_rack_up[flag:] + B_rack_down[
        #                                                                                           :flag] + sorted(
        #             B_rack_down[flag:], reverse=True)
        #
        # seq_dict = {}
        # for i in enumerate(seq_list):
        #     seq_dict[i[1]] = i[0]
        #
        # for order_list_idx in range(len(total_order)):
        #     total_order[order_list_idx] = sorted(total_order[order_list_idx], key=lambda x: seq_dict[x[0]])

    elif SEQ_STRATEGY == 1:

        print('SiteA Matrix calc')

        #############################

        seq_limit = 8  # for permutation limit = 8

        matrix = load_cost_map()

        # seq_order = [10,100,12,15,750,16,19,20,1400,123,124,125,126,127,128,129,13,131,132,133,134,145,146,147,199]

        seq_ordered = list()

        length_list = list()  ############################### length save

        for sequenced_list in seq_list:

            sum_length = 0

            optima_seq = list()

            ####### 사전 시퀀싱 자리 ######

            sequenced_list = [(sequenced_list[i:i + seq_limit]) for i in range(0, len(sequenced_list), seq_limit)]

            ###############################

            for seq_order in sequenced_list:

                if len(seq_order) == 1:

                    #             print(seq_order)

                    result_seq = seq_order

                else:
                    #             print(seq_order)
                    result_seq = []
                    main_length = 99999999999999
                    for seq in itertools.permutations(seq_order, len(seq_order)):
                        sum_length = 0
                        for idx in range(len(seq) - 1):
                            sum_length += matrix[seq[idx]][seq[idx + 1]]
                        if main_length >= sum_length:
                            main_length = sum_length
                            result_seq = seq
                optima_seq.append(list(result_seq))
            optima_seq = list(itertools.chain(*optima_seq))
            # print('------------------------------')
            # print(optima_seq)
            # print(main_length)
            # print('------------------------------')
            length_list.append(main_length)
            seq_ordered.append(optima_seq)
        seq_dict_list = list()
        for seq_list in seq_ordered:
            seq_dict = dict()
            for idx in enumerate(seq_list):
                seq_dict[idx[1]] = idx[0]
            seq_dict_list.append(seq_dict)
        for idx in range(len(seq_dict_list)):
            #             print()
            #             print(idx)
            #             print(total_order[idx])
            #             print(seq_dict_list[idx])
            total_order[idx] = sorted(total_order[idx], key=lambda x: seq_dict_list[idx][x[0]])
            # print('--------------------------')
            total_order[idx].append(length_list[idx])
            # print(total_order[idx])
        #############################
    #     print(seq_list)
    #     print('Date : 2023-{}'.format(date))

    total_order = sorted(total_order, key=lambda x: x[-1], reverse=True)
    del_length = list()

    for order in total_order:
        print('---------------------------')
        print(order)
        del order[-1]
        del_length.append(order)
        print(order)
    print('Total_Batch : {}'.format(len(total_order)))
    #     print('Total_Order : {}'.format((int(len(total_order)-1) * ROBOT_CAPA) + int(len(total_order[-1]))))
    return total_order


### generate random order
def generate_initial_population(num, ROBOT_CAPA, RACK_NUM):
    initial_population = np.random.randint(0, 2, (num, ROBOT_CAPA, RACK_NUM))
    return initial_population


### evaluate fitness
def evaluate_fitness(batch):
    order_diff = 0
    #     print(len(batch[0]))
    for x_idx in range(len(batch[0])):
        sum_diff = 0
        for y_idx in range(len(batch)):
            sum_diff += batch[y_idx][x_idx]
        if sum_diff == 1:
            sum_diff = -1
        elif sum_diff == 0:
            sum_diff = 0
        elif sum_diff == 4:
            sum_diff = 7
        order_diff += sum_diff

    return int(order_diff)


### Crossover_point generate
def random_crossover_point(lower, upper, point):
    cnt = 0
    crossover_point_list = list()
    while cnt != point:
        crossover_point = random.randint(lower, upper)
        if crossover_point not in crossover_point_list:
            crossover_point_list.append(crossover_point)
            cnt += 1
        else:
            pass

    return sorted(crossover_point_list)


# def crossover_n_mutation(batch_1, batch_2, lower, upper, point, ROBOT_CAPA, RACK_NUM):
#     #  cross over
# #     crossover_point_1 = random_crossover_point(lower, upper, point)
# #     crossover_point_2 = random_crossover_point(lower, upper, point)
#     child = np.empty((ROBOT_CAPA, RACK_NUM))
# #     parent 50 : 50
#     for idx in range(len(child)):
#         if idx < point:
#             child[idx] = batch_1[idx]
#         else:
#             child[idx] = batch_2[idx]

#     return child

def crossover_n_mutation(batch_1, batch_2, batch_1_idx, batch_2_idx, point, ROBOT_CAPA, RACK_NUM):
    #  cross over
    child = np.empty((ROBOT_CAPA, RACK_NUM))
    child_idx = np.empty(ROBOT_CAPA)
    #     parent 50 : 50
    for idx in range(len(child)):
        if idx < point:
            child[idx] = batch_1[idx]
            child_idx[idx] = batch_1_idx[idx]
        else:
            child[idx] = batch_2[idx]
            child_idx[idx] = batch_2_idx[idx]

    return child, child_idx


def order_batch_GA(current_population_main, current_population_idx_main, num_iter, num_group, num_parent, point, RACK_NUM):
    #     print(np.shape(current_population_main))
    best_score = -9999999999

    for num in range(num_iter):
        start_time = time.time()
        print(f'{num + 1}/{num_iter} iteration ...')
        ###     validate_score
        fitness_value_list = np.array([evaluate_fitness(solution) for solution in current_population_main])

        if fitness_value_list.max() > best_score:
            best_score = fitness_value_list.max()
            #             print('--------------------------------------')
            #             print(fitness_value_list)
            #             print(fitness_value_list.argmin())
            #             print('--------------------------------------')
            best_solution = current_population_main[fitness_value_list.argmax()]
            best_idx = current_population_idx_main[fitness_value_list.argmax()]

        parents = current_population_main[np.argsort(fitness_value_list)][:num_parent]
        parents_idx = current_population_idx_main[np.argsort(fitness_value_list)][:num_parent]
        new_population = parents
        new_population_idx = parents_idx
        for cnt in range(num_group - num_parent):
            parent_1_idx, parent_2_idx = np.random.choice(num_parent, 2, replace=False)
            parent_1 = parents[parent_1_idx]
            parent_1_idx = parents_idx[parent_1_idx]
            parent_2 = parents[parent_2_idx]
            parent_2_idx = parents_idx[parent_2_idx]

            child, child_idx = crossover_n_mutation(parent_1, parent_2, parent_1_idx, parent_2_idx, point, 4, RACK_NUM)

            #             if cnt % 4 == 0:
            #                 child_idx = random.randint(0,ROBOT_CAPA - 1 - mutation_length)
            #                 not_parents = current_population[np.argsort(fitness_value_list)][num_parent:]
            #                 population_idx = random.randint(0,len(not_parents)-1)
            #                 child[child_idx : child_idx + mutation_length], not_parents[population_idx][child_idx : child_idx + mutation_length] = not_parents[population_idx][child_idx : child_idx + mutation_length], child[child_idx : child_idx + mutation_length]

            new_population = np.vstack([new_population, child.reshape(1, 4, RACK_NUM)])
            new_population_idx = np.vstack([new_population_idx, child_idx.reshape(1, 4)])

        current_population_main = new_population
        current_population_idx_main = new_population_idx
        print(f'Result : {best_score}')
        print(f'Sec : {time.time() - start_time:.2f}')
    return best_idx


def order_batch_compress(best_idx, input_idx_order_list, input_order):
    batch_list = list()
    for idx in best_idx:
        idx = int(idx)
        if idx == 9999:
            pass
        else:
            for idx_order in range(len(input_idx_order_list)):
                if input_idx_order_list[idx_order][0] == idx:
                    for order in input_order[idx_order]:
                        batch_list.append(order)

    batch_list = sorted(batch_list, key=lambda x: x[0])

    cnt = 0
    batch_compress_list = list()
    for idx in range(len(batch_list)):
        if idx == 0:
            batch_compress_list.append(batch_list[idx])
        else:
            if batch_compress_list[idx - 1 - cnt][0] == batch_list[idx][0]:
                batch_compress_list[idx - 1 - cnt][1] += batch_list[idx][1]
                cnt += 1
            else:
                batch_compress_list.append(batch_list[idx])
                cnt = cnt

    return batch_compress_list








