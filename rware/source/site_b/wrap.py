import pandas as pd
import random
import itertools
import warnings
import numpy as np
import math
warnings.filterwarnings('ignore')



def order_generate(list_nodes):
    ###########
    RACK_NUM = 156  # 1 ~ 1356
    ZONE_NUM = 16  # 0 ~ 15
    ROBOT_CAPA = 6
    data_path = 'source/site_b/site_b_data.csv'
    df_site_b = pd.read_csv(data_path)
    loc_list = sorted(list(set(list(df_site_b['LOC']))))  # 272
    job_serialnumber_list = sorted(list(set(list(df_site_b['송장번호']))))  # 644
    ############

    list_rack_1 = list(range(1, 152, 6))
    list_rack_2 = list(range(2, 153, 6))
    list_rack_3 = list(range(3, 154, 6))
    list_rack_4 = list(range(4, 155, 6))
    list_rack_5 = list(range(5, 156, 6))
    list_rack_6 = list(range(6, 157, 6))
    s03_rack = sorted(list_rack_1 + list_rack_2)
    s04_rack = sorted(list_rack_3 + list_rack_4)
    s05_rack = sorted(list_rack_5 + list_rack_6)

    s03_mapping = []
    s04_mapping = []
    s05_mapping = []
    for i in loc_list:
        rack_num = i[4:6]
        if i[:3] == 'S05':
            s03_mapping.append([i, s03_rack[int(rack_num) - 1]])
        elif i[:3] == 'S04':
            s04_mapping.append([i, s04_rack[int(rack_num) - 1]])
        elif i[:3] == 'S03':
            s05_mapping.append([i, s05_rack[int(rack_num) - 1]])

    rack_save_list = []

    for rack in range(len(df_site_b)):
        if df_site_b.iloc[rack][1][:3] == 'S05':
            for map_rack in s03_mapping:
                #             print(map_rack)
                if df_site_b.iloc[rack][1] == map_rack[0]:
                    rack_save_list.append(map_rack[1])

        elif df_site_b.iloc[rack][1][:3] == 'S04':
            for map_rack in s04_mapping:
                #             print(map_rack)
                if df_site_b.iloc[rack][1] == map_rack[0]:
                    rack_save_list.append(map_rack[1])

        elif df_site_b.iloc[rack][1][:3] == 'S03':
            for map_rack in s05_mapping:
                #             print(map_rack)
                if df_site_b.iloc[rack][1] == map_rack[0]:
                    rack_save_list.append(map_rack[1])

    df_site_b['Rack_mapping'] = rack_save_list

    node_list = []
    for i in range(len(list_nodes)):
        if i == []:
            continue
        else:
            node_list.append([i, list_nodes[i]])

    node_list_save = []
    for i in range(len(df_site_b)):
        for j in range(len(node_list)):
            if df_site_b.iloc[i][3] in node_list[j][1]:
                node_list_save.append(node_list[j][0])
    df_site_b['Node_mapping'] = node_list_save

    result_list = []
    for num in job_serialnumber_list:

        job_serialnumber = num
        sku_per_box = 500

        df_save = df_site_b[df_site_b['송장번호'] == job_serialnumber]
        df_save.sort_values(by=['LOC', '출고량', 'Rack_mapping', 'Node_mapping'], ascending=[False, False, False, False],
                            inplace=True)

        for idx in range(len(df_save)):
            if df_save.iloc[idx][2] > sku_per_box:
                single_box = df_save.iloc[idx][2] // sku_per_box
                left_sku = df_save.iloc[idx][2] % sku_per_box
                for num in range(single_box):
                    new_box = {
                        df_save.columns[0]: [df_save.iloc[idx][0]],
                        df_save.columns[1]: [df_save.iloc[idx][1]],
                        df_save.columns[2]: [sku_per_box]
                    }
                    new_df = pd.DataFrame(new_box)
                    df_save = pd.concat([df_save, new_df])

                if left_sku > 0:
                    left_box = {
                        df_save.columns[0]: [df_save.iloc[idx][0]],
                        df_save.columns[1]: [df_save.iloc[idx][1]],
                        df_save.columns[2]: [left_sku]
                    }
                    left_df = pd.DataFrame(left_box)
                    df_save = pd.concat([df_save, left_df])

                else:
                    continue

            else:
                continue
        divide_box_idx = df_save[df_save['출고량'] > sku_per_box].index
        df_save.drop(divide_box_idx, inplace=True)
        df_save.sort_values(by=['출고량', 'LOC', 'Rack_mapping', 'Node_mapping'], ascending=[True, False, False, False],
                            inplace=True)
        df_save.index = range(0, len(df_save))

        list_group_idx = []
        sku_cnt = 0

        df_sku_limit = df_save.copy()
        box_idx_sku_per_picking = df_save[df_save['출고량'] == sku_per_box].index
        df_sku_limit.drop(box_idx_sku_per_picking, inplace=True)
        df_sku_limit.sort_values(by=['출고량', 'LOC', 'Rack_mapping', 'Node_mapping'],
                                 ascending=[False, False, False, False], inplace=True)

        for idx in range(len(df_save)):
            if df_save.iloc[idx][2] == sku_per_box:
                continue

            elif df_save.iloc[idx][2] > sku_per_box:
                print('Error index : {}'.format(idx))

            else:
                df_idx_save = df_sku_limit.index
                list_sku_save = []
                list_sku_idx_save = []
                list_sku_rack_save = []

                for value in range(len(df_sku_limit)):
                    sum_list = sum(list_sku_save)
                    if sum_list < sku_per_box:
                        list_sku_save.append(df_sku_limit.iloc[value][2])
                        list_sku_idx_save.append(df_idx_save[value])
                        list_sku_rack_save.append(df_sku_limit.iloc[value][3])

                    elif sum_list == sku_per_box:
                        break

                    else:
                        del list_sku_save[-1]
                        del list_sku_idx_save[-1]
                        del list_sku_rack_save[-1]

                if sum(list_sku_save) > sku_per_box:
                    del list_sku_save[-1]
                    del list_sku_idx_save[-1]
                    del list_sku_rack_save[-1]

                df_sku_limit.drop(index=list_sku_idx_save, axis=0, inplace=True)

                if list_sku_idx_save == []:
                    continue
                else:
                    list_group_idx.append([list_sku_rack_save, list_sku_save])
        list_node_save = str(sorted(list(set(list(df_save.iloc[:, -1])))))

        list_group_idx.append(len(list_group_idx))
        list_group_idx.append(list_node_save)
        result_list.append(list_group_idx)

    df_order_node = pd.DataFrame([result_list[i][-2:] for i in range(len(result_list))],
                                 columns=['Num_order', 'Node_set'])
    df_order_node['idx'] = list(range(len(result_list)))

    list_node_set = sorted(list(set(list(df_order_node.iloc[:, 1]))))
    order_batch_list_total = []
    for num in range(len(list_node_set)):
        #     idx = 6
        df_node_extract = df_order_node[df_order_node['Node_set'] == list_node_set[num]]
        df_node_extract.sort_values(['Num_order', 'idx'], ascending=[False, False], inplace=True)
        df_node_extract_copy = df_node_extract.copy()

        order_batch_list = []

        for idx in range(len(df_node_extract)):
            if df_node_extract.iloc[idx][0] == ROBOT_CAPA:
                continue

            elif df_node_extract.iloc[idx][0] > ROBOT_CAPA:
                print('Error index : {}'.format(idx))

            else:
                num_order_save_list = []
                idx_save_list = []

                for value in range(len(df_node_extract_copy)):
                    sum_num_list = sum(num_order_save_list)

                    if sum_num_list < ROBOT_CAPA:
                        num_order_save_list.append(df_node_extract_copy.iloc[value][0])
                        idx_save_list.append(df_node_extract_copy.iloc[value][2])

                    elif sum_num_list == 6:
                        break

                    else:
                        del num_order_save_list[-1]
                        del idx_save_list[-1]

                if sum_num_list > ROBOT_CAPA:
                    del num_order_save_list[-1]
                    del idx_save_list[-1]

                df_node_extract_copy.drop(index=idx_save_list, axis=0, inplace=True)

                if idx_save_list == []:
                    continue

                else:
                    order_batch_list.append([sum(num_order_save_list), idx_save_list])
        order_batch_list_total.append(order_batch_list)

    order_batch_min_list = []
    for order_batch_idx in range(len(order_batch_list_total)):
        order_min = order_batch_list_total[order_batch_idx].pop(-1)
        if order_min[0] == 6:
            order_batch_list_total[order_batch_idx].insert(-1, order_min)
        else:
            order_batch_min_list.append(order_min)

    order_batch_min_list = sorted(order_batch_min_list, reverse=True)
    order_batch_min_list_copy = order_batch_min_list.copy()

    batch_list_6 = []

    for _ in range(len(order_batch_min_list)):
        num_batch_save_list = []
        batch_rack_save_list = []
        batch_idx_save = []
        cnt = 0
        for j in range(len(order_batch_min_list_copy)):
            sum_batch_list = sum(num_batch_save_list)

            if sum_batch_list < 6:
                num_batch_save_list.append(order_batch_min_list_copy[j][0])
                batch_rack_save_list.append(order_batch_min_list_copy[j][1])
                batch_idx_save.append(j)

            elif sum_batch_list == 6:
                break

            else:
                del num_batch_save_list[-1]
                del batch_rack_save_list[-1]
                del batch_idx_save[-1]

        for batch_idx in batch_idx_save:
            del order_batch_min_list_copy[batch_idx - cnt]
            cnt += 1

        if sum(num_batch_save_list) == 0:
            continue
        else:
            batch_list_6.append([sum(num_batch_save_list), batch_rack_save_list])

    for pre_idx in range(len(batch_list_6)):
        batch_list_6[pre_idx][1] = list(itertools.chain(*batch_list_6[pre_idx][1]))

    list_batch_total_save = []

    for i in order_batch_list_total:
        if i == []:
            continue
        else:
            for j in i:
                list_batch_total_save.append(j)

    list_order_mapping = list_batch_total_save + batch_list_6

    final_batch_order_list = []
    for order_idx in range(len(list_order_mapping)):
        #     print(list_order_mapping[order_idx])
        batch_order_list = []
        for idx in list_order_mapping[order_idx][1]:
            list_save_node_2_rack = []
            for j in range(result_list[idx][-2]):
                for i in range(len(result_list[idx][j][0])):
                    list_save_node_2_rack.append([result_list[idx][j][0][i], result_list[idx][j][1][i]])
            batch_order_list = batch_order_list + list_save_node_2_rack
        final_batch_order_list.append(batch_order_list)

    for i in range(len(final_batch_order_list)):
        for j in range(len(final_batch_order_list[i])):
            for k in range(len(node_list)):
                if final_batch_order_list[i][j][0] in node_list[k][1]:
                    final_batch_order_list[i][j].append(node_list[k][0])
                else:
                    continue

    flag_position = 10
    list_rack_1_1 = list(range(1, 152, 6))[:flag_position]
    list_rack_1_2 = list(range(1, 152, 6))[flag_position:]
    list_rack_2_1 = sorted(list(range(2, 153, 6))[:flag_position], reverse = True)
    list_rack_2_2 = sorted(list(range(2, 153, 6))[flag_position:], reverse = True)
    list_rack_3_1 = list(range(3, 154, 6))[:flag_position]
    list_rack_3_2 = list(range(3, 154, 6))[flag_position:]
    list_rack_4_1 = sorted(list(range(4, 155, 6))[:flag_position], reverse = True)
    list_rack_4_2 = sorted(list(range(4, 155, 6))[flag_position:], reverse = True)
    list_rack_5_1 = list(range(5, 156, 6))[:flag_position]
    list_rack_5_2 = list(range(5, 156, 6))[flag_position:]
    list_rack_6_1 = sorted(list(range(6, 157, 6))[:flag_position], reverse = True)
    list_rack_6_2 = sorted(list(range(6, 157, 6))[flag_position:], reverse = True)

    list_seq_total = list_rack_1_1 + list_rack_1_2 + list_rack_2_1 + list_rack_2_2 + list_rack_3_1 + list_rack_3_2 + list_rack_4_1 + list_rack_4_2 + list_rack_5_1 + list_rack_5_2 + list_rack_6_1 + list_rack_6_2
    seq_dict = {}
    for i in enumerate(list_seq_total):
        seq_dict[i[1]] = i[0]

    for batch_list_idx in range(len(final_batch_order_list)):
        final_batch_order_list[batch_list_idx] = sorted(final_batch_order_list[batch_list_idx], key = lambda x : seq_dict[x[0]])

    full_order_list = []
    for order_list in final_batch_order_list:
        regenerate_order_list = [order_list[0]]
        cnt = 0
        for idx in range(len(order_list) -1):
            if regenerate_order_list[idx - cnt][0] == order_list[idx + 1][0]:
                regenerate_order_list[idx - cnt][1] = regenerate_order_list[idx - cnt][1] + order_list[idx + 1][1]
                cnt += 1
            else:
                regenerate_order_list.append(order_list[idx + 1])
        full_order_list.append(regenerate_order_list)
    # print(full_order_list)
    return full_order_list, final_batch_order_list


def randomize_order(list_dummy):
    random.seed(42)
    box_per_sku = 60

    list_random = []
    for list_x in list_dummy:
        for value in list_x:
            list_random.append(value)
    random.shuffle(list_random)
    list_random_copy = list_random.copy()

    order_random_list = []
    for i in range(len(list_random)):
        list_order_num = []
        list_order = []
        list_idx = []

        for j in range(len(list_random_copy)):
            sum_total = sum(list_order_num)
            if sum_total < box_per_sku:
                list_order_num.append(list_random_copy[j][1])
                list_order.append(list_random_copy[j])
                list_idx.append(j)

            elif sum_total == box_per_sku:
                break

            else:
                del list_order_num[-1]
                del list_order[-1]
                del list_idx[-1]

        cnt = 0
        for k in list_idx:
            del list_random_copy[k - cnt]
            cnt += 1

        if sum(list_order_num) == 0:
            continue
        else:
            order_random_list.append(list_order)

    cnt = 0
    list_order_6 = []
    list_random_result = []
    for order_list_x in order_random_list:
        list_order_6 = list_order_6 + order_list_x
        cnt += 1
        if cnt == 6:
            list_random_result.append(list_order_6)
            cnt = 0
            list_order_6 = []

    list_random_result.append(random.shuffle(list_order_6))

    return list_random_result

def smallest_sku_node(input_order):
    test_order = input_order.copy()
    result = []

    for test in test_order:
        list_counter = [[0, i] for i in range(16)]
        seq_dict = dict()

        for sample in test:
            list_counter[sample[2]][0] += sample[1]

        list_counter = sorted(list_counter)
        # print(list_counter)

        my_dict = dict()

        for counter in list_counter:
            if counter[0] > 0: my_dict[counter[1]] = counter[0]

        tmp = sorted(test, key=lambda x: my_dict[x[2]])

        result.append(tmp)
        # print(tmp)
        # print()
    return result


# def human_batch(mode, input_order):
#     node_num_dict = {}
#     for order_list in input_order:
#         for pcs_rack in order_list:
#             pcs_rack = pcs_rack[1:]
#             if pcs_rack[1] not in node_num_dict:
#                 node_num_dict[pcs_rack[1]] = pcs_rack[0]
#             elif pcs_rack[1] in node_num_dict:
#                 node_num_dict[pcs_rack[1]] = node_num_dict[pcs_rack[1]] + pcs_rack[0]
#             else:
#                 print('Error')
#
#     #### zone_generate
#     human_zone_1 = [4, 5, 6, 7, 8]
#     human_zone_2 = np.array([13])
#     human_zone_3 = human_zone_2 + 9
#     human_zone_4 = human_zone_3 + 9
#     human_zone_5 = [36,38,40]
#     human_zone_6 = [45,47,49,50]
#     human_zone_7 = np.array([54,56,58,59,60])
#     human_zone_8 = human_zone_7 + 9
#     human_zone_9 = human_zone_8 + 9
#     human_zone_10 = human_zone_9 + 9
#     human_zone_11 = human_zone_10 + 9
#     human_zone_12 = human_zone_11 + 9
#     human_zone_13 = human_zone_12 + 9
#     human_zone_14 = human_zone_13 + 9
#     human_zone_15 = human_zone_14 + 9
#     human_zone_16 = np.array([135,137])
#     human_zone_17 = human_zone_16 + 9
#     human_zone_18 = human_zone_17 + 9
#     human_zone_19 = human_zone_18 + 9
#
#     total_zone = [list(human_zone_1),list(human_zone_2),list(human_zone_3),list(human_zone_4),list(human_zone_5),
#                   list(human_zone_6),list(human_zone_7),list(human_zone_8),list(human_zone_9),list(human_zone_10),
#                   list(human_zone_11),list(human_zone_12),list(human_zone_13),list(human_zone_14),list(human_zone_15),
#                   list(human_zone_16),list(human_zone_17),list(human_zone_18),list(human_zone_19)]
#
#     box_productivity = 26
#     pcs_productivity = box_productivity*3
#     working_hour = 7
#
#     sum_x = 0
#     human_zone_num = list()
#     for human_zone_x in total_zone:
#         sum_x = 0
#         for order in human_zone_x:
#             if order in node_num_dict:
#                 sum_x += node_num_dict[order]
#             else: pass
#         human_zone_num.append(sum_x)
#
#     needed_people = (sum(human_zone_num)/pcs_productivity)/working_hour
#     int_needed_people = math.ceil(needed_people)
#     people_limit = round(needed_people/int_needed_people,2)
#
#     print('Human Productivity(Box) : {}'.format(box_productivity))
#     print('Working Hour : {}'.format(working_hour))
#     print('Needed People : {}'.format(int_needed_people))
#     print('Limit : {}'.format(people_limit))
#
#     human_zone_num = np.array(human_zone_num)
#     human_zone_num = human_zone_num/(pcs_productivity * working_hour)
#
#     if mode == 1:
#         # Big Asile
#         human_zone = list()
#         initial = 0
#         people_capa = 0
#
#         for idx in range(len(human_zone_num)):
#             value = np.round(human_zone_num[idx],2)
#             people_capa += value
#             for _ in range(math.ceil(value)):
#                 if people_capa >= people_limit:
#                     people_capa -= people_limit
#                     saving_zone = sum(total_zone[initial:idx+1],[])
#                     human_zone.append(saving_zone)
#                     initial = idx
#
#                 if idx == len(human_zone)-1:
#                     saving_zone = sum(total_zone[initial:],[])
#                     human_zone.append(saving_zone)
#
#         return human_zone
#
#     elif mode == 2:
#         # Small asile
#         human_zone = list()
#         initial = 0
#         people_capa = 0
#         cnt = 0
#         small_asile_list = sorted(list(node_num_dict.keys()))
#
#         for key in small_asile_list:
#             value = round(node_num_dict[key]/(pcs_productivity*working_hour),2)
#             people_capa += value
#             cnt += 1
#
#             for i in range(math.ceil(value)):
#                 if people_capa >= people_limit:
#                     people_capa -= people_limit
#                     saving_zone = small_asile_list[initial:cnt+1]
#                     human_zone.append(saving_zone)
#                     initial = cnt
#
#                 if key == small_asile_list[-1]:
#                     saving_zone = small_asile_list[initial:]
#                     human_zone.append(saving_zone)
#
#         return human_zone



def human_batch(mode, input_order):
    node_num_dict = {}
    for order_list in input_order:
        for pcs_rack in order_list:
            pcs_rack = pcs_rack[1:]
            if pcs_rack[1] not in node_num_dict:
                node_num_dict[pcs_rack[1]] = pcs_rack[0]
            elif pcs_rack[1] in node_num_dict:
                node_num_dict[pcs_rack[1]] = node_num_dict[pcs_rack[1]] + pcs_rack[0]
            else:
                print('Error')
    #### zone_generate
    human_zone_1 = [4, 5, 6, 7, 8]
    human_zone_2 = np.array([13])
    human_zone_3 = human_zone_2 + 9
    human_zone_4 = human_zone_3 + 9
    human_zone_5 = [36, 38, 40]
    human_zone_6 = [45, 47, 49, 50]
    human_zone_7 = np.array([54, 56, 58, 59, 60])
    human_zone_8 = human_zone_7 + 9
    human_zone_9 = human_zone_8 + 9
    human_zone_10 = human_zone_9 + 9
    human_zone_11 = human_zone_10 + 9
    human_zone_12 = human_zone_11 + 9
    human_zone_13 = human_zone_12 + 9
    human_zone_14 = human_zone_13 + 9
    human_zone_15 = human_zone_14 + 9
    human_zone_16 = np.array([135, 137])
    human_zone_17 = human_zone_16 + 9
    human_zone_18 = human_zone_17 + 9
    human_zone_19 = human_zone_18 + 9

    total_zone = [list(human_zone_1), list(human_zone_2), list(human_zone_3), list(human_zone_4), list(human_zone_5),
                  list(human_zone_6), list(human_zone_7), list(human_zone_8), list(human_zone_9), list(human_zone_10),
                  list(human_zone_11), list(human_zone_12), list(human_zone_13), list(human_zone_14),
                  list(human_zone_15), list(human_zone_16), list(human_zone_17), list(human_zone_18),
                  list(human_zone_19)]

    box_productivity = 26
    pcs_productivity = box_productivity * 3
    working_hour = 7

    sum_x = 0
    human_zone_num = list()
    for human_zone_x in total_zone:
        sum_x = 0
        for order in human_zone_x:
            if order in node_num_dict:
                sum_x += node_num_dict[order]
            else:
                pass
        human_zone_num.append(sum_x)

    needed_people = (sum(human_zone_num) / pcs_productivity) / working_hour
    int_needed_people = math.ceil(needed_people)
    people_limit = round(needed_people / int_needed_people, 2)

    print('인시생산성(박스) : {}'.format(box_productivity))
    print('일하는 시간 : {}'.format(working_hour))
    print('필요 인원 : {}'.format(int_needed_people))
    print('제한 인원 : {}'.format(people_limit))

    human_zone_num = np.array(human_zone_num)
    human_zone_num = human_zone_num / (pcs_productivity * working_hour)

    if mode == 1:
        # Big asile
        human_zone = list()
        initial = 0
        people_capa = 0
        for idx in range(len(human_zone_num)):
            value = np.round(human_zone_num[idx], 2)
            people_capa += value
            for _ in range(math.ceil(value)):
                if people_capa >= people_limit:
                    people_capa -= people_limit
                    saving_zone = sum(total_zone[initial:idx + 1], [])
                    #             saving_zone = (total_zone[initial:idx+1])
                    human_zone.append(saving_zone)
                    initial = idx
                if idx == len(human_zone_num) - 1:
                    saving_zone = sum(total_zone[initial:], [])
                    human_zone.append(saving_zone)


        return human_zone



    elif mode == 2:
        # Small asile
        human_zone = list()
        initial = 0
        people_capa = 0
        cnt = 0
        small_asile_list = sorted(list(node_num_dict.keys()))

        for key in small_asile_list:
            value = round(node_num_dict[key] / (pcs_productivity * working_hour), 2)
            people_capa += value
            cnt += 1
            for i in range(math.ceil(value)):
                if people_capa >= people_limit:
                    people_capa -= people_limit
                    saving_zone = small_asile_list[initial:cnt + 1]
                    human_zone.append(saving_zone)
                    initial = cnt
                if key == small_asile_list[-1]:
                    saving_zone = small_asile_list[initial:]
                    human_zone.append(saving_zone)

        return human_zone









