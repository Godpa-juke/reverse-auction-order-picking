import time
import numpy as np
import math

def modified_human_batch(
    mode,
    input_order,
    box_productivity_value,
    box_per_pcs_cnt_value,
    working_hour_value,
    current_people,
    verbose: bool = False,
):
    node_num_dict = {}
    for order_list in input_order:
        for pcs_rack in order_list:
            pcs_rack = pcs_rack[1:]
            if pcs_rack[1] not in node_num_dict:
                node_num_dict[pcs_rack[1]] = 1
            elif pcs_rack[1] in node_num_dict:
                node_num_dict[pcs_rack[1]] = node_num_dict[pcs_rack[1]] + 1
                # node_num_dict[pcs_rack[1]] = node_num_dict[pcs_rack[1]] + pcs_rack[0]
            else:
                print('Error')

    #### zone_generate
    # human_zone_1 = [0,1,2,3,4,5,6,7]
    # human_zone_2 = np.array([13])
    # human_zone_3 = human_zone_2 + 9
    # human_zone_4 = human_zone_3 + 9
    # human_zone_5 = [36, 38, 40]
    # human_zone_6 = [45, 47, 49, 50]
    # human_zone_7 = np.array([54, 56, 58, 59, 60])
    # human_zone_8 = human_zone_7 + 9
    # human_zone_9 = human_zone_8 + 9
    # human_zone_10 = human_zone_9 + 9
    # human_zone_11 = human_zone_10 + 9
    # human_zone_12 = human_zone_11 + 9
    # human_zone_13 = human_zone_12 + 9
    # human_zone_14 = human_zone_13 + 9
    # human_zone_15 = human_zone_14 + 9
    # human_zone_16 = np.array([135, 137])
    # human_zone_17 = human_zone_16 + 9
    # human_zone_18 = human_zone_17 + 9
    # human_zone_19 = human_zone_18 + 9

    # total_zone = [list(human_zone_1), list(human_zone_2), list(human_zone_3), list(human_zone_4), list(human_zone_5),
    #               list(human_zone_6), list(human_zone_7), list(human_zone_8), list(human_zone_9), list(human_zone_10),
    #               list(human_zone_11), list(human_zone_12), list(human_zone_13), list(human_zone_14),
    #               list(human_zone_15), list(human_zone_16), list(human_zone_17), list(human_zone_18),
    #               list(human_zone_19)]

    total_zone = list()
    for line_idx in range(0,36):
        total_zone.append([line_idx*8+x for x in range(8)])


    if verbose:
        print("###zone###")
        for zone in total_zone:
            print(zone)


    box_productivity = box_productivity_value
    pcs_productivity = box_productivity * box_per_pcs_cnt_value
    working_hour = working_hour_value

    sum_x = 0
    max_idx = 0
    cur_idx = 0
    max_value    = 0
    human_zone_num = list()

    for human_zone_x in total_zone:
        sum_x = 0
        for order in human_zone_x:
            if order in node_num_dict:
                sum_x += node_num_dict[order]
            else:
                pass

        if max_value < sum_x:
            max_value = sum_x
            max_idx = cur_idx
        human_zone_num.append(sum_x)
        cur_idx = cur_idx + 1

    needed_people = (sum(human_zone_num) / pcs_productivity) / working_hour
    int_needed_people = math.ceil(needed_people)
    people_limit = 1

    # NOTE: "필요 인원"은 현재 인력(current_people)이 아니라,
    # 주어진 생산성 가정(박스/시간)으로부터 역산한 '추정 필요 인원'입니다.
    if verbose:
        print('인시생산성(박스) : {}'.format(box_productivity))
        print('일하는 시간 : {}'.format(working_hour))
        print('필요 인원 : {}'.format(int_needed_people))
        print('제한 인원 : {}'.format(people_limit))

    human_ratio = (current_people / needed_people)
    human_zone_num = np.array(human_zone_num)
    human_zone_num = (human_zone_num*human_ratio) / (pcs_productivity * working_hour)
    if verbose:
        print(human_zone_num)


    if current_people >= sum(human_zone_num):
        # To do
        diff = abs(current_people - sum(human_zone_num))
        human_zone_num[max_idx] = human_zone_num[max_idx] + 2*diff

    if verbose:
        print("Current Zone Value : ", human_zone_num)
        print("sum value : ",sum(human_zone_num))

    if mode == 1:
        # Big asile
        human_zone = list()
        initial = 0
        people_capa = 0

        for idx in range(len(human_zone_num)):
            value = human_zone_num[idx]
            people_capa += value

            for _ in range(math.ceil(value)):
                if idx == len(human_zone_num) - 1:
                    saving_zone = sum(total_zone[initial:], [])
                    human_zone.append(saving_zone)
                    break

                elif people_capa >= people_limit:
                    people_capa -= people_limit
                    saving_zone = sum(total_zone[initial:idx + 1], [])
                    #             saving_zone = (total_zone[initial:idx+1])
                    human_zone.append(saving_zone)
                    initial = idx

        return human_zone


    elif mode == 2:
        # Small asile
        human_zone = list()
        initial = 0
        people_capa = 0
        cnt = 0
        small_asile_list = sorted(list(node_num_dict.keys()))

        for key in small_asile_list:
            value = round((node_num_dict[key]*human_ratio) / (pcs_productivity * working_hour), 2)
            people_capa += value
            cnt += 1
            for i in range(math.ceil(value)):
                if key == small_asile_list[-1]:
                    saving_zone = small_asile_list[initial:]
                    human_zone.append(saving_zone)
                    break

                if people_capa >= people_limit:
                    people_capa -= people_limit
                    saving_zone = small_asile_list[initial:cnt + 1]
                    human_zone.append(saving_zone)
                    initial = cnt


        return human_zone


