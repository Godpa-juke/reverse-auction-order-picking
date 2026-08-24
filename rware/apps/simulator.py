"""Legacy simulator CLI extracted from the original rware.warehouse module.

This keeps the historical behaviour available under `python -m rware.apps.simulator`.
"""

import os
import random
import time
from datetime import datetime
from multiprocessing import Process
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from rware.algorithm.batch_sequence.batch_sequencing import *
from rware.algorithm.human_batch.human_batch import *
from rware.env import Warehouse
from rware.source.site_a.wrap import *
from rware.utils.Make_Maze import Make_Maze
from rware.utils.orders import order_gen, order_modi
from rware.engine.warehouse_engine import AgentCounter, WriteLog, RewardType
from rware.core import Action, State
from rware.data.cost_maps import load_cost_map
from rware.config import resolve_human_assignment_strategy
from rware.config import TARGET_TOTAL_SKU


routing_node_list = []


def main(strategy_override: Optional[str] = None):
    start_time_print = time.strftime('%Y.%m.%d - %H:%M:%S')
    start_time = time.time()
    # 맵 설정: rware/data/maps/*.map 파일에서 로드 (RWARE_MAP 환경변수로 교체 가능)
    from rware.config.defaults import MAP_FILE
    with open(MAP_FILE, "r", encoding="utf-8") as _f:
        map = _f.read()

    ############ robot & human information initialization ###########
    human_max_cnt, robot_max_cnt = AgentCounter(map)
    agent_max_cnt = human_max_cnt + robot_max_cnt

    human_id_list = [x for x in range(1, 8 + 1)]
    robot_id_list = [x+human_max_cnt for x in range(1, 20 + 1)]
    agent_id_list = human_id_list + robot_id_list

    human_cnt = len(human_id_list)
    robot_cnt = len(robot_id_list)
    agent_cnt = robot_cnt + human_cnt

    # warehouse initialization
    envInitSettingFlag = True
    env = Warehouse(5, 5, 3, agent_max_cnt, human_max_cnt, robot_max_cnt, 1, 1, 5, None, None, RewardType.GLOBAL,layout=map) # Warehouse 환경 객체 생성
    env.reset(initSettingFlag=envInitSettingFlag) # Warehouse 환경 초기화
    env.current_loaded_node_list = [[] for _ in range(agent_max_cnt)]

    env.human_id_list = human_id_list
    env.robot_id_list = robot_id_list
    env.agent_id_list = agent_id_list

    env.n_humans = human_cnt
    env.n_robots = robot_cnt
    env.n_agents = agent_cnt

    resolved_strategy = strategy_override or resolve_human_assignment_strategy()
    env.set_human_assignment_strategy(resolved_strategy)
    print(f"[Simulator] PID {os.getpid()} strategy='{env.human_assignment_strategy}'")

    cfg = env.config
    cfg.human_assignment_strategy = env.human_assignment_strategy
    render_enabled = bool(cfg.rendering)
    from rware.bridge.publisher import publisher_from_env
    bridge = publisher_from_env(env)
    if bridge:
        print(f"[Simulator] bridge publishing on {bridge.endpoint}")
    order_batch_flag = cfg.order_batch_strategy.value
    batch_sequence_flag = cfg.batch_sequence_strategy.value
    order_sequence_flag = cfg.order_sequence_flag
    robot_capacity = cfg.robot_capacity

    env.request_queue.clear()                                          # shelf request queue 초기화
    env.shelfs_goalpos_remapping()                                     # shelf goalposition remapping for blocking
    global routing_node_list
    routing_node_list, overlap = env.making_routing_node()             # Just Making for Zone
    env.making_routing_graph()


    # 초기 상태 렌더링 및 웨이팅
    if render_enabled:
        env.render()
    pre_actions = [np.array([Action.NOOP.value, 0], dtype='int64') for i in range(agent_max_cnt)]
    actions = [np.array([Action.NOOP.value, 0], dtype='int64') for i in range(agent_max_cnt)]

    total_pcs = 80000
    # divider = 500
    order_cnt = 4

    ############# Import Data #############
    # Get Order Data(WCS or M/W)
    if order_batch_flag == 0:
        input_order = list()
        POI = 185
        rack_num = 3408
        for _ in range(3333):
            order_list = order_gen(POI, rack_num, routing_node_list)
            order_list = order_modi(order_list, 55)
            input_order.append(order_list)

        order_80 = order_gen(80, 3408, routing_node_list)
        input_order.append(order_80)


    elif order_batch_flag in (1, 2):
        print("SiteA Batched Order")
        from rware.config.defaults import ORDER_DATE

        generate_order, A_rack_list, B_rack_list, C_rack_list = order_generate_site_a(
            routing_node_list, ORDER_DATE, robot_capacity
        )
        try:
            generate_order.drop('Unnamed: 0', axis=1, inplace=True)
        except:
            print('No Dummy Columns')
        generate_order = generate_order[['ORDERKEY', 'QTYPICKED', 'ORDERDATE', 'ADDHOUR', 'RACK', 'NODE', 'OLD_NODE', 'STR_SET_NODE']]
        input_order, seq_list = order_batch(generate_order, robot_capacity)

        if order_batch_flag == 2:
            seq_idx_order_list = list()
            input_idx_order_list = list()
            for idx, order in enumerate(seq_list):
                seq_idx_order_list.append([idx, order])
            for idx, order in enumerate(input_order):
                input_idx_order_list.append([idx, order])

            num_iter = 10
            # num_group = 100
            RACK_NUM = 1498
            lower = 0
            upper = RACK_NUM
            point = 2
            mutation_length = 1

            ###### order setting #####
            current_population = list()
            list_4_save = list()
            cnt = 0
            order_idx_list = []
            current_population_idx = list()
            for order in seq_idx_order_list:
                order_idx = [0 for _ in range(1498)]
                for rack in order[1]:
                    if order_idx[rack] != 1:
                        order_idx[rack] = 1
                    else:
                        pass
                order_idx_list.append(order[0])
                list_4_save.append(order_idx)
                cnt += 1
                if cnt == 4:
                    current_population.append(np.array(list_4_save))
                    current_population_idx.append(np.array(order_idx_list))
                    list_4_save = list()
                    order_idx_list = list()
                    cnt = 0

            if len(list_4_save) != 0:
                for num in range(4 - len(list_4_save)):
                    list_4_save.append([0 for _ in range(1498)])
                    #         order_idx_list.append( (len(current_population * 4) + len(list_4_save) + (num)  ))
                    order_idx_list.append(99999)
                current_population.append(np.array(list_4_save))
                current_population_idx.append(np.array(order_idx_list))
            current_population_idx = np.array(current_population_idx)
            current_population = np.array(current_population)

            current_population_main = current_population.copy()
            current_population_idx_main = current_population_idx.copy()

            total_batch_list = list()

            print(np.shape(current_population_idx_main))
            print(np.shape(current_population_main))
            # time.sleep(1000)

            for _ in range(len(current_population_idx)):

                num_group = len(current_population_main)
                num_parent = int(num_group / 2)

                best_idx = order_batch_GA(current_population_main, current_population_idx_main, num_iter, num_group,
                                          num_parent, point, RACK_NUM)
                print(best_idx)
                batch_compress = order_batch_compress(best_idx, input_idx_order_list, input_order)
                total_batch_list.append(batch_compress)

                flatten_idx = current_population_idx_main.flatten().tolist()
                shape = np.shape(current_population_main)
                flatten_order = current_population_main.reshape([shape[0] * shape[1], shape[2]]).tolist()

                del_order_list = list()
                del_idx_list = list()

                for idx in range(len(best_idx)):
                    order_np = np.where(current_population_idx_main == best_idx[idx])
                    del_order_list.append(order_np[0][0] * 4 + order_np[1][0])
                    index_np = np.where(flatten_idx == best_idx[idx])[0][0]
                    del_idx_list.append(index_np)

                #         print(index_np)
                #         flatten_idx = np.delete(flatten_idx, index_np)

                for idx in range(len(del_order_list)):
                    del_order_list = sorted(del_order_list, reverse=True)
                    del_idx_list = sorted(del_idx_list, reverse=True)
                    #         print(del_order_list)
                    #         print(flatten_order)
                    if del_order_list[idx] == 99999:
                        print('yee')
                        pass
                    else:
                        del flatten_order[del_order_list[idx]]
                        del flatten_idx[del_order_list[idx]]
                        # flatten_order.remove(flatten_order[del_order_list[idx]])
                        # flatten_idx.remove(flatten_idx[del_idx_list[idx]])
                #         print(f'Order : {len(flatten_order)}')
                #         print(f'Index : {len(flatten_idx)}')

                flatten_order = np.array([(list(flatten_order)[i:i + 4]) for i in range(0, len(flatten_order), 4)])
                flatten_idx = np.array([(list(flatten_idx)[i:i + 4]) for i in range(0, len(flatten_idx), 4)])
                current_population_main = flatten_order
                current_population_idx_main = flatten_idx

                print(f'Order : {np.shape(current_population_main)}')
                print(f'Index : {np.shape(current_population_idx_main)}')
                print(f'Remain : {len(flatten_order)} left')
                print()

                if len(flatten_order) == 3:
                    break

            for best_idx in current_population_idx_main:
                batch_compress = order_batch_compress(best_idx, input_idx_order_list, input_order)
                total_batch_list.append(batch_compress)

            print(np.shape(total_batch_list))
            print(np.shape(np.array(total_batch_list)))
            print(f'Order : {np.shape(current_population_main)}')
            print(f'Index : {np.shape(current_population_idx_main)}')

            input_order = total_batch_list

            seq_list = list()
            for seq_order in input_order:
                seq_save = list()
                for seq in seq_order:
                    seq_save.append(seq)
                seq_list.append(seq_save)


    # Order Sequencing
    cost_matrix = load_cost_map()
    # for idx in range(len(input_order)):
    #     input_order[idx], cost = aco_based_order_sequence(input_order[idx], cost_matrix)

    # Batch Sequencing
    result = basic_batch_sequencig(cost_matrix, input_order)

    if batch_sequence_flag == 0:
        print('No Batch Seq')
        pass

    elif batch_sequence_flag == 1:
        input_order = sorted(input_order, key=lambda x: len(x), reverse=False)
        print('Batch-short long')

    elif batch_sequence_flag == 2:
        input_order = sorted(input_order, key=lambda x: len(x), reverse=True)
        print('Batch-long short')

    elif batch_sequence_flag == 3:
        random.shuffle(input_order)
        print('Batch-Random')

    # random.shuffle(input_order)
    input_order_length = len(input_order)

    # Human Batch Set
    box_productivity_value =  1
    box_per_pcs_cnt_value  =  3
    working_hour_value     =  1

    def _normalize_working_area(area):
        default_nodes = list(range(len(env.routing_node_all_pos)))
        if not default_nodes:
            default_nodes = [0]

        target_count = max(1, env.n_max_humans or len(env.human_id_list) or 1)

        if not area:
            return [default_nodes.copy() for _ in range(target_count)]

        processed = []
        for nodes in area:
            if nodes:
                processed.append(list(nodes))
            else:
                processed.append(default_nodes.copy())

        if not processed:
            processed.append(default_nodes.copy())

        idx = 0
        while len(processed) < target_count:
            processed.append(list(processed[idx % len(processed)]))
            idx += 1

        return processed[:target_count]

    big_asile = modified_human_batch(
        1,
        input_order,
        box_productivity_value,
        box_per_pcs_cnt_value,
        working_hour_value,
        len(env.human_id_list),
        verbose=bool(cfg.verbose_zone),
    )
    small_asile = modified_human_batch(
        2,
        input_order,
        box_productivity_value,
        box_per_pcs_cnt_value,
        working_hour_value,
        len(env.human_id_list),
        verbose=bool(cfg.verbose_zone),
    )

    big_asile = _normalize_working_area(big_asile)
    small_asile = _normalize_working_area(small_asile)

    env.big_asile = [row.copy() for row in big_asile]
    env.small_asile = [row.copy() for row in small_asile]
    env.task_scheduler.big_asile = [row.copy() for row in big_asile]
    env.task_scheduler.small_asile = [row.copy() for row in small_asile]

    # Agent Init State Setting and Order Mapping
    for i in env.agent_id_list: env.agents[i-1].state = State.NOOP
    env.next_order_cnt = len(input_order)
    check_cnt = 0


    # 메인 루프
    while True:

        if render_enabled:
            env.render()
        env.request_queue.clear()
        human_map = Make_Maze(env,mode=1)
        robot_map = Make_Maze(env,mode=2)

        # 로봇 액션 갱신
        for id in env.agent_id_list:
            if id*2 > env.internal_timer+120: actions[id - 1] = np.array([Action.NOOP.value, 0], dtype='int64')
            else: actions[id-1] = env.agents[id-1].next_action(env, human_map, robot_map)


        # 로봇 액션 수행
        # When Step Action, Internal Counter Increase
        _, _, dones, step_info = env.step(actions)
        if bridge:
            bridge.publish()

        for id in env.agent_id_list: env.agents[id-1].check_status(env, input_order)
        env.next_order_cnt = len(input_order)

        # for id in env.agent_id_list: print(env.agents[id-1].id, env.agents[id-1].state, env.agents[id-1].node_list, actions[id-1], env.agents[id-1].load_box, env.agents[id-1].loadbox_station,env.agents[id-1].station)


        # Order Sequence
        # if   order_sequence_flag == 1: env.order_sequence()
        # elif order_sequence_flag == 2: pass

        # Agent Result Check
        # env.check_running_agent_cnt()

        # Exit Condition
        # 1) SKU 목표 달성  2) 엔진 종료 판정(모든 오더 소진 + 전 로봇 홈 복귀)
        termination_reason = None
        if env.all_of_completed_order >= TARGET_TOTAL_SKU:
            termination_reason = "target_total_sku"
        elif dones and all(dones):
            termination_reason = step_info.get("termination_reason", "engine_done")

        if termination_reason:
            end_time_print = time.strftime('%Y.%m.%d - %H:%M:%S')
            print(end_time_print)
            length_time = (time.time() - start_time)
            print('================end===================')
            print(f"[Simulator] termination_reason={termination_reason} "
                  f"completed_sku={env.all_of_completed_order}/{TARGET_TOTAL_SKU}")
            WriteLog(
                env,
                robot_max_cnt,
                human_max_cnt,
                length_time,
                start_time_print,
                end_time_print,
                simulation_name=getattr(env, "human_assignment_strategy", None),
            )
            # 완전 종료 (추가 입력 대기 없이 프로세스 종료)
            if bridge:
                bridge.close()
            return



def _launch_parallel(strategies: Iterable[str]) -> None:
    procs: List[Process] = []
    for name in strategies:
        worker = Process(target=main, args=(name.lower(),))
        worker.start()
        procs.append(worker)

    for proc in procs:
        proc.join()


def launch_from_config() -> None:
    strategy_list = resolve_human_assignment_strategy(as_list=True)
    if len(strategy_list) <= 1:
        main(strategy_list[0] if strategy_list else None)
    else:
        _launch_parallel(strategy_list)


if __name__ == '__main__':  # pragma: no cover
    launch_from_config()
