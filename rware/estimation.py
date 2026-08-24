import math
import queue

def EstimateMinWorkTime(AllofPcs, PRODUCTIVITY_FACTOR, WorkingHumanCnt, PcsPerBox):
    # AllofPcs  : all of piece count about a order wave
    # PcsPerBox : ratio between Pcs and Box
    # PRODUCTIVITY_FACTOR : amount of worked pcs alone at 1hour
    # WorkingHumanCnt : count of working people

    # First Unit Change Box to Pcs
    # Second Calc amount of work per hour
    # Finally Divided All of Pcs by amount of work per hour
    return (AllofPcs/((PRODUCTIVITY_FACTOR * PcsPerBox) * WorkingHumanCnt))


def EstimateMinWorkHumanCnt(AllofPcs, PRODUCTIVITY_FACTOR, TimeLimit, PcsPerBox):
    # AllofPcs  : all of piece count about a order wave
    # PcsPerBox : ratio between Pcs and Box
    # PRODUCTIVITY_FACTOR : amount of worked pcs alone at 1hour
    # TimeLimit : A deadline of Work

    # First unit Change Box to Pcs
    # Second calc amount of work in dead line
    # Third divided All of Pcs by amount of work per hour
    # Finally ceiling it
    return (AllofPcs/((PRODUCTIVITY_FACTOR * PcsPerBox) * TimeLimit))



def RecommendPeopleWorkArea(AllofPcsPerZone, PRODUCTIVITY_FACTOR, Merge, WorkingHumanCnt, TimeLimit, PcsPerBox):
    # We must change Queue -> Deque

    totalRatio = 0

    divider = (PRODUCTIVITY_FACTOR * PcsPerBox) * TimeLimit
    people = queue.Queue()
    for id in range(WorkingHumanCnt): people.put([id, []])

    # Calc how many people needed in a zone
    for idx in range(len(AllofPcsPerZone)):
        AllofPcsPerZone[idx] = AllofPcsPerZone[idx]
        totalRatio += AllofPcsPerZone[idx]

    ratio_divider = WorkingHumanCnt/totalRatio
    print("total_ratio: ",totalRatio)
    print("ratio_divider : ",ratio_divider)
    # Calc how many people needed in a zone actually
    for idx in range(len(AllofPcsPerZone)):
        AllofPcsPerZone[idx] = ratio_divider * AllofPcsPerZone[idx]
        print(idx, AllofPcsPerZone[idx])


    # Merge
    result = [[] for _ in range(WorkingHumanCnt)]

    start  = 0
    work_list = [0 for x in range(WorkingHumanCnt)]

    # 0: selected zone ,1: unify empty zone
    if Merge == 0:
        if WorkingHumanCnt <= 0: return []
        for work_idx in range(WorkingHumanCnt):
            for idx in range(len(AllofPcsPerZone)):
                if work_list[work_idx] >= 1: break
                else:
                    if AllofPcsPerZone[idx] >= 1:

                        work_list[work_idx]  = work_list[work_idx]
                        AllofPcsPerZone[idx] = AllofPcsPerZone[idx] - 1
                        break

                    else:

                        AllofPcsPerZone[idx] = 0



    return result




def main():
    AllofPcsPerZone = [0.2,1.3,2.1,0.7,0.8,2.2,3.1,0.7,2.1,0.8]

    result=RecommendPeopleWorkArea(AllofPcsPerZone=AllofPcsPerZone, PRODUCTIVITY_FACTOR=15, Merge=0, WorkingHumanCnt=4, TimeLimit=1, PcsPerBox=60)
    print(result)

if __name__ == '__main__':
    main()



