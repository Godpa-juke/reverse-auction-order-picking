import math
import queue

def EstimateMinWorkTime(AllofPcs, PRODUCTIVITY_FACTOR, WorkingHumanCnt, PcsPerBox):
    # AllofPcs  : all of piece count about a order wave
    # PRODUCTIVITY_FACTOR : amount of worked pcs alone at 1 hour
    # PcsPerBox : ratio between Pcs and Box
    # WorkingHumanCnt : count of working people

    # First Unit Change Box to Pcs
    # Second Calc amount of work per hour
    # Finally Divided All of Pcs by amount of work per hour

    CapaofHour = PRODUCTIVITY_FACTOR * PcsPerBox * WorkingHumanCnt
    return math.ceil(AllofPcs/CapaofHour)

def EstimateMinWorkHumanCnt(AllofPcs, PRODUCTIVITY_FACTOR, TimeLimit, PcsPerBox):
    # AllofPcs  : all of piece count about a order wave
    # PRODUCTIVITY_FACTOR : amount of worked pcs alone at 1 hour
    # PcsPerBox : ratio between Pcs and Box
    # TimeLimit : A deadline of Work

    # First unit Change Box to Pcs
    # Second calc amount of work in dead line
    # Third divided All of Pcs by amount of work per hour
    # Finally ceiling it

    CapaofPcs = PRODUCTIVITY_FACTOR * PcsPerBox * TimeLimit
    return math.ceil(AllofPcs/CapaofPcs)

def main():
    AllofPcsPerZone       = [345, 51, 10, 10, 188, 238, 274, 340, 345, 376, 371, 574, 442, 350, 386, 117, 147, 101, 132]
    AllofPcs = 0
    for idx in range(len(AllofPcsPerZone)): AllofPcs = AllofPcs + AllofPcsPerZone[idx]

    AllofPcs = 8153
    PRODUCTIVITY_FACTOR = 15
    PcsPerBox = 60

    set_people = 2
    set_time = 5

    recommended_people = EstimateMinWorkHumanCnt(AllofPcs, PRODUCTIVITY_FACTOR, set_time, PcsPerBox)
    recommended_time   = EstimateMinWorkTime(AllofPcs, PRODUCTIVITY_FACTOR, set_people, PcsPerBox)

    print(recommended_people)
    print(recommended_time)

if __name__ == '__main__':
    main()



