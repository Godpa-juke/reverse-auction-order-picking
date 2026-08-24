# Time Complexity는 H에 따라 다르다.
# O(b^d), where d = depth, b = 각 노드의 하위 요소 수
# heapque를 이용하면 길을 출력할 때 reverse를 안해도 됨
import time
import random
class Node:
    def __init__(self, parent=None, position=None, angle=None, action=None):
        self.parent = parent
        self.position = position
        self.angle = angle
        self.Action = action

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return (self.position == other.position) & (self.angle == other.angle)


def heuristic(node, goal, D=1, A=1):  # Diagonal Distance
    dx = abs(node.position[0] - goal.position[0])
    dy = abs(node.position[1] - goal.position[1])
    summation = (dx + dy)
    return D * (dx + dy)
    # if dx and dy : return D*1.4*(summation)
    # else: return D * (dx + dy)


def aStar(maze, start, end, startAngle, seed_param):
    # Limit Time
    prev = time.time()

    # startNode와 endNode 초기화
    startNode = Node(None, start, startAngle)
    endNode = Node(None, end)

    # openList, closedList 초기화
    openList = []
    closedList = []

    # openList에 시작 노드 추가
    openList.append(startNode)

    # endNode를 찾을 때까지 실행
    while openList:
        looptime = time.time()


        # 현재 노드 지정
        currentNode = openList[0]
        currentIdx = 0

        # # 시간 제한
        short_path = []
        if looptime-prev > 0.02:
            next = currentNode
            short_path.append([next.position, next.angle, next.Action])
            return short_path

        # 이미 같은 노드가 openList에 있고, f 값이 더 크면
        # currentNode를 openList안에 있는 값으로 교체
        for index, item  in enumerate(openList):
            if item.f < currentNode.f:
                currentNode = item
                currentIdx = index

        # openList에서 제거하고 closedList에 추가
        openList.pop(currentIdx)
        closedList.append(currentNode)

        # 현재 노드가 목적지면 current.position 추가하고
        # current의 부모로 이동
        if currentNode.position == endNode.position:
            path = []
            current = currentNode
            while current is not None:
                # maze 길을 표시하려면 주석 해제
                # x, y = current.position
                # maze[x][y] = 7
                path.append([current.position, current.angle, current.Action])
                current = current.parent

                looptime = time.time()

                # # 시간 제한
                if looptime - prev > 0.02:
                    next = currentNode
                    return path[::-1]

            # print("-------------ASTAR--------------")
            # print("path : "+str(path[::-1]))
            return path[::-1]  # reverse

        children = []

        checkPosition = []
        # checkPosition 순서가 Y변화, X변화, Direction, Action 순서임
        # if currentNode.angle == 0:
        #     checkPosition = [(-1, 0, 0, 1), (1, 0, 0, 6), (0, 0, 2, 2), (0, 0, 3, 3)]
        # elif currentNode.angle == 1:
        #     checkPosition = [(1, 0, 1, 1), (-1, 0, 1, 6), (0, 0, 3, 2), (0, 0, 2, 3)]
        # elif currentNode.angle == 2:
        #     checkPosition = [(0, -1, 2, 1), (0, 1, 2, 6), (0, 0, 1, 2), (0, 0, 0, 3)]
        # elif currentNode.angle == 3:
        #     checkPosition = [(0, 1, 3, 1), (0, -1, 3, 6), (0, 0, 0, 2), (0, 0, 1, 3)]

        # checkPosition 순서가 Y변화, X변화  순서임
        checkPosition = [(-1,0,0,0),(-1,1,1,1),(0,1,2,2),(1,1,3,3),(1,0,4,4),(1,-1,5,5),(0,-1,6,6),(-1,-1,7,7)]

        # 인접한 xy좌표 전부
        for newPosition in checkPosition:

            # 노드 위치 업데이트
            nodePosition = (
                currentNode.position[0] + newPosition[0],  # X
                currentNode.position[1] + newPosition[1])  # Y

            nodeAngle = newPosition[2]

            nodeAction = newPosition[3]

            # 미로 maze index 범위 안에 있어야함
            within_range_criteria = [
                nodePosition[0] > (len(maze) - 1),
                nodePosition[0] < 0,
                nodePosition[1] > (len(maze[len(maze) - 1]) - 1),
                nodePosition[1] < 0,
            ]

            if any(within_range_criteria):  # 하나라도 true면 범위 밖임
                continue

            # 장애물이 있으면 다른 위치 불러오기
            if maze[nodePosition[0]][nodePosition[1]] != 0:
                continue

            new_node = Node(currentNode, nodePosition, nodeAngle, nodeAction)
            children.append(new_node)


        # 자식들 모두 loop
        for child in children:

            # 자식이 closedList에 있으면 continue
            if child in closedList:
                continue

            interval = 3
            param = 1+(seed_param/interval)%8
            # f, g, h값 업데이트
            child.g = currentNode.g + 1
            # child.h = heuristic(child, endNode, random.choice([1,2,3,4,5,6,7,8,9,10]))
            child.h = heuristic(child, endNode, param)
            # child.h = heuristic(child, endNode, random.choice([1,2]))

            # print("position:", child.position) 거리 추정 값 보기
            # print("from child to goal:", child.h)
            child.f = child.g + child.h

            # 자식이 openList에 있으고, g값이 더 크면 continue
            if len([openNode for openNode in openList
                    if child == openNode and child.g > openNode.g]) > 0:
                continue

            openList.append(child)


def main():
    # 1은 장애물
    maze = [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # startAngle은 class Direction(Enum)값을 참조
    start = (0, 0)
    startAngle = 4
    end = (7, 7)

    path = aStar(maze, start, end, startAngle,1)

    for i in range(len(maze)):
        print(maze[i])

    for i in range(len(path)):
        print(path[i],end=' ')
        if ((i+1)%5) == 0: print()


if __name__ == '__main__':
    main()
    # [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (5, 4), (6, 5), (7, 6)]
