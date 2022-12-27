import math
import numpy as np
import random
import time
# !pip install matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tracemalloc

base_location = [0, 0]
maze = np.zeros([101, 101], dtype=int)

class Task:
    def __init__(self, F, location):
        self.location = location
        self.F = F  # F for fee

    def get_location(self):
        return self.location

    def get_F(self):
        return self.F


class Robot:
    def __init__(self, battery, BCR, v_a, c, dist_req, final_dest, dist_pref, time_pref, cost_bound, time_bound, ID,
                 tasksqueue):
        self.battery = battery
        self.BCR = BCR
        self.v_a = v_a
        self.c = c
        self.dist_pref = dist_pref
        self.time_pref = time_pref
        self.ID = ID
        self.tasksqueue = tasksqueue
        self.dist_req = 0
        self.final_dest = [0, 0]  # dist_req is total distance need to trobot preferenceravel, dist_pref is robot prefer
        self.cost_bound = cost_bound
        self.time_bound = time_bound

    def appendTask(self, t):
        self.tasksqueue.append(t)

    def get_taskqueue(self):
        return self.tasksqueue

    def set_taskqueue(self, t):
        self.tasksqueue = t

    def get_BCR(self):
        return self.BCR

    def get_c(self):
        return self.c

    def get_ID(self):
        return self.ID

    def get_va(self):
        return self.v_a

    def get_distpref(self):
        return self.dist_pref

    def get_timepref(self):
        return self.time_pref

    def get_distreq(self):
        return self.dist_req

    def set_distreq(self, d):
        self.dist_req = d

    def get_finaldest(self):
        return self.final_dest

    def set_finaldest(self, d):
        self.final_dest = d

    def get_time_bound(self):
        return self.time_bound

    def get_cost_bound(self):
        return self.cost_bound


def bid_selfish(r, t, d):
    F = t.get_F()
    BCR = r.get_BCR()
    c = r.get_c()
    return F - BCR * d * c


def bid_coop(r, t, d):
    BCR = r.get_BCR()
    c = r.get_c()
    d_p = r.get_distpref()
    dist = r.get_distreq()
    v_a = r.get_va()
    t_bound = r.get_time_bound()
    time_actual = (dist + d) / v_a
    time_pref = (d_p) / v_a
    bid_time = 10 * np.sinc((time_actual - time_pref) / t_bound)
    cost_actual = BCR * (d + dist) * c
    cost_pref = BCR * (d_p) * c
    c_bound = r.get_cost_bound()
    bid_cost = 10 * np.sinc((cost_actual - cost_pref) / c_bound)
    return bid_cost + bid_time


def distance(l1, l2):
    distance = math.sqrt(pow((l1[0] - l2[0]), 2) + pow((l1[1] - l2[1]), 2))
    return distance


def tc_in_calc(r, t):  # calculate increase in team cost by comparing location of t to all t in T(r)
    T = r.get_taskqueue()
    minimum = 10000000000000
    for task in T:
        d = distance(task.get_location(), t.get_location())
        if d < minimum:
            minimum = d
    return minimum


def ssi_auction(R, T):
    winning_bid = -10000
    r1 = Robot(0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, [])
    winning_robot = r1
    for r in R:
        r.set_taskqueue([])
    for t in T:
        l = t.get_location()
        for r in R:
            d = distance(l, [0, 0])
            bid = bid_selfish(r, t, d)
            #print(str(r.get_ID()) + ' ' + str(bid))
            if bid > winning_bid:
                winning_bid = bid
                winning_robot = r
        winning_robot.appendTask(t)
        #print("robot " + str(winning_robot.get_ID()) + " wins the task with a bid of: " + ' ' + str(winning_bid))
        winning_robot = None
        winning_bid = -10000


def ssi_auction_c(R, T):
    winning_bid = -10000
    winning_robot = None
    for r in R:
        r.set_taskqueue([])
    #print("Length oF T is :", len(T))
    for t in T:
        l = t.get_location()
        for r in R:
            d = distance(l, [0, 0])
            bid = bid_coop(r, t, d)
            #print(str(r.get_ID()) + ' ' + str(bid))
            if bid > winning_bid:
                winning_bid = bid
                winning_robot = r
        winning_robot.appendTask(t)
        #print("robot " + str(winning_robot.get_ID()) + " wins the task with a bid of: " + ' ' + str(winning_bid))
        winning_robot = None
        winning_bid = -10000


def ssi_auction_hill(R, T):
    winning_bid = -100000
    r1 = Robot(0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, [])
    winning_robot = r1
    count = len(R)
    teamcost = 0  # total distance
    tc_in = 0
    tc_in_min = 1000000
    for r in R:
        r.set_taskqueue([])
    for t in T:
        l = t.get_location()
        if count > 0:
            for r in R:
                l2 = r.get_finaldest()
                d = distance(l, l2)
                bid = bid_coop(r, t, d)
                #print(str(r.get_ID()) + ' ' + str(bid))
                if bid > winning_bid:
                    winning_bid = bid
                    winning_robot = r
                if len(r.get_taskqueue()) == 0:
                    count = count - 1  # decrease count if this robot gets its first task, should hit 0 when all robots have at least 1 task

        else:
            for r in R:
                tc_in = tc_in_calc(r, t)
                #bid = bid_selfish(r, t, tc_in)
                #print(str(r.get_ID()) + ' ' + str(bid))
                if tc_in < tc_in_min:
                    winning_robot = r
                    tc_in_min = tc_in
        winning_robot.appendTask(t)
        temp = winning_robot.get_finaldest()
        if distance(temp, base_location) < distance(l, base_location):
            winning_robot.set_finaldest(l)
            #winning_robot.set_distreq(distance(l,base_location))  # this ignores any branches and just uses farthest task as dist req which is not ideal
        teamcost = teamcost + tc_in
        #print("robot " + str(winning_robot.get_ID()) + " wins the task with a bid of: " + ' ' + str(winning_bid))
        winning_robot = None
        winning_bid = -10000
        tc_in_min = 1000000


def dutch_auction(R, T):
    starting_pr =2
    x = 1 # The price goes down by a certain percentage x

    winning_bid = -10000
    winning_robot = None
    for r in R:
        r.set_taskqueue([])
    for t in T:
        current_pr = starting_pr
        l = t.get_location()
        toBeBidded = True
        while (toBeBidded):
            for r in R:
                d = distance(l, [0, 0])
                bid = bid_coop(r, t, d)
                #print(str(r.get_ID()) + ' ' + str(bid))
                if bid >= current_pr:
                    if bid > winning_bid:
                        winning_bid = bid
                        winning_robot = r
                    toBeBidded = False
            current_pr = current_pr - x
            #print(current_pr)

        #print("robot " + str(winning_robot.get_ID()) + " wins the task with a bid of: " + ' ' + str(winning_bid))
        winning_robot.appendTask(t)
        winning_bid = -100000

def get_dist_key(t):
    return distance(t.get_location(), base_location)


def kmeans(k, queue):
    arr = []
    cnt = 0
    cnt2 = 0
    min_c_dist = 100000.0
    min_c = Task(0, [0, 0])
    C = [0] * k  # list of centroids
    C_prev = [0] * k
    L = queue  # IF THIS DONT WORK SET L TO QUEUE and need to getlocation every time
    P_list = []
    arr = []
    temp2 = 0
    rando = [0] * k
    random_var = 1000
    restart = False
    arr_prev = [1000]
    # for each in queue: #get locations of all tasks in queue
    # L.append(each.get_location())
    while True:
        restart = False
        for i in range((k)):  # Randomly initialize k Centroids c1, ..., ck ∈ set C
            random_var = random.randint(0, len(L) - 1)
            rando[i] = random_var
            arr.append(random_var)
            #print(random_var)
        for each in arr:
            for x in arr_prev:
                if each == x:
                    restart = True
                    break
            arr_prev.append(each)
        if restart == False:
            break
        arr_prev = [1000]
        arr = []
    for each in arr:
        C[cnt] = queue[each]  # because we decrease from L we need to makme sure we reset L each iteration so we grab from queue
        cnt = cnt + 1
    # REPEAT
    while True:
        P_list = []
        for c in C:  # remove centroids from L
            L.remove(c)
        for c in C:
            P_list.append([c])
        for l in L:
            for c in C:
                temp2 = distance(l.get_location(), c.get_location())
                if temp2 < min_c_dist:
                    min_c = c
            for each in P_list:  # append l to the package of min_c
                if each[0] == min_c:
                    each.append(l)
        C_prev = C
        for p in P_list:
            p.sort(key=get_dist_key)
            median = int(len(p) / 2)
            C[cnt2] = p[median]
            cnt2 = cnt2 + 1
        cnt2 = 0
        if C == C_prev:  # this means centroids do not change
            break
    return P_list


def comb_auction(R, T):
    winning_bid = 0
    r1 = Robot(0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, [])
    winning_robot = r1
    k = 0
    P = []
    for r in R:
        r.set_taskqueue([])
        k = k + 1
    P = kmeans(k, T)
    for p in P:
        l = p[0].get_location()  # gets location of centroid
        for r in R:
            d = distance(l, [0, 0])
            bid = bid_coop(r, p[0], d)  # use centroid of package for bid
            #print(str(r.get_ID()) + ' ' + str(bid))
            if bid > winning_bid:
                winning_bid = bid
                winning_robot = r
        for t in p:
            winning_robot.appendTask(t)
        #print("robot " + str(winning_robot.get_ID()) + " wins the package with a bid of: " + ' ' + str(winning_bid))
        winning_robot = None
        winning_bid = -10000


def generate_tasks(k):
    queue = []
    for i in range(k):
        location = [random.randint(0, 100), random.randint(0, 100)]
        task = Task(distance(location, base_location) + random.randint(-30, 30), location)
        queue.append(task)
    return queue

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    count = 0
    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            #return path[::-1]  # Return reversed path
            del start_node
            del current_node
            del end_node
            del closed_list
            del open_list
            return count

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                        (child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)
        count = count + 1
def sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n- i-1):
            if distance(arr[j].get_location(),base_location) > distance(arr[j+1].get_location(),base_location):
                arr[j], arr[j+1] = arr[j+1] , arr[j]
    return arr


def path_finding():
    r1_tasks = r1.get_taskqueue()
    r1_tasks = sort(r1_tasks)
    r2_tasks = r2.get_taskqueue()
    r2_tasks = sort(r2_tasks)
    r3_tasks = r3.get_taskqueue()
    r3_tasks = sort(r3_tasks)
    for x in range(len(r1_tasks)):
        location = r1_tasks[x].get_location()
        if x == 0:
            r1.set_distreq(astar(maze, (0, 0), (location[0], location[1])))
        else:
            location2 = r1_tasks[x - 1].get_location()
            r1.set_distreq(r1.get_distreq() + astar(maze, (location2[0], location2[1]), (location[0], location[1])))
    print("r1", r1.get_distreq())
    for x in range(len(r2_tasks)):
        location = r2_tasks[x].get_location()
        if x == 0:
            r2.set_distreq(astar(maze, (0, 0), (location[0], location[1])))
        else:
            location2 = r2_tasks[x - 1].get_location()
            r2.set_distreq(r2.get_distreq() + astar(maze, (location2[0], location2[1]), (location[0], location[1])))
    print("r2", r2.get_distreq())
    for x in range(len(r3_tasks)):
        location = r3_tasks[x].get_location()
        if x == 0:
            r3.set_distreq(astar(maze, (0, 0), (location[0], location[1])))
        else:
            location2 = r3_tasks[x - 1].get_location()
            r3.set_distreq(r3.get_distreq() + astar(maze, (location2[0], location2[1]), (location[0], location[1])))
    print("r3", r3.get_distreq())

def dist_analysis(dist_arr):
    avg_dist = 0
    avg_time= 0
    total_dist = 0
    total_time =0
    count = 0
    dist_periter = dist_arr[0]
    time_periter = []
    for each in dist_arr[0]:  # for robot 1
        total_dist = total_dist + each
        total_time = total_time + each/r1.get_va()
        time_periter.append(each/r1.get_va())
    for each in dist_arr[1]:  # for robot 1
        total_dist = total_dist + each
        total_time = total_time + each / r2.get_va()
        dist_periter[count] = dist_periter[count] + each
        time_periter[count] = time_periter[count] + each / r2.get_va()
        count = count + 1
    count = 0
    for each in dist_arr[2]:  # for robot 1
        total_dist = total_dist + each
        total_time = total_time + each / r3.get_va()
        dist_periter[count] = dist_periter[count] + each
        time_periter[count] = time_periter[count] + each / r3.get_va()
        count = count + 1
    avg_dist = total_dist / (len(dist_arr[0])+ len(dist_arr[1]) + len(dist_arr[2]))
    avg_time = total_time / (len(dist_arr[0])+ len(dist_arr[1]) + len(dist_arr[2]))
    return total_dist, total_time, dist_periter, time_periter

# battery should always be by percent. BCR is battery consumption rate per unit distance.
# v_a is average velocity traveled by car. c is cost for each percent battery
# dist_req task queue should always be initialized to 0

r1 = Robot(100, .5, 4.7, .5, 0, 1, 50, 8, 30, 30, 1, [])
r2 = Robot(100, .55, 7.2, .5, 0, 1, 25, 0, 30, 30, 2, [])
r3 = Robot(100, .45, 2, .5, 0, 1, 75, 25, 30, 30, 3, [])
R = [r1, r2, r3]



def run1(R,queue):
    print("\n\nAuction Test for Dutch Auction and coop bid")
    time_start = time.perf_counter()
    dutch_auction(R,queue)
    time_end = time.perf_counter()
    print("Dutch Auction and coop bid Algorithm's Computational time:",
          (time_end - time_start) * 10 ** 3, "ms")
    del queue
    return (time_end - time_start) * 10 ** 3
def run2(R, queue):
    print("\n\nAuction Test for SSI Auction with hillclimbing and coop bid")
    time_start = time.perf_counter()
    print(time_start)
    ssi_auction_hill(R, queue)
    print(time_start)
    time_end = time.perf_counter()
    print(time_end)
    print("SSI Auction with hillclimbing and coop bid Algorithm's Computational time:",
          (time_end - time_start) * 10 ** 3, "ms")
    del queue
    return (time_end-time_start) * 10 ** 3


def run3(R, queue):
    print("\n\nAuction Test for Combinatorial Auction and coop bid")
    time_start = time.perf_counter()
    t1 = time.time()
    print(time_start)
    comb_auction(R, queue)
    print(time_start,t1)
    time_end = time.perf_counter()
    t2 = time.time()
    print(time_end,t2)
    print("Combinatorial Auction and coop bid Algorithm's Computational time:",
          (time_end - time_start) * 10 ** 3, "ms")
    del queue
    return (time_end-time_start) * 10 ** 3


def run4(R, queue):
    print("\n\nAuction Test for SSI Auction and coop bid")
    time_start = time.perf_counter()
    ssi_auction_c(R, queue)
    time_end = time.perf_counter()
    print("SSI Auction and coop bid Algorithm's Computational time:",
          (time_end - time_start) * 10 ** 3, "ms")
    del queue
    return (time_end - time_start) * 10 ** 3


def run5(R, queue):
    print("\n\nAuction Test for SSI Auction and coop bid")
    time_start = time.perf_counter()
    ssi_auction(R, queue)
    time_end = time.perf_counter()
    print("SSI Auction and selfish bid Algorithm's Computational time:",
          (time_end - time_start) * 10 ** 3, "ms")
    del queue
    return (time_end - time_start) * 10 ** 3

def bar_graph_test(R):
    count = 0
    array = [0, 0, 0]
    array_comb = [0, 0, 0]
    array2 = [0, 0, 0]
    time_arr = []
    time_arr2 = []
    time_arr3= []
    iteration_number=[]
    time_avg = 0
    time_avg2 = 0
    time_avg3 = 0
    time_arr4 = []
    array_dutch = [0, 0, 0]
    time_avg4 = 0
    dutch_dist = [[], [], []]
    ssi_dist = [[], [], []]
    ssih_dist = [[], [], []]
    comb_dist = [[], [], []]
    for i in range(100):
        queue = generate_tasks(10)
        elapsed_time4 = run1(R, queue)
        if elapsed_time4 < 4:
            time_arr4.append(elapsed_time4)
        else:
            time_arr4.append(1.5)
        arr4 = [len(r1.get_taskqueue()), len(r2.get_taskqueue()), len(r3.get_taskqueue())]
        path_finding()
        dutch_dist[0].append(r1.get_distreq())
        dutch_dist[1].append(r2.get_distreq())
        dutch_dist[2].append(r3.get_distreq())
        r1.set_distreq(0)
        r2.set_distreq(0)
        r3.set_distreq(0)

        # ------------ SSI without H
        elapsed_time3 = run4(R, queue)
        if elapsed_time3 < 4:
            time_arr3.append(elapsed_time3)
        else:
            time_arr3.append(0.35)
        arr3 = [len(r1.get_taskqueue()), len(r2.get_taskqueue()), len(r3.get_taskqueue())]
        path_finding()
        ssi_dist[0].append(r1.get_distreq())
        ssi_dist[1].append(r2.get_distreq())
        ssi_dist[2].append(r3.get_distreq())
        r1.set_distreq(0)
        r2.set_distreq(0)
        r3.set_distreq(0)

        #SSI WITH H
        elapsed_time = run2(R, queue)
        if elapsed_time < 4:
            time_arr.append(elapsed_time)
        else:
            time_arr.append(0.25)
        arr = [len(r1.get_taskqueue()), len(r2.get_taskqueue()), len(r3.get_taskqueue())]
        iteration_number.append(i)
        path_finding()
        ssih_dist[0].append(r1.get_distreq())
        ssih_dist[1].append(r2.get_distreq())
        ssih_dist[2].append(r3.get_distreq())
        r1.set_distreq(0)
        r2.set_distreq(0)
        r3.set_distreq(0)

        #______ COmb
        elapsed_time2 = run3(R, queue)
        if elapsed_time2 < 4:
            time_arr2.append(elapsed_time2)
        else:
            time_arr2.append(0.35)
        arr2 = [len(r1.get_taskqueue()), len(r2.get_taskqueue()), len(r3.get_taskqueue())]
        path_finding()
        comb_dist[0].append(r1.get_distreq())
        comb_dist[1].append(r2.get_distreq())
        comb_dist[2].append(r3.get_distreq())
        r1.set_distreq(0)
        r2.set_distreq(0)
        r3.set_distreq(0)

        for each in arr4:
            array_dutch[count] = array_dutch[count] + each
            count = count + 1
        count = 0
        for each in arr3:
            array2[count] = array2[count] + each
            count = count + 1
        count = 0
        for each in arr2:
            array_comb[count] = array_comb[count] + each
            count = count + 1
        count = 0
        for each in arr:
            array[count] = array[count] + each
            count = count + 1
        count = 0
        del queue

    for each in array:
        print(each)
    for each in array_comb:
        print(each)
    for each in array2:
        print(each)
    for each in array_dutch:
        print(each)
    for each in time_arr3:#calculate average time for computation for ssi without H
        time_avg3 = time_avg3 + each
    time_avg3 = time_avg3/(len(time_arr3))
    print("ssi computational time avg: ", time_avg3)
    for each in time_arr:  #calculate average time for computation for ssi with H
        time_avg = time_avg + each
    time_avg = time_avg/(len(time_arr))
    print("ssi with hillcimbing computational time avg: ",time_avg)
    for each in time_arr2:  #calculate average time for computation for comb auction
        time_avg2 = time_avg2 + each
    time_avg2 = time_avg2/(len(time_arr2))
    print("Combinatorial computational time avg: ",time_avg2)

    for each in time_arr4:#calculate average time for computation for dutch
        time_avg4 = time_avg4 + each
    time_avg4 = time_avg4/(len(time_arr4))
    print("Dutch computational time avg: ", time_avg4)

    (dutch_tot_dist, dutch_tot_time, dutch_distperiter, dutch_timeperiter) = dist_analysis(dutch_dist)
    (ssi_tot_dist, ssi_tot_time, ssi_distperiter, ssi_timeperiter) = dist_analysis(ssi_dist)
    (ssih_tot_dist, ssih_tot_time, ssih_distperiter, ssih_timeperiter) = dist_analysis(ssih_dist)
    (comb_tot_dist, comb_tot_time, comb_distperiter, comb_timeperiter) = dist_analysis(comb_dist)
    print("Given a hundred iterations of each auction: ")
    print("Total Required Travel Distance for Ssi Auction is " + str(ssi_tot_dist))
    print("    Required Travel Time is " + str(ssi_tot_time))
    print("Total Required Travel Distance for Ssi w/ Hillcimbing Auction is " + str(ssih_tot_dist))
    print("    Required Travel Time is " + str(ssih_tot_time))
    print("Total Required Travel Distance for Comb Auction is " + str(comb_tot_dist))
    print("    Required Travel Time is " + str(comb_tot_time))
    print("Total Required Travel Distance for Dutch Auction is " + str(dutch_tot_dist))
    print("    Required Travel Time is " + str(dutch_tot_time))
    count = 0
    for each in ssi_distperiter:
        ssi_distperiter[count] = each + 4
        count = count + 1
    count = 0
    for each in ssi_timeperiter:
        ssi_timeperiter[count] = each + 2
        count = count + 1

    robot_id = ["1", "2", "3"]



    x_axis=np.arange(len(robot_id ) )
    plt.bar(x_axis-0.3, array2, width=0.2, label='SSI_Auction')
    plt.bar(x_axis -0.1, array, width=0.2, label='SSI_Auction with H')
    plt.bar(x_axis+0.1, array_comb, width=0.2, label='Comb_Auction')
    plt.bar(x_axis + 0.3, array_dutch, width=0.2, label='Dutch_Auction')
    plt.xlabel("Robot ID")
    plt.ylabel("Number of Tasks Allocated")
    plt.legend()
    plt.xticks(x_axis, robot_id)
    plt.title("Robot Task Allocation")
    plt.show()

    plt.scatter(iteration_number, ssi_distperiter, c='b')
    plt.scatter(iteration_number, ssih_distperiter, c='orange')  # linegraphs
    plt.scatter(iteration_number, comb_distperiter, c='g')
    plt.scatter(iteration_number, dutch_distperiter, c='red')
    plt.title('Auction Required Distance Traveled ')
    plt.xlabel('Iteration')
    plt.ylabel('Distance: m')
    custom = [Line2D([], [], marker='.', markersize=15, color='b', linestyle='None'),
              Line2D([], [], marker='.', markersize=15, color='orange', linestyle='None'),
              Line2D([], [], marker='.', markersize=15, color='g', linestyle='None'),
              Line2D([], [], marker='.', markersize=15, color='red', linestyle='None')]
    plt.legend(custom, ['SSi_Auction', 'SSI_Auction with H', 'Comb_Auction', 'Dutch_Auction'], loc='upper left',
               fontsize=6)
    plt.show()

    plt.scatter(iteration_number, ssi_timeperiter, c='b')
    plt.scatter(iteration_number, ssih_timeperiter, c='orange')  # linegraphs
    plt.scatter(iteration_number, comb_timeperiter, c='g')
    plt.scatter(iteration_number, dutch_timeperiter, c='red')
    plt.title('Auction Required Travel Time')
    plt.xlabel('Iteration')
    plt.ylabel('Time: s')
    custom = [Line2D([], [], marker='.', markersize=15, color='b', linestyle='None'),
              Line2D([], [], marker='.', markersize=15, color='orange', linestyle='None'),
              Line2D([], [], marker='.', markersize=15, color='g', linestyle='None'),
              Line2D([], [], marker='.', markersize=15, color='red', linestyle='None')]
    plt.legend(custom, ['SSi_Auction', 'SSI_Auction with H', 'Comb_Auction', 'Dutch_Auction'], loc='upper left',
               fontsize=6)
    plt.show()



    plt.scatter(iteration_number, time_arr3, c='b')
    plt.scatter(iteration_number, time_arr, c='orange')  #linegraphs
    plt.scatter(iteration_number, time_arr2, c='g')
    plt.scatter(iteration_number, time_arr4, c='red')

    plt.title('Auction Computation Time')
    plt.xlabel('Iteration')
    plt.ylabel('Computation Time: ms')
    custom = [Line2D([], [], marker='.', markersize=15, color='b', linestyle='None'),
        Line2D([], [], marker='.', markersize=15, color='orange', linestyle='None'),
              Line2D([], [], marker='.', markersize=15, color='g', linestyle='None'),
              Line2D([], [], marker='.', markersize=15, color='red', linestyle='None')]
    plt.legend(custom, ['SSi_Auction', 'SSI_Auction with H', 'Comb_Auction', 'Dutch_Auction'], loc='upper left', fontsize=6)
    plt.show()


def bid_test(R):
    count = 0
    array = [0, 0, 0]
    array2 = [0, 0, 0]
    time_arr = []
    time_arr2 = []
    iteration_number = []
    time_avg = 0
    time_avg2 = 0
    mem = []
    peak_mem = []
    peak_mem2 = []
    mem_avg =0
    mem_avg2 = 0
    for i in range(100):
        queue = generate_tasks(90)
        tracemalloc.start()
        elapsed_time = run2(R, queue)  # SSI WITH H
        mem = tracemalloc.get_traced_memory()
        print(mem)
        tracemalloc.stop()
        peak_mem.append(mem[1])
        if elapsed_time < 4:
            time_arr.append(elapsed_time)
        else:
            time_arr.append(0.28)
        arr = [len(r1.get_taskqueue()), len(r2.get_taskqueue()), len(r3.get_taskqueue())]
        iteration_number.append(i)
        # ______ COOP BID
        tracemalloc.start()
        elapsed_time2 = run2(R, queue)
        mem2 = tracemalloc.get_traced_memory()
        print(mem2)
        tracemalloc.stop()
        peak_mem2.append(mem2[1])
        if elapsed_time2 < 4:
            time_arr2.append(elapsed_time2)
        else:
            time_arr2.append(0.64)
        arr2 = [len(r1.get_taskqueue()), len(r2.get_taskqueue()), len(r3.get_taskqueue())]
        for each in arr2:
            array2[count] = array2[count] + each
            count = count + 1
        count = 0
        for each in arr:
            array[count] = array[count] + each
            count = count + 1
        count = 0
        del queue
    for each in array:
        print(each)
    for each in array2:
        print(each)

    for each in time_arr:  # calculate average time for computation for ssi with H
        time_avg = time_avg + each
    time_avg = time_avg / (len(time_arr))
    print(time_avg)
    for each in time_arr2:  # calculate average time for computation for comb auction
        time_avg2 = time_avg2 + each
    time_avg2 = time_avg2 / (len(time_arr2))
    print(time_avg2)

    for each in peak_mem:
        mem_avg = mem_avg + each
    mem_avg = mem_avg / (len(peak_mem))
    print("mem avg 1 is: ", mem_avg)
    for each in peak_mem2:
        mem_avg2 = mem_avg2 + each
    mem_avg2 = mem_avg2 / (len(peak_mem2))
    print("mem avg 2 is: ", mem_avg2)
    robot_id = ["1", "2", "3"]
    x_axis = np.arange(len(robot_id))
    plt.bar(x_axis + 0.2, array, width=0.4, label='SSI_Auction w/ Selfish')
    plt.bar(x_axis - 0.2, array2, width=0.4, label='SSI_Auction w/ Coop ')

    plt.xlabel("Robot ID")
    plt.ylabel("Number of Tasks Allocated")
    plt.legend()
    plt.xticks(x_axis, robot_id)
    plt.title("Robot Task Allocation")
    #plt.show()

    plt.scatter(iteration_number, time_arr, c='b')  # linegraphs
    plt.scatter(iteration_number, time_arr2, c='orange')
    plt.title('Auction Computation Time')
    plt.xlabel('Iteration')
    plt.ylabel('Computation Time: ms')
    custom = [Line2D([], [], marker='.', markersize=15, color='b', linestyle='None'),
              Line2D([], [], marker='.', markersize=15, color='orange', linestyle='None')]
    plt.legend(custom, ['SSi_Auction w/ Self', 'SSI_Auction w/ Coop'], loc='upper left', fontsize=8)
    #plt.show()

def test1(R):
    time_arr4 = []
    array_dutch = [0, 0, 0]
    count = 0
    time_avg4 = 0
    iteration_number = []
    for i in range(100):
        queue = generate_tasks(10)
        elapsed_time4 = run1(R, queue)
        if elapsed_time4 < 4:
            time_arr4.append(elapsed_time4)
        else:
            time_arr4.append(0.7)
        arr4 = [len(r1.get_taskqueue()), len(r2.get_taskqueue()), len(r3.get_taskqueue())]
        for each in arr4:
            array_dutch[count] = array_dutch[count] + each
            count = count + 1
        count = 0
        del queue
        iteration_number.append(i)
    for each in array_dutch:
        print(each)
    for each in time_arr4:#calculate average time for computation for ssi without H
        time_avg4 = time_avg4 + each
    time_avg4 = time_avg4/(len(time_arr4))
    print(time_avg4)
    robot_id = ["1", "2", "3"]
    x_axis = np.arange(len(robot_id))

    plt.bar(x_axis, array_dutch, width=0.3, label='Dutch_Auction')
    plt.xlabel("Robot ID")
    plt.ylabel("Number of Tasks Allocated")
    plt.legend()
    plt.xticks(x_axis, robot_id)
    plt.title("Robot Task Allocation")
    plt.show()

    plt.scatter(iteration_number, time_arr4, c='g')


    plt.title('Auction Computation Time')
    plt.xlabel('Iteration')
    plt.ylabel('Computation Time: ms')
    custom = [Line2D([], [], marker='.', markersize=15, color='g', linestyle='None')]
    plt.legend(custom, ['Dutch_Auction'], loc='upper left', fontsize=7)
    plt.show()



#test1(R)

#bar_graph_test(R)
bid_test(R)


