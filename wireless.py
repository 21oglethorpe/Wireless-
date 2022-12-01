import math
import numpy as np

print('Hello World')
base_location = [0, 0]


class Task:
    def __init__(self, F, location):
        self.location = location
        self.F = F  # F for fee

    def get_location(self):
        return self.location

    def get_F(self):
        return self.F


class Robot:
    def __init__(self, battery, BCR, v_a, c, dist_req, final_dest, dist_pref, cost_bound, time_bound, ID, tasksqueue):
        self.battery = battery
        self.BCR = BCR
        self.v_a = v_a
        self.c = c
        self.ID = ID
        self.tasksqueue = tasksqueue
        self.dist_req = 0
        self.final_dest = [0, 0]  # dist_req is total distance need to trobot preferenceravel, dist_pref is robot preer
        self.cost_bound = cost_bound
        self.time_bound = time_bound
    def appendTask(self, t):
        self.tasksqueue.append(t)

    def get_taskqueue():
        return self.tasksqueue

    def set_taskqueue(self, t):
        self.tasksqueue = t

    def get_BCR(self):
        return self.BCR

    def get_c(self):
        return self.c

    def get_ID(self):
        return self.ID

    def get_va():
        return self.v_a

    def get_distpref():
        return self.dist_pref

    def get_distreq():
        return self.dist_req

    def set_distreq(self, d):
        self.dist_req = d

    def get_finaldest():
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
    dist = r.get_distreq
    v_a = r.get_va()
    t_bound = r.get_time_bound()
    time_actual = v_a * (dist + d)
    time_pref = v_a * (dist + d_p)
    bid_time = 10 * np.sinc(time_actual/t_bound-time_pref)
    cost_actual = BCR*(d+dist)*c
    cost_pref = BCR*(d+d_p)*c
    c_bound = r.get_cost_bound
    bid_cost = 10 * np.sinc(cost_actual/c_bound-cost_pref)
    return bid_cost + bid_time
def distance(l1, l2):
    distance = math.sqrt(pow((l1[0] - l2[0]), 2) + pow((l1[1] - l2[1]), 2))
    return distance


def tc_in_calc(self, r, t):  # calculate increase in team cost by comparing location of t to all t in T(r)
    T = r.get_taskqueue()
    minimum = 10000000000000;
    for task in T:
        d = distance(task.get_location(), t.get_location())
        if d < minimum:
            minimum = d
    return minimum


def ssi_auction(R, T):
    winning_bid = 0
    r1 = Robot(0, 0, 0, 0, 0, 0, 0, 1, 1, 0, [])
    winning_robot = r1
    for r in R:
        r.set_taskqueue([])
    for t in T:
        l = t.get_location()
        for r in R:
            d = distance(l, [0, 0])
            bid = bid_selfish(r, t, d)
            print(bid)
            if bid > winning_bid:
                winning_bid = bid
                winning_robot = r
        winning_robot.appendTask(t)
        winning_robot = None
        winning_bid = 0


def ssi_auction_hill(R, T):
    winning_bid = 0
    r1 = Robot(0, 0, 0, 0, 0, 0, 0, 0, [])
    winning_robot = r1
    count = length(R)
    teamcost = 0  # total distance
    tc_in = 0;
    tc_in_min = 0;
    for r in R:
        r.set_tasksqueue([])
    for t in T:
        l = t.get_location()
        if count > 0:
            for r in R:
                l2 = r.get_finaldest()
                d = distance(l, l2)
                bid = bid_selfish(r, t, d)
                if bid > winning_bid:
                    winning_bid = bid
                    winning_robot = r
                if r.get_tasksqueue() == 0:
                    count = count - 1  # decrease count if this robot gets its first task, should hit 0 when all robots have at least 1 task

        else:
            for r in R:
                tc_in = tc_in_calc(r, t)
                bid = bid_selfish(r, t, tc_in)
                if tc_in < tc_in_min:
                    winning_robot = r
        winning_robot.appendTask(t)
        temp = winning_robot.get_finaldest()
        if distance(temp, base_location) < distance(l, base_location):
            winning_robot.set_finaldest(l)
            winning_robot.set_distreq(distance(l,
                                               base_location))  # this ignores any branches and just uses farthest task as dist req which is not ideal
        teamcost = teamcost + tc_in
        winning_robot = None
        winning_bid = 0
        tc_in_min = 0


def get_dist_key(t):
    return distance(t.get_location(), base_location)


def kmeans(k, queue):
    arr = []
    cnt = 0
    cnt2 = 0
    min_c = 1000000000000000000
    C = []  # list of centroids
    C_prev = []
    L = queue  # IF THIS DONT WORK SET L TO QUEUE and need to getlocation every time
    P_list = []
    # for each in queue: #get locations of all tasks in queue
    # L.append(each.get_location())

    for i in range(k - 1):  # Randomly initialize k Centroids c1, ..., ck ∈ set C
        arr[i] = randint(0, len(L) - 1)
    for each in arr:
        C[cnt] = queue[
            each]  # because we decrease from L we need to makme sure we reset L each iteration so we grab from queue
        cnt = cnt + 1
    # REPEAT
    while True:
        for c in C:  # remove centroids from L
            L.remove(c)
        for i in range(k - 1):
            P_list.append([c])
        for l in L:
            for c in C:
                if distance(l.get_location(), c.get_location()) < min_c:
                    min_c = c
            for each in P_list:  # append l to the package of min_c
                if each[0] == min_c:
                    each.append(l)
        C_prev = C
        for p in P:
            p.sort(key=get_dist_key)
            median = p[len(p) / 2]
            C[cnt2] = p[median]
            cnt2 = cnt2 + 1
        cnt2 = 0
        if C == C_prev:  # this means centroids do not change
            break
    return P_list


def comb_auction(R, T):
    winning_bid = 0
    r1 = Robot(0, 0, 0, 0, 0, 0, 0, 0, [])
    winning_robot = r1
    for r in R:
        r.set_taskqueue([])
    for p in P:
        l = p[0].get_location()  # gets location of centroid
        for r in R:
            d = distance(l, [0, 0])
            bid = bid_selfish(r, p[0], d)  # use centroid of package for bid
            print(bid)
            if bid > winning_bid:
                winning_bid = bid
                winning_robot = r
        for t in p:
            winning_robot.appendTask(t)
        winning_robot = None
        winning_bid = 0


r = Robot(50, 1, 1, 1, 1, 1, 1, 1, 1, 1, [])
t1 = Task(10, [0, 0])
t2 = Task(15, [1, 2])
t3 = Task(20, [3, 4])
queue = [t1, t2, t3]
ssi_auction([r], queue)
