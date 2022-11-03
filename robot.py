class robot:
    battery_capacity = 100 #percent
    battery_eff = 5 #amount of battery reduce per unit distance
    robot_name = "Test"
    def __init__(self, batcap, bat_eff, name):
        self.battery_capacity = batcap
        self.battery_eff = bat_eff
        self.robot_name = name
    def bid(self):
        bid = 1
        return bid