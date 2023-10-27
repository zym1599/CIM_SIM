class RRAM:
    def __init__(self, area):

        self.computational_power = area/500*30
        self.area = area  #30TOPS;256MB的RRAM面积是500MM2
        # self.utilization = utilization

    def set_computational_power(self, computational_power):
        self.computational_power = computational_power

    def get_computational_power(self):

        return self.computational_power

    def set_area(self, area):
        self.area = area

    def get_area(self):
        return self.area

    def set_utilization(self, utilization):
        self.utilization = utilization

    def get_utilization(self,weight, weight_total=256*1024*1024):
        self.utilization = weight/weight_total
        return self.utilization

class DMAC:
    def __init__(self, area=40,  utilization=0.20):
        self.computational_power = area*0.25
        self.area = area  #dmac算力面积关系，每0.25
        self.utilization = utilization

    def set_computational_power(self, computational_power):
        self.computational_power = computational_power


    def get_computational_power(self):
        return self.computational_power

    def set_area(self, area):
        self.area = area

    def get_area(self):
        return self.area

    def set_utilization(self, utilization):
        self.utilization = utilization

    def get_utilization(self):
        return self.utilization

class SRAM:
    def __init__(self, capacity=32,read_write_bandwidth=8192,utilization=0.8):
        self.area = 1.652944*capacity     #1.652944是1MB容量SRAM的面积
        self.read_write_bandwidth = read_write_bandwidth
        self.capacity = capacity
        self.utilization = utilization

    def set_area(self, area):
        self.area = area

    def get_area(self):
        return self.area

    def set_read_write_bandwidth(self, read_write_bandwidth):
        self.read_write_bandwidth = read_write_bandwidth

    def get_read_write_bandwidth(self):
        return self.read_write_bandwidth

    def set_capacity(self, capacity):
        self.capacity = capacity

    def get_capacity(self):
        return self.capacity

    def set_utilization(self, utilization):
        self.utilization = utilization

    def get_utilization(self):
        return self.utilization



class DRAM:
    def __init__(self, read_write_bandwidth=38.4, capacity=900000, utilization=0.8):
        self.utilization = utilization
        self.area = read_write_bandwidth/19.2*32      #一组19GByte/s的ddr interface面积为32(mm2)
        self.read_write_bandwidth = read_write_bandwidth
        self.capacity = capacity

    def set_area(self, area):
        self.area = area

    def get_area(self):
        return self.area

    def set_read_write_bandwidth(self, read_write_bandwidth):
        self.read_write_bandwidth = read_write_bandwidth

    def get_read_write_bandwidth(self):
        return self.read_write_bandwidth

    def set_capacity(self, capacity):
        self.capacity = capacity

    def get_capacity(self):
        return self.capacity

    def set_utilization(self, utilization):
        self.utilization = utilization

    def get_utilization(self):
        return self.utilization





