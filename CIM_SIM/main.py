from hardware_unit import *

from sim import LayerAnalyzer
if __name__ == "__main__":
    # rram_cell = RRAM(30,  0.8)
    # print("RRAM_Computational Power:", rram_cell.get_computational_power())
    # print("RRAM_Area:", round(rram_cell.get_area(),2))
    # print("RRAM_Utilization:", rram_cell.get_utilization())
    #
    # sram_call = SRAM(30, 8192 , 0.8)
    # print("SRAM_Area:", round(sram_call.get_area(), 2))
    # print("SRAM_read_write_bandwidth:", sram_call.get_read_write_bandwidth())
    # print("SRAM_Utilization:", sram_call.get_utilization())
    # print("SRAM_Capacity:", round(sram_call.get_capacity(), 2))
    #
    # dram_cell = DRAM( 64, 2048)
    # print("DRAM_interface_Area:", dram_cell.get_area())
    # print("DRAM_/Write Bandwidth:", dram_cell.get_read_write_bandwidth())
    # print("DRAM_Utilization:", dram_cell.get_utilization())
    # print("DRAM_Capacity:", dram_cell.get_capacity())
    input=2048*1024
    hidden_dim = 2048
    prompt_len = 1024
    dtype_bytes = 1
    dmac_computational_power=16
    dmac_utilization=0.65
    rram_computational_power=30
    sram_capacity=32
    sram_read_write_bandwidth= 16
    sram_utilization=0.8
    dram_read_write_bandwidth=32
    n=LayerAnalyzer()
    n.prefill(input,hidden_dim,dtype_bytes,prompt_len,dmac_computational_power,dmac_utilization,rram_computational_power,
              sram_capacity,sram_read_write_bandwidth,sram_utilization,dram_read_write_bandwidth)
