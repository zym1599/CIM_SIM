from hardware_unit import *
from hardware_unit import DMAC
from sim import LayerAnalyzer
import matplotlib.pyplot as plt

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
    input = 2048 * 1024
    hidden_dim = 2048
    prompt_len = 1024
    dtype_bytes = 1
    block_number = 1
    FFN_hidden_dim = 8192
    rram_area = 480
    dmac_area =619



    dmac_computational_power = 16
    rram_computational_power = 30
    sram_capacity = 32
    sram_read_write_bandwidth = 24
    sram_utilization = 0.8
    dram_read_write_bandwidth = 32
    n = LayerAnalyzer()
    delay_total=n.prefill(input=input, hidden_dim=hidden_dim, FFN_hidden_dim=FFN_hidden_dim, dtype_bytes=dtype_bytes,
              prompt_len=prompt_len, block_number=block_number,
              sram_capacity=sram_capacity, sram_read_write_bandwidth=sram_read_write_bandwidth,
              sram_utilization=sram_utilization, dram_read_write_bandwidth=dram_read_write_bandwidth
              , dmac_area=dmac_area,rram_area=rram_area)
    print(delay_total)
    # area_totals = []
    # prefill_outputs = []
    # compute_delay_val=100000
    # for i in range(1, 619):
    #     b=619-i
    #     compute_delay = n.prefill(input=input, hidden_dim=hidden_dim, FFN_hidden_dim=FFN_hidden_dim,
    #                               dtype_bytes=dtype_bytes,
    #                               prompt_len=prompt_len, block_number=block_number,
    #                               rram_area=i,dmac_area=b,
    #
    #                               sram_capacity=sram_capacity, sram_read_write_bandwidth=sram_read_write_bandwidth,
    #                               sram_utilization=sram_utilization,
    #                               dram_read_write_bandwidth=dram_read_write_bandwidth)
    #     if compute_delay_val>compute_delay:
    #         print('compute_delay:',compute_delay,'rram_area:',i,'dmac_area:',b)
    #         compute_delay_val=compute_delay


