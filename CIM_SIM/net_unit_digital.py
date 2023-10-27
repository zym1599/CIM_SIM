from hardware_unit import *
from hardware_unit import *

def layer_Norm(input,hidden_dim,dtype_bytes, prompt_len,dmac_area):
    # 计算layer_Norm
    dmac=DMAC(area=dmac_area)
    sram=SRAM()
    input_size  = input
    output_size = input_size
    weights     = 0
    operation   = hidden_dim/2048*16384*prompt_len     #2048长度序列进行layer_Norm的操作数为16384
    comput_delay= operation/(dmac.get_computational_power()*1000000000000)/dmac.get_utilization()*1000
    read_sram_delay =input_size * dtype_bytes /sram.get_read_write_bandwidth()/sram.get_utilization()/1000000000*1000
    write_sram_delay=output_size * dtype_bytes /sram.get_read_write_bandwidth()/sram.get_utilization()/1000000000*1000
    area = dmac.get_area()+sram.get_area()

    read_dram_delay=write_dram_delay=0
    return input_size,output_size,weights,operation,comput_delay,read_sram_delay,write_sram_delay,read_dram_delay,write_dram_delay,area


def QKV(input,hidden_dim,dtype_bytes, prompt_len,rram_area):
    # 计算多头自注意力的计算量
    dmac = DMAC()
    sram = SRAM()
    dram = DRAM()
    input_size  = input
    output_size = hidden_dim * prompt_len * 3          #输出QKV三个矩阵
    weights     = hidden_dim * hidden_dim * 3                     #QKV三个矩阵所以乘以3
    operation   = (hidden_dim * hidden_dim * 2-hidden_dim) * prompt_len * 3     #操作数共计三个矩阵的
    comput_delay= operation / (dmac.get_computational_power()*1000000000000)/dmac.get_utilization()*1000
    read_sram_delay = input_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    write_sram_delay = output_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    write_dram_delay = hidden_dim * prompt_len * dtype_bytes * 2/dram.get_read_write_bandwidth()/1000000000/dram.get_utilization() * 1000
    read_dram_delay  = weights * dtype_bytes * 2/dram.get_read_write_bandwidth()/1000000000/dram.get_utilization() * 1000
    area = sram.get_area()
    return input_size,output_size,weights,operation,comput_delay,read_sram_delay,write_sram_delay,read_dram_delay,write_dram_delay,area


def QK(input,hidden_dim,dtype_bytes, prompt_len,dmac_area,mode):
    # 计算多头自注意力的计算量
    dmac=DMAC(area=dmac_area)
    sram=SRAM()
    dram=DRAM()

    if mode=='prefill':
        input_size = input
        output_size = prompt_len * prompt_len
        weights     = 0                                                             #QK转秩计算没有参量
        operation   = hidden_dim * 2 * prompt_len * prompt_len
        read_dram_delay = write_dram_delay = 0
    else:
        input_size = hidden_dim * prompt_len + hidden_dim * (prompt_len+1024)
        output_size = 1024+prompt_len                                               #1024是prefill阶段的kvcache
        weights = 0
        operation = hidden_dim * 2 * (prompt_len+1024)
        read_dram_delay = (1024+prompt_len)*hidden_dim*2*dtype_bytes/dram.get_read_write_bandwidth()/1000000000/dram.get_utilization()*1000

        write_dram_delay = 0



    comput_delay = operation/(dmac.get_computational_power()*1000000000000)/dmac.get_utilization()*1000
    read_sram_delay = input_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    write_sram_delay = output_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    area= dmac.get_area()+sram.get_area()
    return input_size,output_size,weights,operation,comput_delay,read_sram_delay,write_sram_delay,read_dram_delay,write_dram_delay,area


def Softmax(input,hidden_dim,dtype_bytes, prompt_len,dmac_area):
    # 计算多头自注意力的计算量
    dmac=DMAC(area=dmac_area)
    sram=SRAM()
    input_size  = input
    output_size = input_size
    weights     = 0
    operation   = (prompt_len+(prompt_len-1)+prompt_len)*prompt_len
    comput_delay= operation/(dmac.get_computational_power()*1000000000000)/dmac.get_utilization()*1000
    read_sram_delay = input_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    write_sram_delay = output_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    read_dram_delay = write_dram_delay = 0
    area = dmac.get_area()+sram.get_area()
    return input_size,output_size,weights,operation,comput_delay,read_sram_delay,write_sram_delay,read_dram_delay,write_dram_delay,area

def Mul(input,hidden_dim,dtype_bytes, prompt_len,dmac_area,mode):
    # 计算注意力得分QK乘以V的值
    dmac = DMAC(area=dmac_area)
    sram = SRAM()
    if mode == 'prefill':
        input_size  = input
        output_size = hidden_dim*prompt_len
        weights     = 0
        operation   = prompt_len * 2 * prompt_len * hidden_dim
        comput_delay= operation/(dmac.get_computational_power()*1000000000000)/dmac.get_utilization()*1000
    else:
        input_size  = (prompt_len+1024)+(prompt_len+1024)*hidden_dim
        output_size = hidden_dim*prompt_len
        weights     = 0
        operation   = (prompt_len+1024)*2*hidden_dim
        comput_delay = operation / (dmac.get_computational_power() * 1000000000000) / dmac.get_utilization() * 1000


    read_sram_delay = input_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    write_sram_delay = output_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    read_dram_delay = write_dram_delay = 0
    area = dmac.area + sram.get_area()
    return input_size,output_size,weights,operation,comput_delay,read_sram_delay,write_sram_delay,read_dram_delay,write_dram_delay,area


def Linear(input,Linear_in_dim,Linear_out_dim,hidden_dim,dtype_bytes, prompt_len,rram_area):
    # 计算全连接层
    dmac = DMAC()
    sram = SRAM()
    dram = DRAM()
    input_size  = input
    output_size = Linear_out_dim * prompt_len
    weights     = Linear_in_dim * Linear_out_dim
    operation   = Linear_in_dim * Linear_out_dim * 2 * prompt_len
    comput_delay = operation / (dmac.get_computational_power() * 1000000000000) / dmac.get_utilization() * 1000
    read_sram_delay = input_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    write_sram_delay = output_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000

    read_dram_delay = weights * dtype_bytes * 2/dram.get_read_write_bandwidth()/1000000000/dram.get_utilization() * 1000
    write_dram_delay = 0
    area = sram.get_area()
    return input_size,output_size,weights,operation,comput_delay,read_sram_delay,write_sram_delay,read_dram_delay,write_dram_delay,area


def Add(input,hidden_dim,dtype_bytes, prompt_len,dmac_area):
    # 计算残差连接
    dmac = DMAC(area=dmac_area)
    sram = SRAM()
    input_size = input * 2
    output_size = input_size / 2
    weights     = 0
    operation   = input_size / 2
    comput_delay = operation / (dmac.get_computational_power() * 1000000000000) / dmac.get_utilization() * 1000
    read_sram_delay = input_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    write_sram_delay = output_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    read_dram_delay = write_dram_delay = 0
    area = dmac.get_area()+sram.get_area()
    return input_size,output_size,weights,operation,comput_delay,read_sram_delay,write_sram_delay,read_dram_delay,write_dram_delay,area


def Relu(input,hidden_dim,dtype_bytes, prompt_len,dmac_area):


    dmac = DMAC(area=dmac_area)
    sram = SRAM()

    input_size = input
    output_size = input_size
    weights = 0
    operation = input_size
    comput_delay = operation / (dmac.get_computational_power() * 1000000000000) / dmac.get_utilization() * 1000
    read_sram_delay = input_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    write_sram_delay = output_size * dtype_bytes / sram.get_read_write_bandwidth() / sram.get_utilization() / 1000000000 * 1000
    read_dram_delay = write_dram_delay = 0
    area = dmac.get_area()+sram.get_area()

    return input_size, output_size, weights, operation,comput_delay,read_sram_delay,write_sram_delay,read_dram_delay,write_dram_delay ,area









