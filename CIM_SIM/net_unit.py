from hardware_unit import *

def layer_Norm(input,hidden_dim,dtype_bytes, prompt_len,dcam_computational_power,  dcam_utilization):
    # 计算layer_Norm

    input_size  = input
    output_size = input_size
    weights     = 0
    operation   = hidden_dim/2048*16384*prompt_len     #2048长度序列进行layer_Norm的操作数为16384
    comput_delay= operation/(dcam_computational_power*1000000000000)/dcam_utilization*1000
    read_sram_delay=write_sram_delay=read_dram_delay=write_dram_delay=0
    return input_size,output_size,weights,operation,comput_delay,read_sram_delay,write_sram_delay,read_dram_delay,write_dram_delay


def QKV(input,hidden_dim,dtype_bytes, prompt_len):
    # 计算多头自注意力的计算量

    input_size  = input
    output_size = hidden_dim * prompt_len * 3          #输出QKV三个矩阵
    weights     = hidden_dim * hidden_dim * 3                     #QKV三个矩阵所以乘以3
    operation   = (hidden_dim * hidden_dim * 2-hidden_dim) * prompt_len * 3     #操作数共计三个矩阵的

    return input_size,output_size,weights,operation


def QK(input,hidden_dim,dtype_bytes, prompt_len):
    # 计算多头自注意力的计算量

    input_size  = input
    output_size = prompt_len * prompt_len
    weights     = 0                                                             #QK转秩计算没有参量
    operation   = hidden_dim * 2 * prompt_len * prompt_len  #操作数共计三个矩阵的

    return input_size,output_size,weights,operation


def Softmax(input,hidden_dim,dtype_bytes, prompt_len):
    # 计算多头自注意力的计算量

    input_size  = input
    output_size = input_size
    weights     = 0                                                             #QK转秩计算没有参量
    operation   = (prompt_len+(prompt_len-1)+prompt_len)*prompt_len  #操作数共计三个矩阵的

    return input_size,output_size,weights,operation

def Mul(input,hidden_dim,dtype_bytes, prompt_len):
    # 计算注意力得分QK乘以V的值

    input_size  = input
    output_size = hidden_dim*prompt_len
    weights     = 0
    operation   = prompt_len * 2 * prompt_len * hidden_dim #操作数共计三个矩阵的

    return input_size,output_size,weights,operation


def Linear(input,Linear_in_dim,Linear_out_dim,hidden_dim,dtype_bytes, prompt_len):
    # 计算全连接层

    input_size  = input
    output_size = Linear_out_dim * prompt_len
    weights     = Linear_in_dim * Linear_out_dim
    operation   = Linear_in_dim * Linear_out_dim * 2 * prompt_len

    return input_size,output_size,weights,operation


def Add(input,hidden_dim,dtype_bytes, prompt_len):
    # 计算残差连接

    input_size = input * 2
    output_size = input_size / 2
    weights     = 0
    operation   = input_size / 2

    return input_size,output_size,weights,operation


def Relu(input,hidden_dim,dtype_bytes, prompt_len):



    input_size = input
    output_size = input_size
    weights = 0
    operation = input_size

    return input_size, output_size, weights, operation









