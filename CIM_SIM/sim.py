from net_unit import *
from hardware_unit import *





class LayerAnalyzer:
    def __init__(self):
        pass
        pass
        pass
        pass
        pass

    def prefill(self,
                input,
                hidden_dim,
                dtype_bytes,
                prompt_len,
                dmac_computational_power,
                dmac_utilization,
                rram_computational_power,
                sram_capacity,
                sram_read_write_bandwidth,
                sram_utilization,
                dram_read_write_bandwidth,
                dram_capacity=900000,
                dram_utilization=0.8
                ):

        total_delay=0

        layernorm1=layer_Norm(input, hidden_dim, dtype_bytes, prompt_len,dmac_computational_power,dmac_utilization)
        #print(layernorm)

        print('layernorm:')
        print("input_size:", layernorm1[0],"output_size:", layernorm1[1],"weight:", layernorm1[2],"operation:", layernorm1[3],"compute_delay:",layernorm1[4])

        var = layernorm1[1]
        qkv = QKV(var,hidden_dim, dtype_bytes,prompt_len)
        #print(QKV)
        print('QKV_linear:')
        print("input_size:", qkv[0],"output_size:", qkv[1],"weight:", qkv[2],"operation:", qkv[3])

        var = qkv[0]*2
        qk  = QK(var,hidden_dim,1,prompt_len)
        #print(QK)
        print('QK^T:')
        print("input_size:", qk[0],"output_size:", qk[1],"weight:", qk[2],"operation:", qk[3])

        var = qk[1]
        softmax= Softmax(var,hidden_dim,1,prompt_len)
        #print(softmax)
        print('Softmax:')
        print("input_size:", softmax[0], "output_size:", softmax[1], "weight:", softmax[2], "operation:", softmax[3])

        var=softmax[1]+qkv[0]
        mul=Mul(var,hidden_dim,1,prompt_len)
        #print(mul)
        print('Mul:')
        print("input_size:", mul[0],"output_size:", mul[1],"weight:", mul[2],"operation:", mul[3])

        var = mul[1]
        Linear1=Linear(var,1088,1088,hidden_dim,1,prompt_len)
        #print(mul)
        print('Linear1:')
        print("input_size:", Linear1[0],"output_size:", Linear1[1],"weight:", Linear1[2],"operation:", Linear1[3])


        var=Linear1[1]
        add1=Add(var,hidden_dim,1,prompt_len)
        #print(mul)
        print('add1:')
        print("input_size:", add1[0],"output_size:", add1[1],"weight:", add1[2],"operation:", add1[3])

        var=add1[1]
        layernorm2=layer_Norm(var,hidden_dim,1,prompt_len)
        #print(mul)
        print('layernorm2:')
        print("input_size:", layernorm2[0],"output_size:", layernorm2[1],"weight:", layernorm2[2],"operation:", layernorm2[3])


        var = layernorm2[1]
        Linear2=Linear(var,1088,4352,hidden_dim,1,prompt_len)
        #print(mul)
        print('Linear2:')
        print("input_size:", Linear2[0],"output_size:", Linear2[1],"weight:", Linear2[2],"operation:", Linear2[3])

        var = Linear2[1]
        relu=Relu(var,hidden_dim,1,prompt_len)
        #print(mul)
        print('Relu:')
        print("input_size:", relu[0],"output_size:", relu[1],"weight:", relu[2],"operation:", relu[3])


        Linear3=Linear(var,4352,1088,hidden_dim,1,prompt_len)
        #print(mul)
        print('Linear3:')
        print("input_size:", Linear3[0],"output_size:", Linear3[1],"weight:", Linear3[2],"operation:", Linear3[3])


        var=Linear3[1]
        add2=Add(var,hidden_dim,1,1024)
        #print(mul)
        print('add2:')
        print("input_size:", add2[0],"output_size:", add2[1],"weight:", add2[2],"operation:", add2[3])
