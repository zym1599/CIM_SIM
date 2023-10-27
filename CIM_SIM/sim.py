from net_unit_digital import *
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
                FFN_hidden_dim,
                block_number,
                rram_area,
                dmac_area,
                # dmac_computational_power,
                # dmac_utilization,
                #rram_computational_power,
                sram_capacity,
                sram_read_write_bandwidth,
                sram_utilization,
                dram_read_write_bandwidth,
                dram_capacity=900000,
                dram_utilization=0.8
                ):
        comput_delay_total = 0
        sram_delay_total = 0
        area_total = 0
        mode = 'prefill'
        for i in range(block_number):
           # print('Block:', i + 1)

            layernorm1 = layer_Norm(input, hidden_dim, dtype_bytes, prompt_len,dmac_area=dmac_area)#################
            # print(layernorm)
            print('layernorm:')
            print("input_size:", layernorm1[0], "output_size:", layernorm1[1], "weight:", layernorm1[2], "operation:",
                 layernorm1[3],
                 "compute_delay:", round(layernorm1[4], 5),
                 'read_sram_delay:', layernorm1[5], 'write_sram_delay:', layernorm1[6])

            var = layernorm1[1]
            qkv = QKV(var, hidden_dim, dtype_bytes, prompt_len,rram_area=rram_area)                  #################
            # print(QKV)
            print('QKV_linear:')
            print("input_size:", qkv[0], "output_size:", qkv[1], "weight:", qkv[2], "operation:", qkv[3],
                  "compute_delay:", round(qkv[4], 5),
                  'read_sram_delay:', qkv[5], 'write_sram_delay:', round(qkv[6], 5), 'area:', qkv[9])

            var = qkv[0] * 2
            qk = QK(var, hidden_dim, 1, prompt_len,dmac_area=dmac_area,mode=mode)                                 #################
            # print(QK)
            print('QK^T:')
            print("input_size:", qk[0], "output_size:", qk[1], "weight:", qk[2], "operation:", qk[3],
                  "compute_delay:", round(qk[4], 5),
                  'read_sram_delay:', qk[5], 'write_sram_delay:', round(qk[6], 5), 'area:', qk[9])

            var = qk[1]
            softmax = Softmax(var, hidden_dim, dtype_bytes, prompt_len,dmac_area=dmac_area)                        #################
            # print(softmax)
            print('Softmax:')
            print("input_size:", softmax[0], "output_size:", softmax[1], "weight:", softmax[2], "operation:",
                  softmax[3],
                  "compute_delay:", round(softmax[4], 5),
                  'read_sram_delay:', softmax[5], 'write_sram_delay:', round(softmax[6], 5), 'area:', softmax[9])

            var = softmax[1] + qkv[0]
            mul = Mul(var, hidden_dim, dtype_bytes, prompt_len,dmac_area=dmac_area,mode=mode)                             #################
            # print(mul)
            print('Mul:')
            print("input_size:", mul[0], "output_size:", mul[1], "weight:", mul[2], "operation:", mul[3],
                  "compute_delay:", round(mul[4], 5),
                  'read_sram_delay:', round(mul[5], 5), 'write_sram_delay:', round(mul[6], 5), 'area:', mul[9])

            var = mul[1]
            Linear1 = Linear(var, Linear_in_dim=hidden_dim, Linear_out_dim=hidden_dim, hidden_dim=hidden_dim,#################
                             dtype_bytes=dtype_bytes, prompt_len=prompt_len,rram_area=rram_area)
            # print(mul)
            print('Linear1:')
            print("input_size:", Linear1[0], "output_size:", Linear1[1], "weight:", Linear1[2], "operation:",
                  Linear1[3],
                  "compute_delay:", round(Linear1[4], 5),
                  'read_sram_delay:', Linear1[5], 'write_sram_delay:', round(Linear1[6], 5), 'area:', Linear1[9])

            var = Linear1[1]
            add1 = Add(var, hidden_dim, dtype_bytes, prompt_len,dmac_area=dmac_area)                               #################
            # print(mul)
            print('add1:')
            print("input_size:", add1[0], "output_size:", add1[1], "weight:", add1[2], "operation:", add1[3],
                  "compute_delay:", round(add1[4], 5),
                  'read_sram_delay:', round(add1[5], 5), 'write_sram_delay:', round(add1[6], 5), 'area:', add1[9])

            var = add1[1]
            layernorm2 = layer_Norm(var, hidden_dim, dtype_bytes, prompt_len, dmac_area=dmac_area)                         #################
            # print(mul)
            print('layernorm2:')
            print("input_size:", layernorm2[0], "output_size:", layernorm2[1], "weight:", layernorm2[2], "operation:",
                  layernorm2[3],
                  "compute_delay:", round(layernorm2[4], 5),
                  'read_sram_delay:', round(layernorm2[5], 5), 'write_sram_delay:', round(layernorm2[6], 5), 'area:',
                  layernorm2[9])

            var = layernorm2[1]
            Linear2 = Linear(var, Linear_in_dim=hidden_dim, Linear_out_dim=FFN_hidden_dim, hidden_dim=hidden_dim,       #################
                             dtype_bytes=dtype_bytes, prompt_len=prompt_len,rram_area=rram_area)
            # print(mul)
            print('Linear2:')
            print("input_size:", Linear2[0], "output_size:", Linear2[1], "weight:", Linear2[2], "operation:",
                  Linear2[3],
                  "compute_delay:", round(Linear2[4], 5),
                  'read_sram_delay:', round(Linear2[5], 5), 'write_sram_delay:', round(Linear2[6], 5), 'area:',
                  Linear2[9])

            var = Linear2[1]
            relu = Relu(var, hidden_dim, dtype_bytes, prompt_len, dmac_area=dmac_area)                        #################
            # print(mul)
            print('Relu:')
            print("input_size:", relu[0], "output_size:", relu[1], "weight:", relu[2], "operation:", relu[3],
                  "compute_delay:", round(relu[4], 5),
                  'read_sram_delay:', round(relu[5], 5), 'write_sram_delay:', round(relu[6], 5), 'area:', relu[9])

            Linear3 = Linear(var, Linear_in_dim=FFN_hidden_dim,Linear_out_dim=hidden_dim,  hidden_dim=hidden_dim,       #################
                             dtype_bytes=dtype_bytes, prompt_len=prompt_len,rram_area=rram_area)
            # print(mul)
            print('Linear3:')
            print("input_size:", Linear3[0], "output_size:", Linear3[1], "weight:", Linear3[2], "operation:",
                  Linear3[3],
                  "compute_delay:", round(Linear3[4], 5),
                  'read_sram_delay:', round(Linear3[5], 5), 'write_sram_delay:', round(Linear3[6], 5), 'area:',
                  Linear3[9])

            var = Linear3[1]
            add2 = Add(var, hidden_dim, dtype_bytes, prompt_len=prompt_len,dmac_area=dmac_area)                   #################
            # print(mul)
            print('add2:')
            print("input_size:", add2[0], "output_size:", add2[1], "weight:", add2[2], "operation:", add2[3],
                  "compute_delay:", round(add2[4], 5),
                  'read_sram_delay:', round(add2[5], 5), 'write_sram_delay:', round(add2[6], 5), 'area:', add2[9])

            comput_delay_total = layernorm1[4] + qkv[4] + qk[4] + softmax[4] + mul[4] + \
                                 Linear1[4] + add1[4] + layernorm2[4] + Linear2[4] + \
                                 relu[4] + Linear3[4] + add2[4] + comput_delay_total
            # print(round(comput_delay_total, 5))

            sram_delay_total = layernorm1[5] + qkv[5] + qk[5] + softmax[5] + mul[5] + \
                               Linear1[5] + add1[5] + layernorm2[5] + Linear2[5] + \
                               relu[5] + Linear3[5] + add2[5] + sram_delay_total
            dram_delay = qkv[8]+Linear1[7]+Linear2[7]+Linear3[7]
            # print(round(sram_delay_total, 5))
            # print(round(comput_delay_total + sram_delay_total, 5))
            # dram = DRAM(read_write_bandwidth=38)
            # sram = SRAM(capacity=32,read_write_bandwidth=8192)
            # rram = RRAM(computational_power=30)
            # dmac = DMAC(computational_power=16)
            # area_total = dram.get_area() + sram.get_area() + rram.get_area() + dmac.get_area()
            # print('area _total:', area_total)
        #print('Delat_total:', comput_delay_total+sram_delay_total)

        delay_total=comput_delay_total+sram_delay_total+dram_delay
        print('compute_delay_total:',comput_delay_total)
        return delay_total
        #print('compute_Delat_total:', comput_delay_total)
        # print('area_total:', area_total)



    def decode(self,
                input,
                hidden_dim,
                dtype_bytes,
                prompt_len,
                FFN_hidden_dim,
                block_number,
                rram_area,
                dmac_area,
                # dmac_computational_power,
                # dmac_utilization,
                #rram_computational_power,
                sram_capacity,
                sram_read_write_bandwidth,
                sram_utilization,
                dram_read_write_bandwidth,
                mode='decode',
                dram_capacity=900000,
                dram_utilization=0.8

                ):
        comput_delay_total = 0
        sram_delay_total = 0
        area_total = 0
        mode = 'decode'
        for i in range(block_number):
           # print('Block:', i + 1)

            layernorm1 = layer_Norm(input, hidden_dim, dtype_bytes, prompt_len,dmac_area=dmac_area)#################  layernorm1
            #print(layernorm1)
            # print('layernorm:')
            # print("input_size:", layernorm1[0], "output_size:", layernorm1[1], "weight:", layernorm1[2], "operation:",
            #      layernorm1[3],
            #      "compute_delay:", round(layernorm1[4], 5),
            #      'read_sram_delay:', layernorm1[5], 'write_sram_delay:', layernorm1[6])

            var = layernorm1[1]
            qkv = QKV(var, hidden_dim, dtype_bytes, prompt_len,rram_area=rram_area)                  #################    qkv
            # print(QKV)
            # print('QKV_linear:')
            # print("input_size:", qkv[0], "output_size:", qkv[1], "weight:", qkv[2], "operation:", qkv[3],
            #       "compute_delay:", round(qkv[4], 5),
            #       'read_sram_delay:', qkv[5], 'write_sram_delay:', round(qkv[6], 5), 'area:', qkv[9])

            var = qkv[0] * 2
            qk = QK(var, hidden_dim, 1, prompt_len,dmac_area=dmac_area,mode=mode)                                 #################   QK
            # print(QK)
            # print('QK^T:')
            # print("input_size:", qk[0], "output_size:", qk[1], "weight:", qk[2], "operation:", qk[3],
            #       "compute_delay:", round(qk[4], 5),
            #       'read_sram_delay:', qk[5], 'write_sram_delay:', round(qk[6], 5), 'area:', qk[9])

            var = qk[1]
            softmax = Softmax(var, hidden_dim, dtype_bytes, prompt_len,dmac_area=dmac_area)                        #################    softmax
            # print(softmax)
            # print('Softmax:')
            # print("input_size:", softmax[0], "output_size:", softmax[1], "weight:", softmax[2], "operation:",
            #       softmax[3],
            #       "compute_delay:", round(softmax[4], 5),
            #       'read_sram_delay:', softmax[5], 'write_sram_delay:', round(softmax[6], 5), 'area:', softmax[9])

            var = softmax[1] + qkv[0]
            mul = Mul(var, hidden_dim, dtype_bytes, prompt_len,dmac_area=dmac_area,mode=mode)                             #################    mul
            # print(mul)
            # print('Mul:')
            # print("input_size:", mul[0], "output_size:", mul[1], "weight:", mul[2], "operation:", mul[3],
            #       "compute_delay:", round(mul[4], 7),
            #       'read_sram_delay:', round(mul[5], 5), 'write_sram_delay:', round(mul[6], 5), 'area:', mul[9])

            var = mul[1]
            Linear1 = Linear(var, Linear_in_dim=hidden_dim, Linear_out_dim=hidden_dim, hidden_dim=hidden_dim,#################     Linear1
                             dtype_bytes=dtype_bytes, prompt_len=prompt_len,rram_area=rram_area)
            # print(mul)
            # print('Linear1:')
            # print("input_size:", Linear1[0], "output_size:", Linear1[1], "weight:", Linear1[2], "operation:",
            #       Linear1[3],
            #       "compute_delay:", round(Linear1[4], 5),
            #       'read_sram_delay:', Linear1[5], 'write_sram_delay:', round(Linear1[6], 5), 'area:', Linear1[9])

            var = Linear1[1]
            add1 = Add(var, hidden_dim, dtype_bytes, prompt_len,dmac_area=dmac_area)                               #################    add1
            # print(mul)
            # print('add1:')
            # print("input_size:", add1[0], "output_size:", add1[1], "weight:", add1[2], "operation:", add1[3],
            #       "compute_delay:", round(add1[4], 5),
            #       'read_sram_delay:', round(add1[5], 5), 'write_sram_delay:', round(add1[6], 5), 'area:', add1[9])

            var = add1[1]
            layernorm2 = layer_Norm(var, hidden_dim, dtype_bytes, prompt_len, dmac_area=dmac_area)                         #################    layernorm2
            # print(mul)
            # print('layernorm2:')
            # print("input_size:", layernorm2[0], "output_size:", layernorm2[1], "weight:", layernorm2[2], "operation:",
            #       layernorm2[3],
            #       "compute_delay:", round(layernorm2[4], 5),
            #       'read_sram_delay:', round(layernorm2[5], 5), 'write_sram_delay:', round(layernorm2[6], 5), 'area:',
            #       layernorm2[9])

            var = layernorm2[1]
            Linear2 = Linear(var, Linear_in_dim=hidden_dim, Linear_out_dim=FFN_hidden_dim, hidden_dim=hidden_dim,       #################   Linear2
                             dtype_bytes=dtype_bytes, prompt_len=prompt_len,rram_area=rram_area)
            # print(mul)
            # print('Linear2:')
            # print("input_size:", Linear2[0], "output_size:", Linear2[1], "weight:", Linear2[2], "operation:",
            #       Linear2[3],
            #       "compute_delay:", round(Linear2[4], 5),
            #       'read_sram_delay:', round(Linear2[5], 5), 'write_sram_delay:', round(Linear2[6], 5), 'area:',
            #       Linear2[9])

            var = Linear2[1]
            relu = Relu(var, hidden_dim, dtype_bytes, prompt_len, dmac_area=dmac_area)                        #################   relu
            # print(mul)
            # print('Relu:')
            # print("input_size:", relu[0], "output_size:", relu[1], "weight:", relu[2], "operation:", relu[3],
            #       "compute_delay:", round(relu[4], 5),
            #       'read_sram_delay:', round(relu[5], 5), 'write_sram_delay:', round(relu[6], 5), 'area:', relu[9])

            Linear3 = Linear(var, Linear_out_dim=FFN_hidden_dim, Linear_in_dim=hidden_dim, hidden_dim=hidden_dim,       #################   Linear3
                             dtype_bytes=dtype_bytes, prompt_len=prompt_len,rram_area=rram_area)
            # print(mul)
            # print('Linear3:')
            # print("input_size:", Linear3[0], "output_size:", Linear3[1], "weight:", Linear3[2], "operation:",
            #       Linear3[3],
            #       "compute_delay:", round(Linear3[4], 5),
            #       'read_sram_delay:', round(Linear3[5], 5), 'write_sram_delay:', round(Linear3[6], 5), 'area:',
            #       Linear3[9])

            var = Linear3[1]
            add2 = Add(var, hidden_dim, dtype_bytes, prompt_len=prompt_len,dmac_area=dmac_area)                   #################         add2
            # print(mul)
            # print('add2:')
            # print("input_size:", add2[0], "output_size:", add2[1], "weight:", add2[2], "operation:", add2[3],
            #       "compute_delay:", round(add2[4], 5),
            #       'read_sram_delay:', round(add2[5], 5), 'write_sram_delay:', round(add2[6], 5), 'area:', add2[9])

            comput_delay_total = layernorm1[4] + qkv[4] + qk[4] + softmax[4] + mul[4] + \
                                 Linear1[4] + add1[4] + layernorm2[4] + Linear2[4] + \
                                 relu[4] + Linear3[4] + add2[4] + comput_delay_total
            # print(round(comput_delay_total, 5))

            sram_delay_total = layernorm1[5] + qkv[5] + qk[5] + softmax[5] + mul[5] + \
                               Linear1[5] + add1[5] + layernorm2[5] + Linear2[5] + \
                               relu[5] + Linear3[5] + add2[5] + sram_delay_total
            dram_delay = qkv[8]+qk[7]
            # print(round(sram_delay_total, 5))
            # print(round(comput_delay_total + sram_delay_total, 5))
            # dram = DRAM(read_write_bandwidth=38)
            # sram = SRAM(capacity=32,read_write_bandwidth=8192)
            # rram = RRAM(computational_power=30)
            # dmac = DMAC(computational_power=16)
            # area_total = dram.get_area() + sram.get_area() + rram.get_area() + dmac.get_area()
            # print('area _total:', area_total)
        #print('Delat_total:', comput_delay_total+sram_delay_total)
        if comput_delay_total+sram_delay_total>dram_delay:
            delay_total=comput_delay_total+sram_delay_total
        else:
            delay_total=dram_delay
        print('dram_delay:',dram_delay)
        return delay_total
        #print('compute_Delat_total:', comput_delay_total)
        # print('area_total:', area_total)