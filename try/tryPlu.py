folder = 'C:/Elaborazione_profilometro/Symetrics/Txt_files/plu/'

with open(folder + '_50x_RSM_SYMETRICS_R0001.plu', 'br') as plu:
    lines = plu.readlines()
    print(lines[0])
    summ = 0
    for line in lines:
        print(len(line))
        summ += len(line)

    print(summ)
