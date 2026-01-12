import numpy as np
import yaml




names = {"FPGA_PRESENT_RANDOMIZED_CLOCK": True, "FPGA_PRESENT_TI_MISALIGNED": False}
traceinfo = {}

with open("traces.yml","r") as f:
    ydata = f.read()
    traceinfo = yaml.safe_load(ydata)


def RANDOMIZED_CLOCK():
    for name in names:
        trdata = traceinfo[name]
        nrfiles = trdata["nrfiles"]
        trperfile = trdata["tracesinfile"]
        tracelen = trdata["tracelen"]
        dt = eval(trdata["struct"])
        for filenr in range(nrfiles):
            fname = "FPGA_PRESENT_RANDOMIZED_CLOCK/Traces_1.dat".format(filenr+1) 
            with open(fname, "rb") as f:
                data = np.fromfile(f , dtype = dt, count = trperfile )
            Traces = data["trace"]
            Input = data["group"]
    np.save('Traces_PRESENT_RC.npy', np.column_stack((Traces, Input)))


if __name__== '__main__' :
    RANDOMIZED_CLOCK()
