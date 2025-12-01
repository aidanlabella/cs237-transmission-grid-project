import opendssdirect as dss
import pandas as pd
import re

dss.run_command("compile Master.dss")

df = pd.DataFrame({'bus':dss.Circuit.AllNodeNames()})
df.bus = df.bus.map(lambda b : re.sub("\.[1-3]", "", b))

#t0 = 12480
#for t in range(t0, t0+96, 4):
#    dss.run_command("Solve time="+str(t))
#    df[t] = pd.DataFrame(dss.Circuit.AllBusMagPu())

dss.run_command("Solve time=12496")
df['v'] = pd.DataFrame(dss.Circuit.AllBusMagPu())
df = df.groupby(by='bus').mean()

df.to_csv("./bus_voltages_p13uhs22_1247--p13udt26466.csv")
