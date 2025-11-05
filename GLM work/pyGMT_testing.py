import matplotlib as plt
import pygmt
pygmt.show_versions()

fig = pygmt.Figure()
fig.coast(region=[-69, -68, 43.75, 44.75], shorelines=True)
#fig.show()

fig2 = pygmt.Figure()
fig2.coast(
    region=[-69, -68, 43.75, 44.75], #[lower_long, higher_long, lower_lat, higher_lat] Lat-Long bounds.
    shorelines=True,
    land='khaki',
    water='azure',
    lakes='red',
    projection='M10c', # mercator projection 'M' with figure width of 10 cm '10c'
    frame=['a', '+tMaine'] # title is set with '+t' followed by desired title string.
)
#fig2.show()
import math

print(math.cos(90)**2)
import numpy as np
print(pow(np.cos(90), 2))

print(np.tan(np.deg2rad(15)))