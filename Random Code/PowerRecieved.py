import numpy as np

## Basing these calculations based on the 36D6 Radar system as part of a S-300 variant. Frequency range of 2850 - 3200 MHz (using 3000 MHz for calcs). Peak power of 350 kW. Average power of 3 kW. Range of 200 km.
## Calculated using the Friis Transmission Equation. Target range of 150 miles. Using 0.9 meter parabolic dish. 

freq = 3000000 #Transmitted Frequency
llambda = 300000000/freq
D = 0.9 # Diameter of the dish
d = 150

Pt = 50 # Power transmitted
Gt = 26.872 ## 10* np.log10(((D/llambda)**2)) # Gain of the Antenna ##
Gr = 10 # Gain of the Receiver
X = 20*np.log10((llambda/(4*np.pi*d))**2) # Free space loss function

Pr = Pt+Gt+Gr+X
print(Pr)

blank = 10*np.log10(3.7815317317179176)
print(Gt)

## Half power beam width (HPBW) is 6.66205467 degrees


'''
Friis calc:
https://www.pasternack.com/t-calculator-friis.aspx?srsltid=AfmBOoqGRQhBmpHgIWBLYFWyV6E8xxkhDn_tx7lcoA6hfxdDmeGjKC_o

36D6 data:
https://www.scribd.com/document/337000241/36D6-M-Radar
https://www.radartutorial.eu/19.kartei/11.ancient/karte060.en.html

different Calcs used:
https://www.everythingrf.com/rf-calculators/parabolic-reflector-antenna-gain
https://www.everythingrf.com/rf-calculators/parabolic-reflector-antenna-calculator
https://www.everythingrf.com/rf-calculators/friis-transmission-calculator
https://www.everythingrf.com/rf-calculators/watt-to-dbm

Pasternack Amplifier Chosen:
https://www.pasternack.com/47-db-gain-50-watt-psat-1-ghz-to-3-ghz-high-power-gan-amplifier-class-ab-sma-pe15a5118-p.aspx

S-300 Wikipedia page:
https://en.wikipedia.org/wiki/S-300_missile_system

'''