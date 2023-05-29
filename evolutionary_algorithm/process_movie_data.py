import numpy as np
# L1=np.array([
#     [6.0, 55.02, 5.0, 55.02],
#    [5.85, 57.39, 5.38, 57.53],
#    [5.85, 59.30, 5.61, 59.31],
#    [5.84, 62.05, 5.60, 62.95],
#    [5.85, 67.74, 5.61, 68.91],
#    [5.84, 73.60, 5.61, 77.80],
#    [5.84, 76.48, 5.61, 77.80]
#    ]
# )
# L2=np.array([[5.0, 54.046276, 6.0, 57.01754 ],
#    [4.8333497, 58.838764, 6.055524, 59.514706 ],
#    [4.799975, 62.1289, 6.0563374, 62.56996 ],
#    [4.780407, 65.63535, 6.057657, 65.337585 ],
#    [4.775598, 71.624016, 6.059429, 69.97142 ],
#    [4.773531, 77.4403, 6.060464, 73.593475 ],
#    [4.773531, 77.4403, 6.060464, 73.593475 ],])

# L3=np.array([[6.0827627, 59.03389, 6.708204, 59.04236 ],
#    [6.042831, 64.7611, 6.4574094, 61.603573 ],
#    [5.296978, 66.858055, 6.3421454, 63.75735 ],
#    [6.0205464, 67.72001, 6.229106, 67.54258 ],
#    [6.009515, 70.20684, 6.0730944, 70.44147 ],
#    [5.999148, 74.35725, 5.6840334, 74.47147 ],
#    [5.989865, 80.03124, 5.4583025, 77.012985 ],])

# L4=np.array([[7.615773, 54.046276, 6.0, 49.020405 ],
#    [7.5712786, 55.443665, 6.0656114, 50.39841 ],
#    [7.5177116, 57.48913, 6.072519, 53.188343 ],
#    [7.457038, 58.5235, 6.0596957, 55.695602 ],
#    [7.391533, 58.30952, 6.0621037, 60.041653 ],
#    [7.3237147, 61.294373, 6.07963, 65.39877 ],
#    [7.2562604, 67.424034, 5.8922167, 72.44998 ],])

# L5=np.array([[5.0, 53.018864, 5.0, 44.02272 ],
#    [4.981369, 54.451813, 5.2795916, 46.36809 ],
#    [4.1820207, 57.271286, 5.474103, 48.09366 ],
#    [4.2940955, 59.48109, 6.039458, 49.081566 ],
#    [6.000116, 57.384666, 6.786364, 48.4768 ],
#    [6.0049515, 58.18075, 6.786364, 48.4768 ],
#    [6.0049515, 58.18075, 6.786364, 48.4768 ],])

L1=np.array([
    [6.0*2, 55.018177, 5.0*2, 55.018177],
[5.8518233*2, 57.39338, 5.3855777*2, 57.5326],
    [5.8516088*2, 59.3043, 5.451714*2, 59.84981],
    [5.842594*2, 62.056427, 5.443049*2, 63.631752],
    [5.8537993*2, 67.74216, 5.4570127*2, 70.31358],
    [5.845182*2, 73.60027, 5.444647*2, 77.4403],
    [5.8425913*2, 76.48529, 5.443199*2, 80.32434],
   ]
)
L2=np.array([
    [5.0*2, 54.046276, 6.0*2, 57.01754],
    [4.8333497*2, 58.838764, 6.055524*2, 59.514706],
    [4.799975*2, 62.1289, 6.0563374*2, 62.56996],
    [4.780407*2, 65.63535, 6.057657*2, 65.337585],
    [4.775598*2, 71.624016, 6.059429*2, 69.97142],
    [4.773531*2, 77.4403, 6.060464*2, 73.593475],
    [4.773531*2, 77.4403, 6.060464*2, 73.593475]])

L3=np.array([
    [6.0827627*2, 59.03389, 6.708204*2, 59.04236],
    [6.042831*2, 64.7611, 6.4574094*2, 61.603573],
    [5.296978*2, 66.858055, 6.3421454*2, 63.75735],
    [6.0205464*2, 67.72001, 6.229106*2, 67.54258],
    [6.009515*2, 70.20684, 6.0730944*2, 70.44147],
    [5.999148*2, 74.35725, 6.0730944*2, 70.44147],
    [5.989865*2, 80.03124, 6.0730944*2, 70.44147],])

L4=np.array([
    [7.615773*2, 54.046276, 6.0*2, 49.020405],
    [7.5037265*2, 57.567352, 6.0656114*2, 50.39841],
    [7.4378967*2, 59.514706, 6.072519*2, 53.188343],
    [7.366795*2, 62.1128, 6.0596957*2, 55.695602],
    [7.2931447*2, 65.51336, 6.0621037*2, 60.041653],
    [7.2198496*2, 69.079666, 6.07963*2, 65.39877],
    [6.887282*2, 69.699356, 5.8922167*2, 72.44998],])

L5=np.array([
    [5.0*2, 53.018864, 5.0*2, 44.02272],
  [4.981369*2, 54.451813, 4.942497*2, 45.55217],
 [4.1820207*2, 57.271286, 3.398935*2, 48.124836],
[4.2940955*2, 59.48109, 4.729985*2, 45.814846],
[6.000116*2, 57.384666, 6.840393*2, 48.062458],
 [6.0049515*2, 58.18075, 6.6811924*2, 51.971146],
[6.0049515*2, 58.18075, 6.5893493*2, 56.302753],])

slide = []
for i in range(7):
    slide.append([L1[i], L2[i],L3[i],L4[i],L5[i]])
slide = np.array(slide)
slide = np.round(slide, 2)
print(slide)