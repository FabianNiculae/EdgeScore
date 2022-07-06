import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

##### Nr of devices handled and margin error
# data100 = [[10, 97 ,5], [20, 132 ,10], [30, 134 ,11], [40, 136 ,12], [50, 137 ,12], [60, 138 ,12], [70, 139 ,13], [80, 141 ,12],
#            [90, 141 ,13], [100, 142,10]]
# data100R = [[10, 60,10], [20, 85 ,12], [30, 100 ,15], [40, 113 ,13], [50, 117 ,12], [60, 117 ,10], [70, 118 ,14], [80, 126 ,12],
#            [90, 141,13], [100, 142 ,10]]
# data200 = [[10, 302,15], [20, 360 ,20], [30, 365 ,22], [40, 382 ,23], [50, 383 ,25], [60, 385,26], [70, 386 ,22], [80, 388 ,24],
#            [90, 389 ,18], [100, 391 ,22]]
# data200R = [[10, 180,22], [20, 280 ,25], [30, 300 ,19], [40, 315 ,17], [50, 327 ,22], [60, 334 ,21], [70, 336 ,22], [80, 340 ,19],
#            [90, 342 ,23], [100, 344 ,24]]
# data300 = [[10, 540 ,14], [20, 716 ,22], [30, 796 ,21], [40, 798 ,24], [50, 799 ,24], [60, 801 ,36], [70, 803 ,21], [80, 803 ,20],
#            [90, 805 ,21], [100, 805 ,19]]
# data300R = [[10, 350 ,21], [20, 490 ,23], [30, 530 ,21], [40, 540 ,23], [50, 545 ,25], [60, 547 ,24], [70, 550 ,23], [80, 554 ,28],
#            [90, 558 ,23], [100, 565 ,25]]
# data400 = [[10, 758 ,21], [20, 776 ,26], [30, 796 ,29], [40, 797 ,26], [50, 799 ,25], [60, 800 ,27], [70, 804 ,21], [80, 805 ,22],
#            [90, 806 ,23], [100, 806 ,25]]
# data400R = [[10, 520 ,21], [20, 570 ,18], [30, 720 ,19], [40, 723 ,18], [50, 726 ,16], [60, 734 ,14], [70, 743 ,14], [80, 746 ,23],
#            [90, 747 ,23], [100, 750 ,21]]
# data500 = [[10, 776 ,14], [20, 850 ,19], [30, 855 ,21], [40, 860, 22], [50, 862 ,17], [60, 863 ,13], [70, 865 ,22], [80, 867 ,21],
#            [90, 870 ,15], [100, 872 ,12]]
# data500R = [[10, 630 ,23], [20, 740 ,31], [30, 770 ,23], [40, 775 ,25], [50, 783 ,23], [60, 788 ,21], [70, 790 ,20], [80, 792 ,19],
#            [90, 796 ,22], [100, 800 ,31]]
# data600 = [[10, 927 ,10], [20, 933 ,9], [30, 934 ,8], [40, 934 ,7], [50, 936 ,9], [60, 937 ,8], [70, 938 ,6], [80, 940 ,5],
#            [90, 941 ,6], [100, 943 ,7]]
# data600R = [[10, 690,31], [20, 840 ,32], [30, 860 ,31], [40, 870 ,35], [50, 885 ,24], [60, 899 ,17], [70, 913 ,13], [80, 921 ,11],
#            [90, 931 ,10], [100, 943,6]]
##### Total Cost and margin error
# data100 = [[10, 3300, 200], [20, 7600, 300], [30, 7800 ,500], [40, 7900, 600], [50, 7950, 800], [60, 8000,1000], [70, 8200, 1200], [80, 8250, 1300],
#            [90, 8270, 1400], [100, 8300, 1500]]
# data600R = [[10, 3800, 300], [20, 8290, 400], [30, 12500, 400], [40, 17300, 800], [50, 22500,1100], [60, 27000,2000], [70, 31070,2400], [80, 35270 ,2500],
#            [90, 39000, 2700], [100, 42500,2800]]
# data200 = [[10, 3240, 300], [20, 7540, 350], [30, 8800 ,360], [40, 9470, 400], [50, 10960, 500], [60, 10850,800], [70, 10900, 900], [80, 11000, 930],
#            [90, 11120,1000], [100, 11200 ,800]]
# data200R = [[10, 3800], [20, 8290], [30, 12500], [40, 17300], [50, 22500], [60, 27000], [70, 31070], [80, 35270],
#            [90, 39000], [100, 42500]]
# data300 = [[10, 3040 ,200], [20, 7190, 350], [30, 8480, 400], [40, 8850, 470], [50, 9000 ,500], [60, 9200 ,550], [70, 9300 ,600], [80, 9320, 640],
#            [90, 9325 ,700], [100, 9330, 800]]
# data300R = [[10, 3800], [20, 8290], [30, 12500], [40, 17300], [50, 22500], [60, 27000], [70, 31070], [80, 35270],
#            [90, 39000], [100, 42500]]
# data400 = [[10, 3040 ,200], [20, 4940, 300], [30, 6320 ,500], [40, 6500, 600], [50, 6650, 750], [60, 6666, 800], [70, 6671, 750], [80, 6672, 600],
#            [90, 6680 ,500], [100, 6682 ,400]]
# data400R = [[10, 3800], [20, 8290], [30, 12500], [40, 17300], [50, 22500], [60, 27000], [70, 31070], [80, 35270],
#            [90, 39000], [100, 42500]]
# data500 = [[10, 2340 ,150], [20, 4250 ,200], [30, 4530 ,300], [40, 4600 ,320], [50, 4630 ,330], [60, 4670 ,400], [70, 4675 ,420], [80, 4690 ,400],
#            [90, 4700 ,350], [100, 4705 ,360]]
# data500R = [[10, 3800], [20, 8290], [30, 12500], [40, 17300], [50, 22500], [60, 27000], [70, 31070], [80, 35270],
#            [90, 39000], [100, 42500]]
# data600 = [[10, 2250 ,100], [20, 3000 ,200], [30, 3600 ,250], [40, 3630 ,240], [50, 3640 ,260], [60, 3650 ,240], [70, 3660 ,278], [80, 3675 ,270],
#            [90, 3690 ,260], [100, 3720 ,265]]
# data600R = [[10, 3800], [20, 8290], [30, 12500], [40, 17300], [50, 22500], [60, 27000], [70, 31070], [80, 35270],
#            [90, 39000], [100, 42500]]
# 787.11

#### Area Coverage Ration

data100 = [[10, 0.057], [20, 0.079], [30, 0.080], [40, 0.0805], [50, 0.0809], [60, 0.0813], [70, 0.0815], [80, 0.0817],
           [90, 0.0819], [100, 0.082]]

data200 = [[10, 0.22], [20, 0.255], [30, 0.259], [40, 0.262], [50, 0.264], [60, 0.266], [70, 0.267], [80, 0.268],
           [90, 0.269], [100, 0.27]]

data300 = [[10, 0.44], [20, 0.46], [30, 0.475], [40, 0.485], [50, 0.494], [60, 0.501], [70, 0.506], [80, 0.508],
           [90, 0.509], [100, 0.51]]

data400 = [[10, 0.61], [20, 0.62], [30, 0.628], [40, 0.634], [50, 0.638], [60, 0.64], [70, 0.643], [80, 0.645],
           [90, 0.647], [100, 0.65]]

data500 = [[10, 0.72], [20, 0.726], [30, 0.73], [40, 0.734], [50, 0.736], [60, 0.737], [70, 0.7375], [80, 0.738],
           [90, 0.739], [100, 0.74]]

data600 = [[10, 0.78], [20, 0.784], [30, 0.786], [40, 0.787], [50, 0.7876], [60, 0.788], [70, 0.7882], [80, 0.7886],
           [90, 0.789], [100, 0.79]]

# Create the pandas DataFrame
# df = pd.DataFrame(data100, columns=['Percentage', 'Ratio', 'mean'])
# dfr = pd.DataFrame(data100R, columns=['Percentage', 'Ratio','mean'])
# da = pd.DataFrame(data200, columns=['Percentage', 'Ratio','mean'])
# dar = pd.DataFrame(data200R, columns=['Percentage', 'Ratio','mean'])
# db = pd.DataFrame(data300, columns=['Percentage', 'Ratio','mean'])
# dbr = pd.DataFrame(data300R, columns=['Percentage', 'Ratio','mean'])
# dc = pd.DataFrame(data400, columns=['Percentage', 'Ratio','mean'])
# dcr = pd.DataFrame(data400R, columns=['Percentage', 'Ratio','mean'])
# dd = pd.DataFrame(data500, columns=['Percentage', 'Ratio','mean'])
# ddr = pd.DataFrame(data500R, columns=['Percentage', 'Ratio','mean'])
# de = pd.DataFrame(data600, columns=['Percentage', 'Ratio','mean'])
# der = pd.DataFrame(data600R, columns=['Percentage', 'Ratio', 'mean'])


# Create the pandas DataFrame
df = pd.DataFrame(data100, columns=['Percentage', 'Ratio'])
da = pd.DataFrame(data200, columns=['Percentage', 'Ratio'])
db = pd.DataFrame(data300, columns=['Percentage', 'Ratio'])
dc = pd.DataFrame(data400, columns=['Percentage', 'Ratio'])
dd = pd.DataFrame(data500, columns=['Percentage', 'Ratio'])
de = pd.DataFrame(data600, columns=['Percentage', 'Ratio'])

# plt.errorbar(de["Percentage"], de["Ratio"],de["mean"], label="radius 600 m",  marker=".")
# plt.errorbar(der["Percentage"], der["Ratio"],der["mean"], label="radius 600 m RANDOM",  marker=".")
# plt.errorbar(dd["Percentage"], dd["Ratio"],dd["mean"], label="radius 500 m",  marker=".")
# plt.errorbar(ddr["Percentage"], ddr["Ratio"],ddr["mean"], label="radius 500 m RANDOM",  marker=".")
# plt.errorbar(dc["Percentage"], dc["Ratio"],dc["mean"], label="radius 400 m",  marker=".")
# plt.errorbar(dcr["Percentage"], dcr["Ratio"],dcr["mean"], label="radius 400 m RANDOM",  marker=".")
# plt.errorbar(db["Percentage"], db["Ratio"],db["mean"], label="radius 300 m",  marker=".")
# plt.errorbar(dbr["Percentage"], dbr["Ratio"], dbr["mean"],label="radius 300 m RANDOM",  marker=".")
# plt.errorbar(da["Percentage"], da["Ratio"],da["mean"], label="radius 200 m",  marker=".")
# plt.errorbar(dar["Percentage"], dar["Ratio"],dar["mean"], label="radius 200 m RANDOM",  marker=".")
# plt.errorbar(df["Percentage"], df["Ratio"],df["mean"], label="radius 100 m" ,  marker=".")
# plt.errorbar(dfr["Percentage"], dfr["Ratio"],dfr["mean"], label="radius 100 m RANDOM")

plt.errorbar(de["Percentage"], de["Ratio"], label="radius 600 m", marker=".")
plt.errorbar(dd["Percentage"], dd["Ratio"], label="radius 500 m", marker=".")
plt.errorbar(dc["Percentage"], dc["Ratio"], label="radius 400 m", marker=".")
plt.errorbar(db["Percentage"], db["Ratio"], label="radius 300 m", marker=".")
plt.errorbar(da["Percentage"], da["Ratio"], label="radius 200 m", marker=".")
plt.errorbar(df["Percentage"], df["Ratio"], label="radius 100 m", marker=".")

plt.xlabel('Percentage of selected traffic lights')
plt.ylabel('Coverage Ratio')
plt.title('Cumulative Distribution Function based on the radius and percentage of selected traffic lights')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("Radius.png", bbox_inches='tight')

plt.close()
