import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



def plot_sensors(A,location_df):
    positions = location_df.values
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='b',s=4,alpha=0.2)
    ax.scatter(positions[A,0],positions[A,1],positions[A,2],c='r',s=10,alpha=1)