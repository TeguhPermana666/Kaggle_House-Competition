"""
Termasuk sebuah jenis algoritma unsupervised learning
Clustering->pengelompokan beberapa titik ke suatu grup berdasarkan kedekatannya dengan satu titik dengan titik lainnya.
K-means->algoritma unsupervised
=>mengukur sebuah kesamaan dengan garis lurus biasa. yang menciptakan sebuah cluster yang menempatkan sebuah titik (centroid)
titik dalam kumpulan di tetapkan kedalam cluster dari centroid yang paling terdekat dengan sebuah titik.
k-means=>k adalah berapa banyak centroid yang ditetapkan yang mana centroid disini digunakan untuk penentuan cluster
penentuan k dapat dilakukan dengan metode elbow yakni mencari setiap ineria antar titik didalam sebuah centroid misal di alkumlasu 11 centroid
dari line elbow di temukan sebuah kemiringan garis yang tidak signifikan(perubahan elbow tidak drastis) maka disana titik potong centroid untuk di kalkulasi
menjadi k

ketika sebuah set lingkaran cluster tumpang tindih akan menciptakan sebuah tessalasi voronoi


3 parameter untuk k_means:
scikit-learn: n_clusters, max_iter, dan n_init.
1.tentukan jumlah k->centroid yang akan digunakan
2.assign point untuk centroid cluster terdekat
3.pindahkan centroid untuk meminimalkan jarak antara titik titiknya 

dua langkah (2,3) terus berulang sampai centroid tidak bergerak lagi
atau sampai beberapa jumlah maksimum iterasi has passed (max_)

centroid awal akan ditandai dengan clustering yang buruk
->maka dari itu diperlukan sebuah alasan untuk mengulangi algo beberapa kali(n_init)
dan mengembalikan pengelompokan yang memiliki jarak total kecil antara titik yang mana akan mengembalikan
sebuah nilai dengan jarak total paling kecil antara titik dan pusatnya, pengelompokan yang optiomal
n_cluster digunakan untuk sebuah centroid yang ditentukan.

ketergantungan hasil pada centroid awal dan pentingnya sebuah iterasi sampai konvegerasi
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans 
data=pd.read_csv("feature_enginer\housing.csv")
print(data)

X=data.loc[:,["MedInc","Latitude","Longitude"]]
print(X.head())
#create cluster
cluster=[]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(X)
    cluster.append(km.inertia_)
print(cluster)

fig,ax=plt.subplots(figsize=(10,8))
sns.lineplot(x=range(1,11),y=cluster)
ax.set_title("Cari elbow")
ax.set_xlabel("Clusters")
ax.set_ylabel("Inertia")
plt.show()

kmr2=KMeans(n_clusters=2)
X["Cluster"]=kmr2.fit_predict(X)
# X["Cluster"]=X["Cluster"].astype("category")
print(X.Cluster)
# X.to_csv("feature_enginer/output.csv")
print("Submission is succes")
fig,axs=plt.subplots(1,2,figsize=(8,6))
sns.scatterplot(x=X["Longitude"],y=X["Latitude"],hue=X["Cluster"],ax=axs[0])
sns.relplot(x="Longitude", y="Latitude",data=X,hue="Cluster",ax=axs[1],col=axs[1])
plt.show()