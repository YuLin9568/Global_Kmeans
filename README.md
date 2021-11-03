# Global Kmeans

## Introduction
This project is based on the paper, The global k-means clustering algorithm. Likas et al. proposed the global K-means algorithm to generate cluster result in global minimum, but the disadvantage is too cost in clustering huge datasets. Although the authors developed the fast global K-means algorithm to accelerate global K-means, this algorithm also sacrifice the quality of cluster result in our datasets. Thus, this project will show how to modify the algorithms by combining the advantage of global K-means and fast global K-means.
## Algorithm Mechanism
Here is the brief introduction about each algorithm, and assume<br />
M: Number of cluster, N: Number of data point, D: Feature dimension in a data points, T: Iteraction step <br /><br />
### Global K-means <br />
The algorithm proceeds in increamental way to solve a clustering problem with M clusters, the cluster center is gradually added from k=1 to k=M in the process. which is different with the tradional K-means algorithm, all cluster centers are spreaded in the datasets firstly, and search the convergenced solution or lower inertia by Lloyd's algorithm. 
However, there are N exeutions of the k-means algorithm for each adding cluster center step in global K-means algorithm, and the detail are shown below:<br />
* 1st cluster center: <br />
Mean of all data points to get the minimum inertia<br /> <br />

* 2nd cluster center: <br />
<pre>
for i in range(N): 
    append(1st cluster center, data[i])
    execute K-means
    calculate the inertia after K-means(final inertia)
Choose the data point who has lowest final inertia
</pre>

* 3rd cluster center: 
<pre>
for i in range(N): 
    append(result of 1~2 cluster center, data[i])
    execute K-means
    calculate the inertia after K-means(final inertia)
Choose the data point who has lowest final inertia
</pre>

......<br />
......<br />
* Mth cluster center: <br />
<pre>
for i in range(N): 
    append(result of 1~(M-1) cluster center, data[i])
    execute K-means
    calculate the inertia after K-means(final inertia)
Choose the data point who has lowest final inertia
</pre>

<br />Complexity: K^2xN^2xDxT <br /><br />

### Fast Global K-means<br />
The algorithm is similar to Global K-means, but the fast global K-means only execute the K-means algorithm one time for the data point which has minimum initial inertia(inertia before K-means), in each adding cluster center step. The authors assume that the data point with minimum initial inertia also has the lower final inertia.

<br />Complexity: KxN^2xD + K^2xNxDxT<br /><br />
### Mix Global K-means <br />
From the result of Fast global K-means, the lowest initial inertia doesn't mean the lowest final inertia, and the difference gradually increase in each step of adding cluster center. However, the direction of choosing the point with lower inertia is the correct strategy, because the relation between initial inertia and final inertia imply that lower initial inertia has higher chance with lower final inertia.

Thus, the efficiency could be improved by only execution of K-means for the points with lower initial inertia, and the mix global K-means select N^0.5 data points to execute K-means algorithm.
<br />Complexity: KxN^2xD + K^2xN^(3/2)xDxT<br /><br />
## Reference:
https://www.sciencedirect.com/science/article/abs/pii/S0031320302000602
