function eucDis(x, y)
   #sqrt((clusterPoint[1] - person[xVar])^2 + (clusterPoint[2] - person[yVar])^2)
    return(sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2))
end

"""
Handles the first step of the kmeans algorithm by randomly choosing k cluster centroids from the provided data and then assigning all the other data points to the nearest centroid.

Parameters:
    data - A dataframe that contains the data we will be working with 
    k - Represents the number of clusters we want for this iteration
    xVar - represents the first feature we want to use for clustering
    yVar - represents the second feature we want to use for clustering

Return:
    group - A dictionary where each data point has a key and its corresponding value is the cluster it belongs to
"""
function firstIteration(data, k, xVar, yVar)
    #grabs random points to serve as our starting comp. vertices and declares our dictionary to track which vertex each point belongs to
    indexes = rand(1:size(data)[1], k)
    subset = data[indexes, :]
    subset = zip(subset[!, xVar], subset[!, yVar])
    group = reassignPoints(data, subset, xVar, yVar)
    return group
end

"""
A helper function that helps us calculate the euclidean distance from any individual datapoint to all of our current clusters

Parameters:
    data - A dataframe that contains the data we will be working with 
    newClusterPoints - A vector that contains the coordinates for our clusters
    xVar - represents the first feature we want to use for clustering
    yVar - represents the second feature we want to use for clustering

Return:
    group - A dictionary where each data point has a key and its corresponding value is the cluster it belongs to
"""
function reassignPoints(data, newClusterPoints, xVar, yVar)
    
    group = Dict(i => 0 for i in 1:size(data)[1])
    for (index, person) in enumerate(eachrow(data))
        min = Inf
        kVertex = 0
        for (ind, clusterPoint) in enumerate(newClusterPoints)
            eucDistance = eucDis(clusterPoint, person)
            #eucDistance = sqrt((clusterPoint[1] - person[xVar])^2 + (clusterPoint[2] - person[yVar])^2)
            if eucDistance < min
                min = eucDistance
                kVertex = ind
            end
        end
        group[index] = kVertex
    end    
    return group
end

"""
This function takes care of the rest of the kmeans algorithm. It computes the average point for each of our clusters then reassigns all of the data points to the nearest centroid. This step is repeated untill we reach our max iterations or our model converges.
Parameters:
    data - A dataframe that contains the data we will be working with 
    xVar - represents the first feature we want to use for clustering
    yVar - represents the second feature we want to use for clustering
    groupValues - A vector that contains the clusters that each of our data points belongs to

Return:
    data - A new dataframe that now has the most up-to-date clusters in a column labeled "Group"

"""
function otherIterations(data, xVar, yVar, groupValues)
    iter = 0
    data.Group = groupValues
    #loop will either run for 200 iterations or break when our final clusters arent changing
    while iter < 200
        #This creates the new cluster starting points which are computed by taking the average of each cluster
        clusters = unique(groupValues)
        newClusterPoints = []
        for cluster in clusters
            tmpData = filter(row -> row.Group == cluster, data)
            push!(newClusterPoints, (mean(tmpData[!, xVar]), mean(tmpData[!, yVar])))
        end
        
        #reassigns points to the closest cluster centroid
        group = reassignPoints(data, newClusterPoints, xVar, yVar)    
        
        #checks to see if the clusters are still being updated
        sortedGroup = sort(collect(group))
        allValues = [p.second for p in sortedGroup]
        if !(data[!, :Group] == allValues)
            break
        end
        data[!, :Group] = allValues
        iter += 1
    end
    return data
end

"""
The overall function for our implmentation of kmeans. It even attempts to address the shortcomings that can come with using the base kmeans algorithm by running it multiple times for some k and then taking the clusters with the smallest WCSS before returning.

Parameters:
    data - A dataframe that contains the data we will be working with 
    k - represents the number of clusters we want for this iteration
    xVar - represents the first feature we want to use for clustering
    yVar - represents the second feature we want to use for clustering
    
Return:
    newData - A new dataframe that now has the group column appended to it
"""
function kMeansAlgo(data, k, xVar, yVar)
    #At our base we will run kmeans multiple times and take the clusters with the lowest WCSS
    runsList = []
    newData = 0
    min = Inf
    for runs in collect(1:10)
        group = firstIteration(data, k, xVar, yVar)
        groupValues = collect(values(sort(group)))
        newData = otherIterations(data, xVar, yVar, groupValues)
        sum = WCSS(newData, xVar, yVar)
        if sum < min
            min = sum
            runsList = copy(newData.Group)
        end
    end
    newData.Group = runsList
    return newData
end

"""
A helper function that calculates the sum of squared distances between data points and their assigned cluster centroid. The aim is to minimize this value since that means our clustes are a better fit.

Parameters:
    data - A dataframe that contains the data we will be working with 
    xVar - represents the first feature we want to use for clustering
    yVar - represents the second feature we want to use for clustering

Return:
    sum(sumList) - The sum of all the squared distances
"""
function WCSS(data, xVar, yVar)
    clusters = unique(data.Group) 
    sumList = []
    #look at each individual cluster
    for cluster in clusters
        subset = data[data.Group .== cluster, :]
        clusterStats = (mean(subset[!, Symbol(xVar)]), mean(subset[!, Symbol(yVar)]))
        
        #look at each point in each cluster and start to calculate sum
        sum = 0
        for row in eachrow(subset)
            sum = sum + sqrt((clusterStats[1] - row[Symbol(xVar)])^2 + (clusterStats[2] - row[Symbol(yVar)])^2)
        end
        push!(sumList, sum^2)
    end 
    return sum(sumList)
end      

function silhouetteScore(data, xVar, yVar)
    #This code block attempts to calculate the intra-cluster difference for every point
    clusters = unique(data.Group)
    intraClusterDifference = []
    
    #look at the data cluster by cluster making sure to filter our dataframe to just those points ∈ our cluster
    clusterAve = []
    for cluster in clusters
        #filters down to our current cluster and slims the dataframe down to just our features
        tmpData = filter(row -> row.Group == cluster, data)
        push!(clusterAve, (((mean(tmpData[!, xVar]), mean(tmpData[!, yVar])), cluster)))
        tmpData = tmpData[!, [xVar, yVar]]
        
        #looks at each ind. point and calculates the euc. dis. to the rest of the points ∈ in the cluster before 
        #averaging it out. includes the centroid or average of all the points in our cluster
        centroid = (mean(tmpData[!, xVar]), mean(tmpData[!, yVar]))
        for currRow in eachrow(tmpData)
            currTuple = (currRow[xVar], currRow[yVar])
            copy = filter(row -> row != currRow, tmpData)
            tuples = collect(zip(getproperty(copy, xVar), getproperty(copy, yVar)))
            centroidDistance = euclidean(centroid, currTuple)
            distances = [euclidean((x, y), currTuple) for (x, y) in tuples]
            push!(intraClusterDifference, (sum(distances) + centroidDistance)/size(tmpData)[1])
        end
    end

    #This code block attempts to calculate the nearest-cluster difference
    nearestClusterDifference = []
    for currRow in eachrow(data)
        currTuple = (currRow[xVar], currRow[yVar])
        min = Inf
        k = 0
        #This finds the cluster whos centroid has the smallest distance from our current data point
        for elem in clusterAve
            #skips if the cluster matches up with the cluster we are currently in
            if currRow["Group"] == elem[2]
                continue
            end
            eucDistance = eucDis(currTuple, elem[1])
            #eucDistance = sqrt((currTuple[1] - elem[1][1])^2 + (currTuple[2] - elem[1][2])^2)
            if eucDistance < min
                min = eucDistance
                k = elem[2]
            end
        end
        
        #Now that we know which cluster centroid is the closest to our data point we can calc dis from curr to these points
        tmpData = filter(row -> row.Group == k, data)
        tuples = collect(zip(getproperty(tmpData, xVar), getproperty(tmpData, yVar)))
        distances = [sqrt((x - currRow[xVar])^2 + (y - currRow[yVar])^2) for (x, y) in tuples]
        push!(nearestClusterDifference, sum(distances)/size(tmpData)[1])
    end
    
    comb = collect(zip(intraClusterDifference, nearestClusterDifference))
    answer = [(y-x)/max(x, y) for (x, y) in comb]
    return mean(answer)
end
