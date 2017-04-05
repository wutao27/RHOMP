function init_kneser_ney(filePath, splitRatio)
    numDic1 = Dict(); numDic2 = Dict()
    maxId = 0; dist = zeros(50000)
    n1 = 0; n2 = 0
    f = open(filePath)
    for ln in eachline(f)
        tempArray = readdlm(IOBuffer(ln),Int)
        n = length(tempArray)
        if n<2
            continue
        end
        maxId = max(maxId,maximum(tempArray))
        for i = 1:n
            state = round(Int, tempArray[i])
            if state > length(dist)
                append!(dist, zeros(state-length(dist)))
            end
            dist[state] += 1
        end
        
        ran = rand()
        if ran <= splitRatio
            # -1 denotes the START state
            if haskey( numDic1, (-1,tempArray[1],tempArray[2]) )
                numDic1[(-1,tempArray[1],tempArray[2])]+=1
            else
                numDic1[(-1,tempArray[1],tempArray[2])]=1
            end
            for i = 3:n
                tempTuple = (tempArray[i-2],tempArray[i-1],tempArray[i])
                if haskey(numDic1, tempTuple) numDic1[tempTuple]+=1
                else numDic1[tempTuple] = 1
                end
            end
        else
            if haskey( numDic2, (-1,tempArray[1],tempArray[2]) ) numDic2[(-1,tempArray[1],tempArray[2])]+=1
            else numDic2[(-1,tempArray[1],tempArray[2])]=1
            end
            for i = 3:n
                tempTuple = (tempArray[i-2],tempArray[i-1],tempArray[i])
                if haskey(numDic2, tempTuple) numDic2[tempTuple]+=1
                else numDic2[tempTuple] = 1
                end
            end
        end         
    end

    for key in keys(numDic1)
        if numDic1[key] == 1
            n1 += 1
        elseif numDic1[key] == 2
            n2 += 1
        end
    end
    dist = dist/sum(dist)
    # the total number of states is maxId + 1 (including the start state)
    return numDic1, numDic2, dist[1:maxId], n1/(n1+2*n2)
end


function first_kneser_ney(numDic1, dist, D)
    maxId = length(dist)
    P_first = [zeros(maxId) for i_ = 1:maxId]
    for key in keys(numDic1)
        i,j,k = key
        val = numDic1[key]
        P_first[j][k]+=val
    end
    sumP = [sum(P_first[i]) for i = 1:maxId]
    for i=1:maxId
        tempRatio = D*countnz(P_first[i])/sumP[i]
        tempDist = tempRatio * dist
        if tempDist[i]!=0
            tempDist = tempRatio * tempDist/(tempRatio - tempDist[i])
            tempDist[i] = 0
        end
        P_first[i] = max( P_first[i] - D, zeros(maxId) )
        if sumP[i] > 0
            P_first[i] = P_first[i]/sumP[i] + tempDist
        else
            P_first[i] = zeros(maxId)
        end
    end
    return P_first
end


function second_kneser_ney(numDic1, numDic2, P_first, D)
    maxId = length(P_first)
    indDic = Dict()
    index = 1
    for key in keys(numDic1)
        i,j,k = key
        if !haskey(indDic, (i,j))
            indDic[(i,j)] = index
            index+=1
        end
    end
    for key in keys(numDic2)
        i,j,k = key
        if !haskey(indDic, (i,j))
            indDic[(i,j)] = index
            index+=1
        end
    end
    gama = zeros(index-1)
    totalCount = zeros(index-1)
    uniq = [Dict() for i = 1:index-1]
    for key in keys(numDic1)
        i,j,k = key
        id = indDic[(i,j)]
        gama[id] += numDic1[key]
        totalCount[id] += numDic1[key]
        if !haskey(uniq[id], k)
            uniq[id][k] = 1
        else
            uniq[id][k] += 1
        end
    end
    for i=1:index-1
        if gama[i]>0
            gama[i] = length(uniq[i])/gama[i]
        else
            gama[i] = 1
        end
    end
    return indDic, gama, uniq, totalCount
end


function mrr_first_kneser_ney(numDic2, P_first)
    res, total = 0,0
    n = length(P_first)
    R_first = [n+1-sortperm(sortperm(P_first[i])) for i=1:n]
    for key in keys(numDic2)
        i,j,k = key
        val = numDic2[key]
        res += val/R_first[j][k]
        total += val
    end
    return res/total
end

function precision_first_kneser_ney(numDic2, P_first, topVal)
    m = length(topVal)
    n = length(P_first)
    res = zeros(m); total = 0
    R_first = [n+1-sortperm(sortperm(P_first[i])) for i=1:n]
    for key in keys(numDic2)
        i,j,k = key
        val = numDic2[key]
        for ii in 1:m
            if R_first[j][k] <= topVal[ii]
                res[ii] += val
            end
        end
        total += val
    end
    return res/total
end
    
    
function mrr_second_kneser_ney(numDic2, P_first, D, indDic, gama, uniq, totalCount)
    res, total = 0,0
    n = length(P_first)
    ipre, jpre = -1, -1
    rankVec = zeros(n)
    for key in sort(collect(keys(numDic2)))
        i,j,k = key
        val = numDic2[key]
        if i!=ipre || j!=jpre
            id = indDic[(i,j)]
            if gama[id] == 0
                rankVec = [n/2 for i_=1:n]
            else
                tempVec = D*gama[id]*P_first[j]
                for x in keys(uniq[id])
                    tempVec[x] += (uniq[id][x]-D)/totalCount[id]
                end
                rankVec = n+1-sortperm(sortperm(tempVec))
            end
        end
        ipre, jpre = i,j
        res += val/rankVec[k]
        total += val
    end
    return res/total
end


function precision_second_kneser_ney(numDic2, P_first, D, indDic, gama, uniq, totalCount, topVal)
    m = length(topVal)
    n = length(P_first)
    res = zeros(m); total = 0
    ipre, jpre = -1, -1
    rankVec = zeros(n)
    for key in sort(collect(keys(numDic2)))
        i,j,k = key
        val = numDic2[key]
        if i!=ipre || j!=jpre
            id = indDic[(i,j)]
            if gama[id] == 0
                rankVec = [n/2 for i_=1:n]
            else
                tempVec = D*gama[id]*P_first[j]
                for x in keys(uniq[id])
                    tempVec[x] += (uniq[id][x]-D)/totalCount[id]
                end
                rankVec = n+1-sortperm(sortperm(tempVec))
            end
        end
        ipre, jpre = i,j
        for ii = 1:m
            if rankVec[k] <= topVal[ii]
                res[ii] += val
            end
        end
        total += val
    end
    return res/total
end

function pr_piece_first_kneser_ney(countArray, numDic2, P_first, topVal)
    n = length(P_first)
    pieceDic = Dict()
    perm = sortperm(countArray,rev = true)

    R_first = [n+1-sortperm(sortperm(P_first[i])) for i=1:n]
    for key in keys(numDic2)
        i,j,k = key
        val = numDic2[key]
        if !haskey(pieceDic, j)
            pieceDic[j] = [val, 0]
        else
            pieceDic[j][1]+=val
        end
        if R_first[j][k] <= topVal
            pieceDic[j][2] += val
        end
    end
    x = []; y = []
    for i = 1:length(perm)
        ind = perm[i]
        if !haskey(pieceDic, ind)
            continue
        end
        push!(x, pieceDic[ind][1])
        push!(y, pieceDic[ind][2])
    end
    return x,y
end

function pr_piece_second_kneser_ney(countArray, numDic2, P_first, D, indDic, gama, uniq, totalCount, topVal)
    pieceDic = Dict()
    perm = sortperm(countArray,rev = true)
    n = length(P_first)
    ipre, jpre = -1, -1
    rankVec = zeros(n)
    for key in sort(collect(keys(numDic2)))
        i,j,k = key
        val = numDic2[key]
        if !haskey(pieceDic, j)
            pieceDic[j] = [val, 0]
        else
            pieceDic[j][1]+=val
        end

        if i!=ipre || j!=jpre
            id = indDic[(i,j)]
            if gama[id] == 0
                rankVec = [n/2 for i_=1:n]
            else
                tempVec = D*gama[id]*P_first[j]
                for x in keys(uniq[id])
                    tempVec[x] += (uniq[id][x]-D)/totalCount[id]
                end
                rankVec = n+1-sortperm(sortperm(tempVec))
            end
        end
        ipre, jpre = i,j
        if rankVec[k] <= topVal
            pieceDic[j][2] += val
        end
    end
    x = []; y = []
    for i = 1:length(perm)
        ind = perm[i]
        if !haskey(pieceDic, ind)
            continue
        end
        push!(x, pieceDic[ind][1])
        push!(y, pieceDic[ind][2])
    end
    return x,y
end
