function precision_rhomp(numDic, Q, P, alpha, topVal)
    QQ = alpha*Q; PP = (1-alpha)*P
    n = size(Q,1)
    m = length(topVal)
    res = zeros(m); total = 0
    ipre = -1; jpre = -1
    tempCol = QQ[:,1] + PP[:,1]
    # for key in keys(numDic)
    for key in sort(collect(keys(numDic)))
        i,j,k = key
        if i==-1
            i = n
        end
        val = numDic[key]
        total += val
        if QQ[k,i] + PP[k,j] <= SMALLNUM
            continue
        end
        if i!=ipre || j!=jpre
            tempCol = QQ[:,i] + PP[:,j]
            ipre = i; jpre = j
        end
        nzval = sort(tempCol.nzval, rev = true)
        for ii = 1:m
            if tempCol[j,1] >= tempCol[k,1] && (length(nzval)<topVal[ii]+1 || tempCol[k,1]>=nzval[topVal[ii]+1])
                res[ii]+=val
            elseif length(nzval)<topVal[ii] || tempCol[k,1]>=nzval[topVal[ii]]
                res[ii]+=val
            end
        end
    end
    return res/total
end


function precision_first(numDic, P, topVal)
    m = length(topVal)
    res = zeros(m); total = 0
    for key in keys(numDic)
        i,j,k = key
        val = numDic[key]
        total += val
        tempCol = P[:,j]
        if tempCol[k,1] == 0
            continue
        end
        nzval = sort(tempCol.nzval, rev = true)
        for ii = 1:m
            if length(nzval)<topVal[ii] || tempCol[k,1]>=nzval[topVal[ii]]
                res[ii]+=val
            end
        end
    end
    return res/total
end


function precision_second(numDic, P, topVal, indDic1)
    m = length(topVal)
    res = zeros(m); total = 0
    for key in keys(numDic)
        i,j,k = key
        val = numDic[key]
        total += val
        if !haskey(indDic1,(i,j)) || !haskey(indDic1,(j,k))
            continue
        end
        index1 = indDic1[(i,j)]
        index2 = indDic1[(j,k)]
        tempCol = P[:,index1]
        if tempCol[index2,1] == 0
            continue
        end
        nzval = sort(tempCol.nzval, rev = true)
        for ii = 1:m
            if length(nzval)<topVal[ii] || tempCol[index2,1]>=nzval[topVal[ii]]
                res[ii]+=val
            end
        end
    end
    return res/total
end


function mrr_rhomp(numDic, Q, P, alpha)
    res = 0; total = 0
    n = size(Q,1)
    for key in keys(numDic)
        i,j,k = key
        if i==-1
            i = n
        end
        val = numDic[key]
        total += val
        tempCol = alpha*Q[:,i] + (1-alpha)*P[:,j]
        if tempCol[k,1] <= SMALLNUM
            res += val/((nnz(tempCol) + n)/2)
            continue
        end
        nzval = sort(tempCol.nzval, rev = true)
        nzind = 1; flag = false
        while nzind<=length(nzval)
            if nzval[nzind] == tempCol[k,1]
                count = 1; tempVal = nzind
                nzind+=1
                while nzind<length(nzval) && nzval[nzind] == tempCol[k,1]
                    count+=1; tempVal+=nzind; nzind+=1
                end
                if tempCol[j,1] >= tempCol[k,1] && j!=k
                    res += val/(tempVal/count - 1)
                else
                    res += val/(tempVal/count)
                end
                flag = true
                break
            end
            nzind+=1
        end
        if flag == false
            error("in rank_rhomp no value found")
        end
    end
    return res/total
end


function mrr_first(numDic, P)
    res = 0; total = 0
    n = size(P,1)
    for key in keys(numDic)
        i,j,k = key
        val = numDic[key]
        total += val
        tempCol = P[:,j]
        if tempCol[k,1] == 0
            res += val/((nnz(tempCol) + n)/2)
            continue
        end
        nzval = sort(tempCol.nzval, rev = true)
        nzind = 1; flag = false
        while nzind<=length(nzval)
            if nzval[nzind] == tempCol[k,1]
                count = 1; tempVal = nzind
                nzind+=1
                while nzind<length(nzval) && nzval[nzind] == tempCol[k,1]
                    count+=1; tempVal+=nzind; nzind+=1
                end
                res += val/(tempVal/count)
                flag = true
                break
            end
            nzind+=1
        end
        if flag == false
            error("in rank_rhomp no value found")
        end
    end
    return res/total
end


function mrr_second(numDic, P, indDic1, n)
    res = 0; total = 0
    for key in keys(numDic)
        i,j,k = key
        val = numDic[key]
        total += val
        if !haskey(indDic1,(i,j))
            res += val/(n/2)
            continue
        end
        index1 = indDic1[(i,j)]
        tempCol = P[:,index1]

        if !haskey(indDic1,(j,k))
            res += val/((nnz(tempCol) + n)/2)
            continue
        end
        index2 = indDic1[(j,k)]
        if tempCol[index2,1] == 0
            res += val/((nnz(tempCol) + n)/2)
            continue
        end
        nzval = sort(tempCol.nzval, rev = true)
        nzind = 1; flag = false
        while nzind<=length(nzval)
            if nzval[nzind] == tempCol[index2,1]
                count = 1; tempVal = nzind
                nzind+=1
                while nzind<length(nzval) && nzval[nzind] == tempCol[index2,1]
                    count+=1; tempVal+=nzind; nzind+=1
                end
                res += val/(tempVal/count)
                flag = true
                break
            end
            nzind+=1
        end
        if flag == false
            error("in rank_rhomp no value found")
        end
    end
    return res/total
end

function compute_count(numDic1, maxId)
    countArray = zeros(maxId+1)

    for key in keys(numDic1)
        i,j,k = key
        countArray[j]+=numDic1[key]
    end
    
    return countArray
end

function pr_piece_first(countArray, numDic, P, topVal)
    pieceDic = Dict()
    perm = sortperm(countArray,rev = true)
    for key in keys(numDic)
        i,j,k = key
        val = numDic[key]
        if !haskey(pieceDic, j)
            pieceDic[j] = [val, 0]
        else
            pieceDic[j][1]+=val
        end

        tempCol = P[:,j]
        if tempCol[k,1] == 0
            continue
        end
        nzval = sort(tempCol.nzval, rev = true)
        if length(nzval)<topVal || tempCol[k,1]>=nzval[topVal]
            pieceDic[j][2]+=val
        end
    end
    x = []; y=[]
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

function pr_piece_second(countArray, numDic, P, indDic1, topVal)
    pieceDic = Dict()
    perm = sortperm(countArray,rev = true)
    for key in keys(numDic)
        i,j,k = key
        val = numDic[key]
        if !haskey(pieceDic, j)
            pieceDic[j] = [val, 0]
        else
            pieceDic[j][1]+=val
        end
        # total += val
        if !haskey(indDic1,(i,j)) || !haskey(indDic1,(j,k))
            continue
        end
        index1 = indDic1[(i,j)]
        index2 = indDic1[(j,k)]
        tempCol = P[:,index1]
        if tempCol[index2,1] == 0
            continue
        end
        nzval = sort(tempCol.nzval, rev = true)
        if length(nzval)<topVal || tempCol[index2,1]>=nzval[topVal]
            pieceDic[j][2]+=val
        end
    end

    x = []; y=[]
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

function pr_piece_rhomp(countArray, numDic, Q, P, alpha, topVal)
    n = size(Q,1)
    pieceDic = Dict()
    perm = sortperm(countArray,rev = true)
    for key in keys(numDic)
        i,j,k = key
        if i==-1
            i = n
        end
        val = numDic[key]
        if !haskey(pieceDic, j)
            pieceDic[j] = [val, 0]
        else
            pieceDic[j][1]+=val
        end

        if alpha*Q[k,i] + (1-alpha)*P[k,j] <= SMALLNUM
            continue
        end
        tempCol = alpha*Q[:,i] + (1-alpha)*P[:,j]
        nzval = sort(tempCol.nzval, rev = true)
        if tempCol[j,1] >= tempCol[k,1] && (length(nzval)<topVal+1 || tempCol[k,1]>=nzval[topVal+1])
            pieceDic[j][2]+=val
            continue
        end
        if length(nzval)<topVal || tempCol[k,1]>=nzval[topVal]
            pieceDic[j][2]+=val
        end

    end
    x = []; y=[]
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