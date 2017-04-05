using Distributions
include("initialization.jl")

function PITF(inputData, cate, dict,n, mu, std, alpha, lambda, dim, maxiter)
# input is a dictionary d[(i,j)] = [k1,k2,k3 ...]
# n is the total number of states
    dist = Normal(mu, std);
    U = rand(dist, n+1, dim); I = rand(dist, n+1, dim);
    TU = rand(dist, n+1, dim); TI = rand(dist, n+1, dim);

    for iter = 1:maxiter
        u,i,ta,tb = sam(inputData, cate, dict, n)
        if u == -1
            u = n+1
        end
        y = U[u,:]*TU[ta,:]' + I[i,:]*TI[ta,:]' - U[u,:]*TU[tb,:]' - I[i,:]*TI[tb,:]'
        delta = 1 - 1/(1+exp(-y[1]))
        for k = 1:dim
            U[u,k] = U[u,k] + alpha*( delta*( TU[ta,k] - TU[tb,k] ) - lambda*U[u,k] )
            I[i,k] = I[i,k] + alpha*( delta*( TI[ta,k] - TI[tb,k] ) - lambda*I[i,k] )
            TU[ta,k] = TU[ta,k] + alpha*( delta*U[u,k] - lambda*TU[ta,k] )
            TU[tb,k] = TU[tb,k] + alpha*( -delta*U[u,k] - lambda*TU[tb,k] )
            TI[ta,k] = TI[ta,k] + alpha*( delta*I[i,k] - lambda*TI[ta,k] )
            TI[tb,k] = TI[tb,k] + alpha*( -delta*I[i,k] - lambda*TI[tb,k] )
        end
    end
    return U, I, TU, TI
end

function sam(inputData, cate, dict, n)
    # ---------- sampling function --------
    ind = rand(cate)
    u,i = dict[ind]
    ta = rand(inputData[(u,i)])
    tb = -2
    while true
        tb = rand(1:n)
        if !(tb in inputData[(u,i)])
            break
        end
    end
    return u,i,ta,tb
end

function init_ritf(filePath, splitRatio)
    numDic1, numDic2, maxId, splitVec = init_dict_second(filePath, splitRatio)
    inputData = Dict()
    for key in keys(numDic1)
        u,i,t = key
        if !haskey(inputData, (u,i))
            inputData[(u,i)] = [t]
        else
            push!(inputData[(u,i)],t)
        end
    end
    tempCate = Array(Int64,0); dict = [];
    for key in keys(inputData)
        push!(dict, key)
        l = length(inputData[key])
        push!(tempCate, l*(maxId-l))
    end
    cate = Categorical(tempCate/sum(tempCate))
    return inputData, cate, dict, numDic1, numDic2, maxId
end

function compute_rank(U, I, TU, TI, numDic)
    n = size(U,1) - 1
    m = round(Int,length(numDic)/100)
    rankDic = Dict()
    for key in keys(numDic)
        u,i,t = key
        if u == -1
            u = n+1
        end
        if !haskey(rankDic,(u,i))
            y = U[u,:]*TU' + I[i,:]*TI'
            y = y[:]
            rankDic[(u,i)] = n+2-sortperm(sortperm(y))
        end
    end
    return rankDic
end

function precision_ritf(numDic, rankDic, topVal)
    m = length(topVal)
    n = size(U,1) - 1
    res = zeros(m); total = 0
    for key in keys(numDic)
        u,i,t = key
        if u == -1
            u = n+1
        end
        val = numDic[key]
        total += val
        rank = rankDic[(u,i)]
        for ii = 1:m
            if rank[t] <= topVal[ii]
                res[ii]+=val
            end
        end
    end
    return res/total
end

function mrr_ritf(numDic, rankDic)
    n = size(U,1) - 1
    res = 0; total = 0
    for key in keys(numDic)
        u,i,t = key
        if u == -1
            u = n+1
        end
        val = numDic[key]
        total += val
        rank = rankDic[(u,i)]
        res += val*(1/rank[t])
    end
    return res/total
end
