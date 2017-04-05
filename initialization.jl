# define constant
SMALLNUM = 10.0^(-20)
MAXITER = 1000
STEPSIZEINIT = 1
STEPSIZEMIN = 1e-3
EPSILON = 1e-5

function init_dict_second(filePath, splitRatio)
    # ----- function returns the training and testing dataset for the sequence ------
    # filePath: the path of the input sequence file
    # splitRatio: the ratio of traning data size
    numDic1 = Dict(); numDic2 = Dict()
    maxId = 0
    splitVec = []
    f = open(filePath)
    for ln in eachline(f)
        tempArray = readdlm(IOBuffer(ln),Int)
        n = length(tempArray)
        if n<2
            continue
        end
        maxId = max(maxId,maximum(tempArray))
        ran = rand()
        if ran <= splitRatio
            push!(splitVec, true)
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
            push!(splitVec, false)
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
    # numDic1: the dictionary recording the counts of i->j->k, for training data
    # numDic2: the dictionary recording the counts of i->j->k, for testing data
    # maxId: the maximum index number, Note the total number of states is maxId + 1 (including the RESET state)
    # splitVec: the boolean vector indicate the the sequence from the input file belong to training set
    return numDic1, numDic2, maxId, splitVec
end

function init_from_split(splitVec, filePath)
    # ----- function returns the training and testing dataset for the sequence ------
    # splitVec: the boolean vector indicate the the sequence from the input file belong to training set
    # filePath: the path of the input sequence file
    numDic1 = Dict(); numDic2 = Dict(); ind = 0;
    f = open(filePath)
    for ln in eachline(f)
        ind += 1;
        tempArray = readdlm(IOBuffer(ln),Int);
        n = length(tempArray);
        if n<2
            continue
        end
        if splitVec[ind]
            if haskey( numDic1, (-1,tempArray[1],tempArray[2]) )
                numDic1[(-1,tempArray[1],tempArray[2])]+=1;
            else
                numDic1[(-1,tempArray[1],tempArray[2])]=1;
            end
            for i = 3:n
                tempTuple = (tempArray[i-2],tempArray[i-1],tempArray[i]);
                if haskey(numDic1, tempTuple) numDic1[tempTuple]+=1;
                else numDic1[tempTuple] = 1;
                end
            end
        else
            if haskey( numDic2, (-1,tempArray[1],tempArray[2]) ) numDic2[(-1,tempArray[1],tempArray[2])]+=1;
            else numDic2[(-1,tempArray[1],tempArray[2])]=1;
            end
            for i = 3:n
                tempTuple = (tempArray[i-2],tempArray[i-1],tempArray[i]);
                if haskey(numDic2, tempTuple) numDic2[tempTuple]+=1;
                else numDic2[tempTuple] = 1;
                end
            end
        end         
    end
    return numDic1, numDic2
    # numDic1: the dictionary recording the counts of i->j->k, for training data
    # numDic2: the dictionary recording the counts of i->j->k, for testing data
end

function init_matrix_rhomp(numDic1, maxId)
    # -----  initialization for the RHOMP model (with order = 2) --------
    # numDic1: the dictionary recording the counts of i->j->k, for training data
    # maxId: the maximum index number
    dicLen = length(numDic1)
    I, J, K, V  = zeros(dicLen), zeros(dicLen), zeros(dicLen), zeros(dicLen)
    
    index = 1
    for key in keys(numDic1)
        I[index],J[index],K[index] = key
        V[index] = numDic1[key]
        if I[index] == -1
            I[index] = maxId + 1
        end
        index += 1
    end

    I, J, K = round(Int64,I), round(Int64,J), round(Int64,K) 
            
    Q = sparse(K,I,V,maxId+1,maxId+1); P = sparse(K,J,V,maxId+1,maxId+1)
    sumQ = sum(Q,1); sumP = sum(P,1)
    for i=1:maxId+1
        if sumQ[i] == 0
            sumQ[i] = 1
        else
            sumQ[i] = 1/sumQ[i]
        end
        if sumP[i] == 0
            sumP[i] = 1
        else
            sumP[i] = 1/sumP[i]
        end
    end
    Q = Q.*sumQ; P = P.*sumP
    # Q and P are the transition matrices the RHOMP model (order = 2)
    return Q, P
end


function init_matrix_second(numDic1)
    # -----  initialization for the second-order MC model (order = 2) --------
    # numDic1: the dictionary recording the counts of i->j->k, for training data
    dicLen = length(numDic1)
    I, J, V  = zeros(dicLen), zeros(dicLen), zeros(dicLen)

    indDic1 = Dict(); indDic2 = Dict()
    ind = 1; index = 1
    for key in keys(numDic1)
        i,j,k = key
        if !haskey(indDic1, (i,j))
            indDic1[(i,j)] = ind
            indDic2[ind] = (i,j)
            ind+=1
        end
        if !haskey(indDic1, (j,k))
            indDic1[(j,k)] = ind
            indDic2[ind] = (j,k)
            ind+=1
        end
        I[index] = indDic1[(i,j)]; J[index] = indDic1[(j,k)]
        V[index] = numDic1[key]
        index+=1
    end

    I, J = round(Int64,I), round(Int64,J)
    P = sparse(J,I,V,ind,ind)
    sumP = sum(P,1)
    for i=1:ind
        if sumP[i] == 0
            sumP[i] = 1
        else
            sumP[i] = 1/sumP[i]
        end
    end

    P = P.*sumP
    # P is the Markov transition matrix
    # indDic1 is the map from index in P to paired states in the orignal state space
    return P, indDic1
end


function init_matrix_rhomp_high(numDic1, maxId, d)
    # -----  initialization for the RHOMP model (order = d) --------
    # numDic1: the dictionary for training data
    # maxId: the maximum index number
    # d: the order of the RHOMP model
    dicLen = length(numDic1)
    IJK = [ zeros(dicLen) for i_=1:d+1 ]
    V = zeros(dicLen)
    index = 1
    for key in keys(numDic1)
        for ind = 1:d+1
            IJK[ind][index] = key[ind]
        end
        V[index] = numDic1[key]
        index += 1
    end
    IJK = [ round(Int64,IJK[i_]) for i_ = 1:d+1 ]

    P = [ sparse(IJK[d+1], IJK[d - i_ + 1], V, maxId, maxId) for i_ = 1:d ]
    sumP = [ sum(P[i_],1) for i_ = 1:d ]
    for i = 1:maxId
        for j = 1:d
            if sumP[j][i] == 0
                sumP[j][i] = 1
            else
                sumP[j][i] = 1/sumP[j][i]
            end
        end
    end
    P = [ P[i_].*sumP[i_] for i_ = 1:d ]
    # P is the list of transition matrices for the RHOMP model.
    return P
end

function init_matrix_highmc(numDic1, d)
    # -----  initialization for the higher-order MC model (order = d) --------
    # numDic1: the dictionary for training data
    # d: the order of the MC model
    dicLen = length(numDic1)
    I, J, V  = zeros(dicLen), zeros(dicLen), zeros(dicLen)

    indDic1 = Dict();
    ind = 1; index = 1
    for key in keys(numDic1)
        tempKey1 = key[1:d]
        tempKey2 = key[2:d+1]
        if !haskey(indDic1, tempKey1)
            indDic1[tempKey1] = ind
            ind+=1
        end
        if !haskey(indDic1, tempKey2)
            indDic1[tempKey2] = ind
            ind+=1
        end
        I[index] = indDic1[tempKey1]; J[index] = indDic1[tempKey2]
        V[index] = numDic1[key]
        index+=1
    end

    I, J = round(Int64,I), round(Int64,J)
    P = sparse(J,I,V,ind,ind)
    sumP = sum(P,1)
    for i=1:ind
        if sumP[i] == 0
            sumP[i] = 1
        else
            sumP[i] = 1/sumP[i]
        end
    end

    P = P.*sumP
    # P is the Markov transition matrix
    # indDic1 is the map from index in P to paired states in the orignal state space
    return P, indDic1
end


function init_high_kneser_ney(numDic1, d, maxId, D)
    # -----  initialization for the Kneser Ney model (order = d) --------
    # numDic1: the dictionary for training data
    # d: the order of the MC model
    # maxId: the maximum index number
    # D: model parameter (precomputed)
    dicLen = length(numDic1)
    I, J, V  = zeros(dicLen), zeros(dicLen), zeros(dicLen)

    indDic1 = Dict();
    ind = 1; index = 1
    for key in keys(numDic1)
        tempKey1 = key[1:d]
        if !haskey(indDic1, tempKey1)
            indDic1[tempKey1] = ind
            ind+=1
        end
        I[index] = indDic1[tempKey1]; J[index] = key[d+1]
        V[index] = numDic1[key] - D
        index+=1
    end

    I, J = round(Int64,I), round(Int64,J)
    P = sparse(J,I,V,maxId,ind)
    sumP = zeros(1,size(P,2))
    for key in keys(numDic1)
        tempKey1 = key[1:d]
        sumP[indDic1[tempKey1]] += numDic1[key]
    end

    for i=1:length(sumP)
        if sumP[i] == 0
            sumP[i] = 1
        else
            sumP[i] = 1/sumP[i]
        end
    end

    P = P.*sumP
    gama = zeros(ind); totalCount = zeros(ind)
    for key in keys(numDic1)
        st, en = indDic1[key[1:d]], key[d+1]
        totalCount[st] += numDic1[key]
        gama[st] += 1
    end
    gama = max(1,gama)./max(1, totalCount)
    # P is the Markov transition matrix
    # indDic1 is the map from index in P to paired states in the orignal state space
    return P, indDic1, D*gama
end


