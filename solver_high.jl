include("util.jl")

function high_solver_rhomp(alpha, maxId, Prob, splitVec, order, filePath)
    for d = 3:order
        println("solving higher-order model with d = $(d)")
        ind = 0
        numDic1 = Dict()
        f = open(filePath)
        for ln in eachline(f)
            ind += 1
            if splitVec[ind] == false
                continue
            end
            tempArray = readdlm(IOBuffer(ln),Int)
            n = length(tempArray)
            if n < d+1
                continue
            end
            for i = d+1:n
                tempTuple = tuple([tempArray[k] for k = i-d:i]...)
                if haskey(numDic1, tempTuple)
                    numDic1[tempTuple] += 1
                else
                    numDic1[tempTuple] = 1
                end
            end
        end
        close(f)
        P = init_matrix_rhomp_high(numDic1, maxId, d)
        oldlogLi = compute_likelihood_rhomp_high(numDic1, P, alpha)
        println("inititial likelihood is $(oldlogLi)")
        newlogLi = oldlogLi
        stepSize = STEPSIZEINIT;
        for i = 1:MAXITER
            PCopy = [ P[i_][:,:] for i_ = 1:d ]
            flag = true
            stepSize = min(STEPSIZEINIT, stepSize*2);
            while flag || newlogLi < oldlogLi
                flag = false
                if newlogLi < oldlogLi
                    stepSize *= 0.5
                end
                if stepSize < STEPSIZEMIN
                    break
                end
                P = [ PCopy[i_][:,:] for i_ = 1:d ]
                GP = [ spones(P[i_])*SMALLNUM for i_=1:d ]
                update_gradient_high(numDic1, P, GP, alpha)
                for i_ = 1:d
                    projection(P[i_], GP[i_], stepSize, SMALLNUM)
                end
                newlogLi = compute_likelihood_rhomp_high(numDic1, P, alpha)
            end
            if i%20 == 0
                println("iter $(i) likelihood is $(newlogLi) ---- the diff is $(newlogLi - oldlogLi) --- stepSize is $(stepSize)")
            end
            if abs( (newlogLi - oldlogLi)/oldlogLi ) < EPSILON
                break
            end
            oldlogLi = newlogLi;
        end
        println("\n\n *********** $(newlogLi) ***********\n\n")
        push!(Prob, P)
    end
    for d = 2:order
        r = alpha/(1-alpha)
        a = 1/sum( [r^i_ for i_ = 0:d-1] )
        for i = 1:d
            Prob[d-1][i] = a*r^(i-1)*Prob[d-1][i]
        end
    end
    return Prob
end


function high_solver_mc(Prob, dicts, splitVec, order, filePath)
    for d = 3:order
        println("solving higher-order model with d = $(d)")
        ind = 0
        numDic1 = Dict()
        f = open(filePath)
        for ln in eachline(f)
            ind += 1
            if splitVec[ind] == false
                continue
            end
            tempArray = readdlm(IOBuffer(ln),Int)
            n = length(tempArray)
            if n < d+1
                continue
            end
            for i = d+1:n
                tempTuple = tuple([tempArray[k] for k = i-d:i]...)
                if haskey(numDic1, tempTuple)
                    numDic1[tempTuple] += 1
                else
                    numDic1[tempTuple] = 1
                end
            end
        end
        close(f)
        P, indDic1 = init_matrix_highmc(numDic1, d);
        push!(Prob, P);
        push!(dicts, indDic1);
    end
    return Prob
end

function high_slover_kn(filePath, splitRatio, order)
    numDic1, numDic2, maxId, splitVec = init_dict_second(filePath, splitRatio)
    dist = zeros(maxId); n1 = 0; n2 = 0
    gama = zeros(maxId); totalCount = zeros(maxId)
    for key in keys(numDic1)
        dist[key[3]] += numDic1[key]
        gama[key[2]] += 1
        totalCount[key[2]] += numDic1[key]
        if numDic1[key] == 1
            n1+=1
        elseif numDic1[key] == 2
            n2 += 1
        end
    end
    D = n1/(n1+2*n2)
    dist = dist/sum(dist)
    gama = D*max(1,gama)./max(1, totalCount)

    dicLen = length(numDic1)
    I, J, V  = zeros(dicLen), zeros(dicLen), zeros(dicLen)
    index = 0
    for key in keys(numDic1)
        index += 1
        I[index] = key[2]
        J[index] = key[3]
        V[index] = numDic1[key] - D
    end
    I, J = round(Int64,I), round(Int64,J)
    P_first = sparse(J, I, V, maxId, maxId)
    sumP = zeros(1,size(P_first,2))
    for key in keys(numDic1)
        sumP[key[2]] +=  numDic1[key]
    end

    for i=1:length(sumP)
        if sumP[i] == 0
            sumP[i] = 1
        else
            sumP[i] = 1/sumP[i]
        end
    end

    P_first = P_first.*sumP


    Prob = []; gamas = []; dicts = []
    for d = 2:order
        println("solving higher-order model with d = $(d)")
        ind = 0
        numDic = Dict()
        f = open(filePath)
        for ln in eachline(f)
            ind += 1
            if splitVec[ind] == false
                continue
            end
            tempArray = readdlm(IOBuffer(ln),Int)
            n = length(tempArray)
            if n < d+1
                continue
            end
            for i = d+1:n
                tempTuple = tuple([tempArray[k] for k = i-d:i]...)
                if haskey(numDic, tempTuple)
                    numDic[tempTuple] += 1
                else
                    numDic[tempTuple] = 1
                end
            end
        end
        close(f)
        P, indDic1, newgama = init_high_kneser_ney(numDic, d, maxId, D)
        push!(Prob, P);
        push!(dicts, indDic1);
        push!(gamas, newgama);
    end
    return numDic1, numDic2, Prob, dicts, gamas, dist+zeros(length(dist),1), P_first, gama, splitVec
end

