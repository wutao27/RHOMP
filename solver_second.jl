include("util.jl")

function one_solve(numDic1, maxId, alpha)
    Q, P = init_matrix_rhomp(numDic1, maxId);
    # flag = false; learnRate = 1
    oldlogLi = compute_likelihood_rhomp(numDic1, Q, P, alpha);
    println("inititial likelihood is $(oldlogLi)")
    newlogLi = oldlogLi;
    stepSize = STEPSIZEINIT;
    
    # tempCount = 0
    for i=1:MAXITER
        QCopy, PCopy = Q[:,:], P[:,:];
        flag = true
        stepSize = min(STEPSIZEINIT, stepSize*2);

        while flag || newlogLi < oldlogLi
            flag = false
            if newlogLi < oldlogLi
                stepSize *= 0.5;
            end
            if stepSize < STEPSIZEMIN
                return Q, P, newlogLi
            end
            Q, P = QCopy[:,:], PCopy[:,:]
            GQ = spones(Q)*SMALLNUM;
            GP = spones(P)*SMALLNUM;
            update_gradient(numDic1, Q, P, GQ, GP, alpha);
            projection(Q, GQ, stepSize, SMALLNUM);
            projection(P, GP, stepSize, SMALLNUM);
            newlogLi = compute_likelihood_rhomp(numDic1, Q, P, alpha)
        end
        if i%20 == 0
            println("iter $(i) likelihood is $(newlogLi) ---- the diff is $(newlogLi - oldlogLi) --- stepSize is $(stepSize)")
        end
        if abs( (newlogLi - oldlogLi)/oldlogLi ) < EPSILON
            return Q, P, newlogLi
        end
        oldlogLi = newlogLi;
    end
    println("\n\n *********** $(newlogLi) ***********\n\n")
    return Q, P, newlogLi
end

function overall_solve(numDic1, maxId, n, stepLength)
    logres = zeros(n)
    alphaList = chebyshev_nodes(n)
    for i = 1:n
        println("---------- solve alpha = $(alphaList[i])")
        Q, P, logres[i] = one_solve(numDic1, maxId, alphaList[i])
    end
    ind = max_point(stepLength, alphaList, logres)
    Q, P, likelihood = one_solve(numDic1, maxId, ind)
    return Q, P, ind
end
