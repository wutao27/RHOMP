include("initialization.jl")
include("evaluation_second.jl")
include("evaluation_high.jl")

function chebyshev_nodes(n)
    return [0.5+0.5*cos( ( (2*i - 1)/(2*n) )*pi ) for i = 1:n]
end

function inter_point{T<:Number,N}(x::Vector{T}, y::AbstractArray{T,N}, x0::T)

    if length(x)!=size(y,1)
        error("x and y have to be of the same size.")
    end

    n = length(x)
    p = zero(T)

    for i=1:n
        Li =one(T)
        dLi=zero(T)
        for j=1:n
            if j==i
                continue
            else
                Li*=(x0-x[j])/(x[i]-x[j])
            end
        end
        p.+=Li.*y[i, [1:size(y,n) for n=2:N]...]
    end
    return p
end

function max_point(stepLength, x, y)
    maxInd = 0;maxVal = inter_point(x, y, 0.0)
    for i=0:stepLength:1
        val = inter_point(x, y, i)
        if val>maxVal
            maxVal = val
            maxInd = i
        end
    end
    return maxInd
end

function update_gradient(numDic, Q, P, GQ, GP, alpha)
    n = size(Q,1)
    for key in keys(numDic)
        i,j,k = key; val = numDic[key]
        if i==-1
            i = n
        end
        divisor =  alpha*Q[k,i] + (1-alpha)*P[k,j]
        GQ[k,i] += ( alpha*val ) / divisor
        GP[k,j] += ( (1-alpha)*val ) / divisor
    end
end


function update_gradient_high(numDic1, P, GP, alpha)
    d = length(P)
    r = alpha/(1-alpha)
    a = 1/ sum([r^i_ for i_ = 0:d-1])
    for key in keys(numDic1)
        val = numDic1[key]
        divisor = 0
        for i = 1:d
            divisor += a*r^(i-1)*P[i][ key[d+1],key[d-i+1] ]
        end
        for i = 1:d
            GP[i][ key[d+1],key[d-i+1] ] += val*a*r^(i-1)/divisor
        end
    end
end

function projection(P, GP, stepSize, sm)
    n = P.n
    for i=1:n
        col = P[:,i]
        if nnz(col) == 0
            continue
        end
        tempCol = GP[:,i]
        tempCol = ( stepSize/( maximum(tempCol)*nnz(tempCol) ) ) * tempCol
        col = col + tempCol
        rowval = col.rowval
        nzval = sort(col.nzval, rev = true)
        count = 1; total = nzval[1] - 1; theta = total/count
        while count < length(nzval) && nzval[count] > theta
            count += 1
            total += nzval[count]
            theta = total/count
        end
        
        if nzval[count] < theta
            total -= nzval[count]
            count -= 1
            theta = total/count
        end
        for j in rowval
            P[j,i] = max(P[j,i] + tempCol[j,1] - theta, sm)
        end
    end
end     

function compute_likelihood_rhomp(numDic, Q, P, alpha)
    n = size(Q,1)
    logres = 0;
    for key in keys(numDic)
        i,j,k = key; val = numDic[key]
        if i==-1
            i = n
        end
        logres += val*log( alpha*Q[k,i] + (1-alpha)*P[k,j] )
    end
    return logres
end

function compute_likelihood_rhomp_high(numDic1, P, alpha)
    d = length(P)
    r = alpha/(1-alpha)
    a = 1/ sum([r^i_ for i_ = 0:d-1])
    logres = 0
    for key in keys(numDic1)
        val = numDic1[key]
        temp = 0
        for i = 1:d
            temp += a*r^(i-1)*P[i][ key[d+1],key[d-i+1] ]
        end
        logres += val*log(temp)
    end
    return logres
end