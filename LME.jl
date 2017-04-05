function process_LME_file(filePath)
    f = open(filePath)
    x = readline(f)
    x = readline(f)
    n = int(split(x)[2])
    x = readline(f)
    d = int(split(x)[2])
    readline(f);readline(f);readline(f);readline(f);readline(f)
    points = []
    while !eof(f)
        x = readline(f)
        if x[1]=='='
            break
        end
        items = split(x)
        v = zeros(d)
        for i = 1:d
            v[i] = float(items[i])
        end
        push!(points,v)
    end

    order = []
    prob = []
    for i = 1:n
        v = zeros(n)
        tempTotal = 0
        for j = 1:n
            if i==j
                v[i] = 0
            else
                for k = 1:d
                    v[j] += (points[i][k] - points[j][k])^2
                end
                v[j] = exp(-v[j])
                tempTotal += v[j]
            end
        end
        push!(order, n+1-sortperm(sortperm(v)))
        push!(prob, v/tempTotal)
    end
    return prob, order, n
end

function precision_LME(filePath, order, topVal)
    m = length(topVal)
    res = zeros(m); total = 0
    f = open(filePath)
    readline(f);readline(f)
    while !eof(f)
        x = readline(f)
        items = split(x)
        for i = 1:length(items)-1
            current, next = int(items[i])+1, int(items[i+1])+1
            for ii = 1:m
                if order[current][next] <= topVal[ii]
                    res[ii] += 1
                end
            end
            total += 1
        end
    end
    return res/total
end

function mrr_LME(filePath, order)
    res = 0; total = 0
    f = open(filePath)
    readline(f);readline(f)
    while !eof(f)
        x = readline(f)
        items = split(x)
        for i = 1:length(items)-1
            current, next = int(items[i])+1, int(items[i+1])+1
            res += 1/order[current][next]
            total += 1
        end
    end
    return res/total
end